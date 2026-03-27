# DoomAgent: RL Agent for Doom on RV32 + LiteRT Micro

## Project Overview

Train a reinforcement learning agent to play Doom (E1M1 initially), export it as an INT8 TFLite model runnable under LiteRT Micro, and integrate it with `doomgeneric` for host-CPU verification. Device deployment (RV32IMF_Zve32f_Zve32x, VLEN=128) is a future exercise.

---

## 1. Architecture

### Observation Space (Hybrid)

**Visual input:**
- Source: doomgeneric framebuffer (320×200 RGBA)
- Preprocessing: grayscale → downsample to 60×45 → stack 4 consecutive frames
- Tensor shape: `(4, 45, 60)` channels-first during training, `(45, 60, 4)` NHWC for TFLite

**Game state vector** (extracted from doomgeneric internals):
| Feature               | Source                          | Encoding       |
|-----------------------|---------------------------------|----------------|
| Health                | `player->health`                | float [0, 1]   |
| Armor                 | `player->armorpoints`           | float [0, 1]   |
| Ammo (current weapon) | `player->ammo[weaponinfo[].ammo]` | float [0, 1] |
| Ammo (all 4 types)    | `player->ammo[0..3]`           | 4× float [0,1] |
| Current weapon        | `player->readyweapon`          | one-hot (9)    |
| Kill count delta      | frame-over-frame `killcount`   | float           |
| Velocity XY           | `player->mo->momx/momy`        | 2× float norm  |
| On ground             | `player->mo->z == floorz`      | binary          |

Total game state vector: ~20 floats.

### Action Space

Discrete multi-binary — 6 independent binary actions per tick:
| Action         | doomgeneric key     |
|----------------|---------------------|
| Move forward   | `KEY_UPARROW`       |
| Move backward  | `KEY_DOWNARROW`     |
| Turn left      | `KEY_LEFTARROW`     |
| Turn right     | `KEY_RIGHTARROW`    |
| Shoot          | `KEY_FIRE` (Ctrl)   |
| Use / Open     | `KEY_USE` (Space)   |

Weapon switching omitted initially (agent learns to use whatever it picks up).
Action repeat = 4 tics (≈ 8.6 actions/sec at 35 tic/sec game speed).

### Model Architecture

```
                    [60×45×4 frames]          [20-dim state vec]
                          │                         │
                    DepthwiseSepConv2D 3×3/2, 16    │
                    → BatchNorm → ReLU              │
                          │                         │
                    DepthwiseSepConv2D 3×3/2, 32    │
                    → BatchNorm → ReLU              │
                          │                         │
                    DepthwiseSepConv2D 3×3/2, 32    │
                    → BatchNorm → ReLU              │
                          │                         │
                    Flatten (8×6×32 = 1536)          │
                          │                         │
                          └────── concat ───────────┘
                                   │
                              Dense(256, ReLU)
                                   │
                         ┌─────────┴─────────┐
                    Dense(6, sigmoid)    Dense(1, linear)
                      π (policy)         V (value)
```

**Parameter estimate:**
- Conv layers (depthwise separable): ~5K params
- Flatten→Dense(256): ~400K params (this dominates)
- Policy + value heads: ~1.6K params
- **Total: ~407K params → ~407KB INT8**

Fits comfortably in 1MB ITCM. Activation peak (largest intermediate): 8×6×32 = 1536 bytes INT8 — trivial.

---

## 2. Training Pipeline

### Environment: VizDoom

```
vizdoom==1.2.3  (pip)
stable-baselines3[extra]
gymnasium
```

VizDoom wraps the actual Doom engine and exposes a Gym-compatible API. It provides both screen buffers AND game variables natively — no doomgeneric needed during training.

### VizDoom Scenario Config

Start with a custom scenario based on E1M1:

```ini
# e1m1_agent.cfg
doom_scenario_path = maps/E1M1.wad
doom_map = E1M1

# Screen
screen_resolution = RES_160X120    # VizDoom downsamples for us
screen_format = GRAY8

# Game variables (our state vector)
available_game_variables = {
    HEALTH ARMOR AMMO2 AMMO3 AMMO4 AMMO5
    SELECTED_WEAPON KILLCOUNT
    POSITION_X POSITION_Y
    VELOCITY_X VELOCITY_Y
}

# Actions
available_buttons = {
    MOVE_FORWARD MOVE_BACKWARD
    TURN_LEFT TURN_RIGHT
    ATTACK USE
}

# Episode
episode_timeout = 4200           # 2 min at 35 tics/sec
living_reward = 0
death_penalty = 100
```

### Custom Gym Wrapper

```python
import gymnasium as gym
import numpy as np
import vizdoom as vzd

class DoomHybridEnv(gym.Env):
    """Hybrid observation: downsampled frames + game state vector."""

    def __init__(self, cfg_path="e1m1_agent.cfg", frame_skip=4, stack=4):
        self.game = vzd.DoomGame()
        self.game.load_config(cfg_path)
        self.game.set_window_visible(False)
        self.game.init()

        self.frame_skip = frame_skip
        self.stack = stack
        self.frames = np.zeros((stack, 45, 60), dtype=np.float32)

        n_buttons = self.game.get_available_buttons_size()
        self.action_space = gym.spaces.MultiBinary(n_buttons)

        self.observation_space = gym.spaces.Dict({
            "visual": gym.spaces.Box(0, 1, (stack, 45, 60), np.float32),
            "state":  gym.spaces.Box(-1, 1, (20,), np.float32),
        })

    def _preprocess_frame(self, buf):
        """160×120 GRAY8 → 60×45 float [0,1]"""
        if buf is None:
            return np.zeros((45, 60), dtype=np.float32)
        # Simple area downsample (or use cv2.resize)
        frame = buf[::2, ::2].astype(np.float32) / 255.0  # ~80×60
        # Crop/resize to exact 60×45
        frame = frame[:45, :60]
        return frame

    def _get_state_vec(self):
        gv = self.game.get_game_variable
        v = np.zeros(20, dtype=np.float32)
        v[0] = gv(vzd.GameVariable.HEALTH) / 200.0
        v[1] = gv(vzd.GameVariable.ARMOR) / 200.0
        v[2] = gv(vzd.GameVariable.AMMO2) / 50.0   # clip ammo
        v[3] = gv(vzd.GameVariable.AMMO3) / 50.0
        v[4] = gv(vzd.GameVariable.AMMO4) / 50.0
        v[5] = gv(vzd.GameVariable.AMMO5) / 300.0
        sel = int(gv(vzd.GameVariable.SELECTED_WEAPON))
        if 0 <= sel < 9:
            v[6 + sel] = 1.0  # one-hot weapon (indices 6-14)
        v[15] = gv(vzd.GameVariable.VELOCITY_X) / 30.0
        v[16] = gv(vzd.GameVariable.VELOCITY_Y) / 30.0
        # indices 17-19 reserved for extensions
        return np.clip(v, -1, 1)

    def step(self, action):
        reward = self.game.make_action(action.tolist(), self.frame_skip)
        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            frame = self._preprocess_frame(state.screen_buffer)
        else:
            frame = np.zeros((45, 60), dtype=np.float32)

        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = frame

        obs = {"visual": self.frames.copy(), "state": self._get_state_vec()}
        return obs, reward, done, False, {}

    def reset(self, **kwargs):
        self.game.new_episode()
        self.frames[:] = 0
        state = self.game.get_state()
        frame = self._preprocess_frame(state.screen_buffer)
        self.frames[-1] = frame
        return {"visual": self.frames.copy(), "state": self._get_state_vec()}, {}
```

### Reward Shaping

VizDoom's built-in rewards are sparse. Add shaping:

```python
# In step():
shaped_reward = 0.0
shaped_reward += kill_delta * 50.0        # kills are good
shaped_reward += health_delta * 0.5       # health pickups
shaped_reward += ammo_delta * 0.2         # ammo pickups
shaped_reward += item_delta * 5.0         # items/secrets
shaped_reward += movement_magnitude * 0.01 # encourage exploration
shaped_reward -= 0.001                     # tiny time penalty
# death_penalty from config: -100
```

### PPO Training (stable-baselines3)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class DoomFeatureExtractor(BaseFeaturesExtractor):
    """Hybrid CNN + state vector feature extractor."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # Depthwise-separable conv blocks
        self.visual_net = nn.Sequential(
            # Block 1: depthwise + pointwise
            nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=4),
            nn.Conv2d(4, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Block 2
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16),
            nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Block 3
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 45, 60)
            cnn_out_dim = self.visual_net(dummy).shape[1]

        state_dim = observation_space["state"].shape[0]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + state_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        vis = self.visual_net(observations["visual"])
        state = observations["state"]
        return self.fc(torch.cat([vis, state], dim=1))


policy_kwargs = dict(
    features_extractor_class=DoomFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[64], vf=[64]),  # small policy/value heads
)

model = PPO(
    "MultiInputPolicy",
    DoomHybridEnv(),
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tb_doom/",
)

model.learn(total_timesteps=5_000_000)
model.save("doom_agent_ppo")
```

Training estimate: ~5M steps, ~6-12 hours on a single GPU (RTX 3080 class).

### Curriculum (Progressive Difficulty)

| Phase | Scenario | Steps | Goal |
|-------|----------|-------|------|
| 1 | `basic` (single room, 1 enemy) | 500K | Learn to aim + shoot |
| 2 | `deadly_corridor` (hallway, multiple enemies) | 1M | Move + shoot + dodge |
| 3 | `defend_the_center` (360° enemies) | 1M | Spatial awareness |
| 4 | E1M1 full level | 2.5M+ | Navigation + combat |

---

## 3. Export to TFLite (INT8)

### PyTorch → ONNX → TensorFlow → TFLite

The SB3 model is PyTorch. The cleanest LiteRT path:

```python
import torch
import onnx
import onnx_tf
import tensorflow as tf

# 1. Extract the trained policy network (no value head needed for inference)
class InferencePolicy(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.features_extractor = trained_model.policy.features_extractor
        self.pi_net = trained_model.policy.mlp_extractor.policy_net
        self.action_net = trained_model.policy.action_net

    def forward(self, visual, state):
        features = self.features_extractor({"visual": visual, "state": state})
        latent = self.pi_net(features)
        return torch.sigmoid(self.action_net(latent))  # multi-binary probs

inference_model = InferencePolicy(model)
inference_model.eval()

# 2. Export ONNX
dummy_vis = torch.randn(1, 4, 45, 60)
dummy_state = torch.randn(1, 20)
torch.onnx.export(
    inference_model, (dummy_vis, dummy_state),
    "doom_agent.onnx",
    input_names=["visual", "state"],
    output_names=["action_probs"],
    opset_version=13,
)

# 3. ONNX → TF SavedModel
onnx_model = onnx.load("doom_agent.onnx")
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph("doom_agent_tf")

# 4. TFLite INT8 quantization
def representative_dataset():
    """Calibration data from actual gameplay."""
    env = DoomHybridEnv()
    obs, _ = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
        yield [
            obs["visual"][np.newaxis].astype(np.float32),  # NCHW → needs transpose if TF expects NHWC
            obs["state"][np.newaxis].astype(np.float32),
        ]

converter = tf.lite.TFLiteConverter.from_saved_model("doom_agent_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open("doom_agent_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
# Expected: ~400-500 KB
```

### Important: BatchNorm Folding

BatchNorm layers get folded into preceding conv weights during export — no runtime overhead. Verify this happened:

```python
# Check the tflite model has no BatchNorm ops
interpreter = tf.lite.Interpreter(model_path="doom_agent_int8.tflite")
ops = set()
for detail in interpreter._get_ops_details():
    ops.add(detail['op_name'])
print(ops)
# Should see: CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, RESHAPE, CONCATENATION
# Should NOT see: BATCH_NORM, MUL, ADD (from unfused BN)
```

### Alternative: Quantization-Aware Training (QAT)

If PTQ accuracy drops noticeably:
```python
# In PyTorch, before training or during fine-tuning:
import torch.quantization as quant
model_prepared = quant.prepare_qat(inference_model, inplace=False)
# Fine-tune for ~500K more steps with fake-quantized weights
# Then export as above
```

---

## 4. LiteRT Micro Integration with doomgeneric

### doomgeneric Architecture Recap

```
doomgeneric/
├── doomgeneric.h       // Platform API you implement:
│   ├── DG_Init()
│   ├── DG_DrawFrame()      ← we grab pixels here
│   ├── DG_GetKey()         ← we inject agent actions here
│   ├── DG_GetTickCount()
│   └── DG_SetWindowTitle()
├── doomgeneric.c       // DG_ScreenBuffer (uint32_t* RGBA)
└── doomkeys.h          // KEY_UPARROW, KEY_FIRE, etc.
```

### Agent Integration Layer

```c
// doom_agent.h
#pragma once
#include <stdint.h>

// Action bitfield
#define AGENT_ACT_FORWARD   (1 << 0)
#define AGENT_ACT_BACKWARD  (1 << 1)
#define AGENT_ACT_TURN_L    (1 << 2)
#define AGENT_ACT_TURN_R    (1 << 3)
#define AGENT_ACT_FIRE      (1 << 4)
#define AGENT_ACT_USE       (1 << 5)

typedef struct {
    float health;
    float armor;
    float ammo[4];
    float weapon_onehot[9];
    float velocity_xy[2];
    float _reserved[3];
} agent_game_state_t;  // 20 floats

int  agent_init(const char* model_path);
uint8_t agent_infer(const uint32_t* framebuf,   // 320×200 RGBA
                    int fb_width, int fb_height,
                    const agent_game_state_t* gs);
void agent_destroy(void);
```

```c
// doom_agent.c
#include "doom_agent.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Tensor arena — sized for our model's peak activation memory
// CNN activations dominate: ~50KB should be generous
static uint8_t tensor_arena[64 * 1024];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_visual  = nullptr;
static TfLiteTensor* input_state   = nullptr;
static TfLiteTensor* output_action = nullptr;

// Frame buffer for preprocessed stacked frames
static int8_t frame_stack[4][45][60];  // INT8 quantized
static int frame_idx = 0;

static void preprocess_frame(const uint32_t* rgba, int w, int h,
                             int8_t out[45][60]) {
    // RGBA 320×200 → grayscale 60×45, quantized to INT8
    // Downsample with simple area averaging
    const int sx = w / 60;  // ~5.3, use 5
    const int sy = h / 45;  // ~4.4, use 4

    for (int y = 0; y < 45; y++) {
        for (int x = 0; x < 60; x++) {
            int sum = 0;
            for (int dy = 0; dy < sy; dy++) {
                for (int dx = 0; dx < sx; dx++) {
                    int src_y = y * sy + dy;
                    int src_x = x * sx + dx;
                    if (src_y < h && src_x < w) {
                        uint32_t pixel = rgba[src_y * w + src_x];
                        uint8_t r = (pixel >> 16) & 0xFF;
                        uint8_t g = (pixel >> 8)  & 0xFF;
                        uint8_t b =  pixel        & 0xFF;
                        sum += (r * 77 + g * 150 + b * 29) >> 8;
                    }
                }
            }
            uint8_t gray = sum / (sx * sy);
            // Quantize: float [0,1] → INT8 with zero_point=-128, scale=1/255
            out[y][x] = (int8_t)(gray - 128);
        }
    }
}

static void quantize_state(const agent_game_state_t* gs,
                           int8_t out[20],
                           float scale, int32_t zero_point) {
    const float* raw = (const float*)gs;
    for (int i = 0; i < 20; i++) {
        int32_t q = (int32_t)roundf(raw[i] / scale) + zero_point;
        out[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
}

int agent_init(const char* model_path) {
    // In LiteRT Micro, model is typically linked as a const array
    // For host testing, load from file
    extern const unsigned char doom_agent_model[];
    extern const unsigned int doom_agent_model_len;

    const tflite::Model* model = tflite::GetModel(doom_agent_model);

    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddConcatenation();
    resolver.AddLogistic();      // sigmoid
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return -1;
    }

    // Find input/output tensors by index
    // Index 0 = visual (4×45×60 INT8 NHWC), Index 1 = state (20 INT8)
    input_visual  = interpreter->input(0);
    input_state   = interpreter->input(1);
    output_action = interpreter->output(0);

    memset(frame_stack, -128, sizeof(frame_stack));  // zero in quantized space
    return 0;
}

uint8_t agent_infer(const uint32_t* framebuf, int w, int h,
                    const agent_game_state_t* gs) {
    // 1. Preprocess frame into stack
    preprocess_frame(framebuf, w, h, frame_stack[frame_idx % 4]);
    frame_idx++;

    // 2. Copy stacked frames into input tensor (NHWC: 45×60×4)
    int8_t* vis_data = input_visual->data.int8;
    for (int y = 0; y < 45; y++) {
        for (int x = 0; x < 60; x++) {
            for (int c = 0; c < 4; c++) {
                vis_data[y * 60 * 4 + x * 4 + c] = frame_stack[c][y][x];
            }
        }
    }

    // 3. Quantize game state into input tensor
    float state_scale = input_state->params.scale;
    int32_t state_zp  = input_state->params.zero_point;
    quantize_state(gs, input_state->data.int8, state_scale, state_zp);

    // 4. Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        return 0;
    }

    // 5. Decode output → action bitfield
    // Output is 6× INT8 sigmoid probabilities
    uint8_t actions = 0;
    float out_scale = output_action->params.scale;
    int32_t out_zp  = output_action->params.zero_point;

    for (int i = 0; i < 6; i++) {
        float prob = (output_action->data.int8[i] - out_zp) * out_scale;
        if (prob > 0.5f) {
            actions |= (1 << i);
        }
    }

    return actions;
}
```

### Hooking into doomgeneric Platform Layer

```c
// doomgeneric_agent.c — the platform implementation
#include "doomgeneric.h"
#include "doomkeys.h"
#include "doom_agent.h"
#include "d_player.h"    // for player_t access

extern player_t players[];

static uint8_t current_actions = 0;
static int pending_keys[12];
static int n_pending = 0;
static int tick_count = 0;
static int frames_since_infer = 0;

#define INFER_EVERY_N_FRAMES 4   // action repeat

void DG_Init(void) {
    agent_init(NULL);  // model is compiled-in
}

void DG_DrawFrame(void) {
    frames_since_infer++;
    if (frames_since_infer < INFER_EVERY_N_FRAMES) return;
    frames_since_infer = 0;

    // Build game state from engine internals
    agent_game_state_t gs = {0};
    player_t* p = &players[0];
    gs.health = (float)p->health / 200.0f;
    gs.armor  = (float)p->armorpoints / 200.0f;
    for (int i = 0; i < 4; i++)
        gs.ammo[i] = (float)p->ammo[i] / 200.0f;
    int w = p->readyweapon;
    if (w >= 0 && w < 9) gs.weapon_onehot[w] = 1.0f;
    gs.velocity_xy[0] = (float)p->mo->momx / (float)FRACUNIT / 30.0f;
    gs.velocity_xy[1] = (float)p->mo->momy / (float)FRACUNIT / 30.0f;

    // Run inference
    current_actions = agent_infer(
        DG_ScreenBuffer, DOOMGENERIC_RESX, DOOMGENERIC_RESY, &gs);

    // Convert to key events
    n_pending = 0;
    struct { uint8_t mask; unsigned char key; } map[] = {
        {AGENT_ACT_FORWARD,  KEY_UPARROW},
        {AGENT_ACT_BACKWARD, KEY_DOWNARROW},
        {AGENT_ACT_TURN_L,   KEY_LEFTARROW},
        {AGENT_ACT_TURN_R,   KEY_RIGHTARROW},
        {AGENT_ACT_FIRE,     KEY_FIRE},
        {AGENT_ACT_USE,      KEY_USE},
    };
    for (int i = 0; i < 6; i++) {
        pending_keys[n_pending++] = (current_actions & map[i].mask)
            ? map[i].key : -map[i].key;  // negative = key up
    }
}

int DG_GetKey(int* pressed, unsigned char* key) {
    if (n_pending == 0) return 0;
    n_pending--;
    int k = pending_keys[n_pending];
    *pressed = (k > 0) ? 1 : 0;
    *key = (unsigned char)(k > 0 ? k : -k);
    return 1;
}

uint32_t DG_GetTickCount(void) {
    // For host: real wall clock. For device: timer peripheral.
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
```

---

## 5. Host Verification Build

### Directory Structure
```
doom-agent/
├── training/
│   ├── env.py              # DoomHybridEnv
│   ├── train.py            # PPO training script
│   ├── export.py           # PyTorch → ONNX → TFLite
│   └── e1m1_agent.cfg      # VizDoom config
├── inference/
│   ├── doom_agent.h
│   ├── doom_agent.c
│   ├── doomgeneric_agent.c
│   └── CMakeLists.txt
├── doomgeneric/            # git submodule
├── tflite-micro/           # git submodule (or system install)
├── models/
│   └── doom_agent_int8.tflite
└── Makefile
```

### Host Build (x86 Linux, LiteRT Micro)

```makefile
# Top-level Makefile sketch
TFLM_DIR = tflite-micro
DOOM_DIR = doomgeneric/doomgeneric

CC = gcc
CXX = g++
CFLAGS = -O2 -I$(DOOM_DIR) -I$(TFLM_DIR)/include
LDFLAGS = -lm -lpthread -lSDL2

# Build doomgeneric (all .c files except other platform backends)
DOOM_SRC = $(filter-out %_sdl.c %_win.c, $(wildcard $(DOOM_DIR)/*.c))

# Build agent + TFLite Micro
AGENT_SRC = inference/doom_agent.c inference/doomgeneric_agent.c

# TFLite Micro static lib (pre-built or build from source)
TFLM_LIB = $(TFLM_DIR)/lib/libtensorflow-microlite.a

doom_agent: $(DOOM_SRC) $(AGENT_SRC) $(TFLM_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)
```

### Smoke Test

```bash
# After training + export:
cd doom-agent
xxd -i models/doom_agent_int8.tflite > inference/doom_agent_model.c
make
./doom_agent    # Watch it play E1M1!
```

---

## 6. Milestones & Timeline

| # | Milestone | Est. Time | Deliverable |
|---|-----------|-----------|-------------|
| 1 | VizDoom env + basic training loop | 1-2 days | Agent shoots enemies in `basic` scenario |
| 2 | Hybrid obs + curriculum training | 3-5 days | Agent navigates `deadly_corridor` |
| 3 | E1M1 full training | 1-2 days | Agent plays E1M1 (may not complete) |
| 4 | TFLite INT8 export + accuracy check | 1 day | `.tflite` file, INT8 vs FP32 comparison |
| 5 | doomgeneric integration + host build | 2-3 days | Self-playing Doom binary on x86 |
| 6 | Tuning & polish | 2-3 days | Reliable E1M1 completion |

**Total: ~2-3 weeks of evening/weekend work.**

---

## 7. Open Questions & Future Work

- **NCHW vs NHWC:** PyTorch trains NCHW, TFLite prefers NHWC. Need a transpose either in export or in the C preprocessing. Verify the ONNX→TF conversion handles this.
- **Action space refinement:** Multi-binary might cause contradictory actions (forward + backward). Could mask these or switch to a factored discrete space.
- **Turning resolution:** Binary turn L/R is coarse. Could add a "turn amount" continuous action or use 5-way discrete (hard left, left, center, right, hard right) via `TURN_LEFT_RIGHT_DELTA`.
- **Determinism:** doomgeneric uses `I_GetTime()` for RNG seeding. Ensure reproducible demos by fixing the seed.
- **Device deployment:** NHWC layout, INT8 → RVV Zve32x kernel optimization, ITCM/DTCM placement of model weights vs activations, DMA from DDR for frame data.
- **World model variant:** Once the basic agent works, try a latent-space planning model as the "conference talk" version.
