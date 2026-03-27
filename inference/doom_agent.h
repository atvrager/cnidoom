/*
 * doom_agent.h — RL agent inference interface.
 *
 * Runs an INT8 TFLite model via LiteRT Micro to produce actions from
 * the doomgeneric framebuffer and game state.
 *
 * Lifecycle:
 *   agent_init()   → once at startup
 *   agent_infer()  → every N frames (action repeat)
 *   agent_destroy() → cleanup (no-op on bare metal)
 */

#ifndef DOOM_AGENT_H
#define DOOM_AGENT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Action bitfield — one bit per discrete action. */
#define AGENT_ACT_FORWARD (1u << 0)
#define AGENT_ACT_BACKWARD (1u << 1)
#define AGENT_ACT_TURN_L (1u << 2)
#define AGENT_ACT_TURN_R (1u << 3)
#define AGENT_ACT_FIRE (1u << 4)
#define AGENT_ACT_USE (1u << 5)
#define AGENT_NUM_ACTIONS 6

/* Preprocessed frame dimensions. */
#define AGENT_FRAME_W 60
#define AGENT_FRAME_H 45
#define AGENT_FRAME_STACK 4

/* Game state vector — 20 floats matching the training observation. */
typedef struct {
  float health;           /* [0, 1] normalized from 0–200 */
  float armor;            /* [0, 1] normalized from 0–200 */
  float ammo[4];          /* [0, 1] ammo types 0–3 */
  float weapon_onehot[9]; /* one-hot selected weapon */
  float velocity_xy[2];   /* normalized momx/momy */
  float _reserved[3];     /* pad to 20 floats */
} agent_game_state_t;

/*
 * Initialize the agent. Loads the compiled-in TFLite model and
 * allocates tensors in the static arena.
 *
 * Returns 0 on success, -1 on failure.
 */
int agent_init(void);

/*
 * Run one inference step.
 *
 * Preprocesses the framebuffer (RGBA 320×200 → grayscale 60×45 INT8),
 * appends it to the frame stack, quantizes the game state, runs the
 * model, and returns an action bitfield.
 *
 * framebuf: doomgeneric screen buffer (RGBA packed uint32_t, row-major)
 * fb_width, fb_height: framebuffer dimensions (typically 320×200)
 * gs: current game state (caller fills from engine internals)
 *
 * Returns action bitfield (AGENT_ACT_* flags OR'd together).
 */
uint8_t agent_infer(const uint32_t* framebuf, int fb_width, int fb_height,
                    const agent_game_state_t* gs);

/*
 * Tear down the agent. Safe to call multiple times.
 */
void agent_destroy(void);

/*
 * Preprocess a single RGBA frame to grayscale INT8.
 * Exposed for unit testing.
 *
 * rgba: input framebuffer (RGBA packed, row-major)
 * w, h: input dimensions
 * out: output buffer, AGENT_FRAME_H × AGENT_FRAME_W INT8
 */
void agent_preprocess_frame(const uint32_t* rgba, int w, int h,
                            int8_t out[AGENT_FRAME_H][AGENT_FRAME_W]);

/*
 * Quantize a game state vector to INT8.
 * Exposed for unit testing.
 *
 * gs: input game state (20 floats)
 * out: output buffer (20 int8_t values)
 * scale: quantization scale (from TFLite tensor params)
 * zero_point: quantization zero point
 */
void agent_quantize_state(const agent_game_state_t* gs, int8_t out[20],
                          float scale, int32_t zero_point);

#ifdef __cplusplus
}
#endif

#endif /* DOOM_AGENT_H */
