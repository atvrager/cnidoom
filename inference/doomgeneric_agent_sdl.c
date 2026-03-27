/*
 * doomgeneric_agent_sdl.c — SDL2 platform layer with RL agent.
 *
 * Implements the doomgeneric platform API. Renders to an SDL2 window
 * while the RL agent drives gameplay. Also accepts real keyboard input
 * for ESC (quit) and P (pause/unpause agent).
 *
 * Model path: --model <path> CLI arg, or DOOM_MODEL env var.
 *
 * Build with -DDOOM_AGENT_HOST to select this platform + host backend.
 */

#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "doom_agent.h"
#include "doomgeneric.h"
#include "doomkeys.h"
#include "golden_log.h"

/* Access engine internals for the game state vector. */
#include "d_player.h"
#include "doomdef.h"
#include "doomstat.h"

/* ------------------------------------------------------------------ */
/* Configuration                                                      */
/* ------------------------------------------------------------------ */

#define INFER_INTERVAL 4 /* action repeat: infer every N tics */
#define WINDOW_SCALE 3   /* scale up the 320×200 framebuffer */

/* ------------------------------------------------------------------ */
/* SDL2 state                                                         */
/* ------------------------------------------------------------------ */

static SDL_Window* window = NULL;
static SDL_Renderer* renderer = NULL;
static SDL_Texture* texture = NULL;

/* ------------------------------------------------------------------ */
/* Agent state                                                        */
/* ------------------------------------------------------------------ */

static float frame_stack_f[AGENT_FRAME_STACK][AGENT_FRAME_H][AGENT_FRAME_W];
static int frame_count = 0;
static int agent_paused = 0;
static uint8_t prev_actions = 0;
static int frames_since_infer = 0;
static const char* model_path = NULL;

/* Performance stats. */
static int64_t last_inference_us = 0;
static int total_frames = 0;
static uint32_t fps_tick_start = 0;
static int fps_frame_count = 0;
static float current_fps = 0.0f;

/* ------------------------------------------------------------------ */
/* Key event queue                                                    */
/* ------------------------------------------------------------------ */

#define KEY_QUEUE_SIZE 32

typedef struct {
  unsigned char key;
  int pressed;
} key_event_t;

static key_event_t key_queue[KEY_QUEUE_SIZE];
static int key_queue_head = 0;
static int key_queue_count = 0;

static void key_queue_push(unsigned char key, int pressed) {
  if (key_queue_count >= KEY_QUEUE_SIZE) return;
  const int idx = (key_queue_head + key_queue_count) % KEY_QUEUE_SIZE;
  key_queue[idx].key = key;
  key_queue[idx].pressed = pressed;
  key_queue_count++;
}

static const unsigned char action_to_key[AGENT_NUM_ACTIONS] = {
    KEY_UPARROW,    KEY_DOWNARROW, KEY_LEFTARROW,
    KEY_RIGHTARROW, KEY_FIRE,      KEY_USE,
};

/* ------------------------------------------------------------------ */
/* Game state extraction                                              */
/* ------------------------------------------------------------------ */

static void fill_game_state(float state[20]) {
  memset(state, 0, 20 * sizeof(float));

  player_t* p = &players[consoleplayer];

  state[0] = (float)p->health / 200.0f;
  state[1] = (float)p->armorpoints / 200.0f;
  for (int i = 0; i < 4; i++) {
    state[2 + i] = (float)p->ammo[i] / 200.0f;
  }
  const int w = p->readyweapon;
  if (w >= 0 && w < 9) {
    state[6 + w] = 1.0f;
  }
  if (p->mo != NULL) {
    state[15] = (float)p->mo->momx / (float)FRACUNIT / 30.0f;
    state[16] = (float)p->mo->momy / (float)FRACUNIT / 30.0f;
  }
}

/* ------------------------------------------------------------------ */
/* SDL2 keyboard → doomgeneric key translation                        */
/* ------------------------------------------------------------------ */

static unsigned char sdl_to_doom_key(SDL_Keycode sym) {
  switch (sym) {
    case SDLK_LEFT:
      return KEY_LEFTARROW;
    case SDLK_RIGHT:
      return KEY_RIGHTARROW;
    case SDLK_UP:
      return KEY_UPARROW;
    case SDLK_DOWN:
      return KEY_DOWNARROW;
    case SDLK_RETURN:
      return KEY_ENTER;
    case SDLK_ESCAPE:
      return KEY_ESCAPE;
    case SDLK_SPACE:
      return KEY_USE;
    case SDLK_LCTRL:
    case SDLK_RCTRL:
      return KEY_FIRE;
    case SDLK_LSHIFT:
    case SDLK_RSHIFT:
      return KEY_RSHIFT;
    case SDLK_TAB:
      return KEY_TAB;
    default:
      if (sym >= SDLK_a && sym <= SDLK_z)
        return (unsigned char)(sym - SDLK_a + 'a');
      if (sym >= SDLK_0 && sym <= SDLK_9)
        return (unsigned char)(sym - SDLK_0 + '0');
      return 0;
  }
}

/* ------------------------------------------------------------------ */
/* doomgeneric platform API                                           */
/* ------------------------------------------------------------------ */

void DG_Init(void) {
  /* Find model path. */
  if (model_path == NULL) {
    model_path = getenv("DOOM_MODEL");
  }
  if (model_path == NULL) {
    model_path = "models/doom_agent_int8.tflite";
  }

  fprintf(stderr, "Loading model: %s\n", model_path);
  if (agent_init_host(model_path) != 0) {
    fprintf(stderr, "FATAL: agent_init_host failed\n");
    exit(1);
  }

  golden_log_init();

  /* Initialize SDL2. */
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    fprintf(stderr, "FATAL: SDL_Init failed: %s\n", SDL_GetError());
    exit(1);
  }

  window = SDL_CreateWindow(
      "cnidoom — agent", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
      DOOMGENERIC_RESX * WINDOW_SCALE, DOOMGENERIC_RESY * WINDOW_SCALE, 0);
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
  texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                              SDL_TEXTUREACCESS_STREAMING, DOOMGENERIC_RESX,
                              DOOMGENERIC_RESY);

  memset(frame_stack_f, 0, sizeof(frame_stack_f));
  frame_count = 0;
  fps_tick_start = SDL_GetTicks();
}

void DG_DrawFrame(void) {
  /* Update SDL texture from doomgeneric's screen buffer. */
  void* pixels;
  int pitch;
  if (SDL_LockTexture(texture, NULL, &pixels, &pitch) == 0) {
    for (int y = 0; y < DOOMGENERIC_RESY; y++) {
      memcpy((uint8_t*)pixels + y * pitch,
             (uint8_t*)DG_ScreenBuffer + y * DOOMGENERIC_RESX * 4,
             DOOMGENERIC_RESX * 4);
    }
    SDL_UnlockTexture(texture);
  }
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
  SDL_RenderPresent(renderer);

  /* Process SDL events (keyboard). */
  SDL_Event ev;
  while (SDL_PollEvent(&ev)) {
    switch (ev.type) {
      case SDL_QUIT:
        exit(0);
        break;
      case SDL_KEYDOWN:
        if (ev.key.keysym.sym == SDLK_p) {
          agent_paused = !agent_paused;
          fprintf(stderr, "Agent %s\n", agent_paused ? "PAUSED" : "RESUMED");
        } else {
          unsigned char dk = sdl_to_doom_key(ev.key.keysym.sym);
          if (dk) key_queue_push(dk, 1);
        }
        break;
      case SDL_KEYUP: {
        unsigned char dk = sdl_to_doom_key(ev.key.keysym.sym);
        if (dk) key_queue_push(dk, 0);
      } break;
    }
  }

  /* FPS counter. */
  total_frames++;
  fps_frame_count++;
  uint32_t now = SDL_GetTicks();
  uint32_t elapsed = now - fps_tick_start;
  if (elapsed >= 1000) {
    current_fps = (float)fps_frame_count * 1000.0f / (float)elapsed;
    fps_frame_count = 0;
    fps_tick_start = now;
    /* Print stats to stderr. */
    fprintf(stderr, "\r[FPS: %.1f | Inference: %ldus | Frame: %d]   ",
            current_fps, (long)last_inference_us, total_frames);
  }

  /* Auto-start: push Enter during title/demo/intermission screens
   * to navigate menus and get into gameplay. */
  if (gamestate != GS_LEVEL) {
    static int autostart_cooldown = 0;
    if (++autostart_cooldown >= 15) { /* every ~0.4s at 35 tics/sec */
      key_queue_push(KEY_ENTER, 1);
      key_queue_push(KEY_ENTER, 0);
      autostart_cooldown = 0;
    }
    return;
  }

  /* Agent inference. */
  if (agent_paused) return;

  frames_since_infer++;
  if (frames_since_infer < INFER_INTERVAL) return;
  frames_since_infer = 0;

  /* 1. Preprocess new frame into the float frame stack. */
  const int slot = frame_count % AGENT_FRAME_STACK;
  agent_preprocess_frame_float(DG_ScreenBuffer, DOOMGENERIC_RESX,
                               DOOMGENERIC_RESY, frame_stack_f[slot]);
  frame_count++;

  /* 2. Flatten frame stack to NHWC layout. */
  float nhwc[AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK];
  for (int y = 0; y < AGENT_FRAME_H; y++) {
    for (int x = 0; x < AGENT_FRAME_W; x++) {
      for (int c = 0; c < AGENT_FRAME_STACK; c++) {
        nhwc[y * AGENT_FRAME_W * AGENT_FRAME_STACK + x * AGENT_FRAME_STACK +
             c] = frame_stack_f[c][y][x];
      }
    }
  }

  /* 3. Build game state vector. */
  float state[20];
  fill_game_state(state);

  /* 4. Run inference. */
  float probs[AGENT_NUM_ACTIONS];
  int64_t inf_us;
  const uint8_t actions = agent_infer_host(nhwc, state, probs, &inf_us);
  last_inference_us = inf_us;

  /* 5. Golden log. */
  golden_log_write(total_frames, actions, probs, inf_us);

  /* 6. Generate key events by diffing against previous actions. */
  for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
    const uint8_t mask = 1u << i;
    const int was_pressed = (prev_actions & mask) != 0;
    const int is_pressed = (actions & mask) != 0;

    if (is_pressed && !was_pressed) {
      key_queue_push(action_to_key[i], 1);
    } else if (!is_pressed && was_pressed) {
      key_queue_push(action_to_key[i], 0);
    }
  }
  prev_actions = actions;
}

int DG_GetKey(int* pressed, unsigned char* key) {
  if (key_queue_count == 0) return 0;

  *pressed = key_queue[key_queue_head].pressed;
  *key = key_queue[key_queue_head].key;
  key_queue_head = (key_queue_head + 1) % KEY_QUEUE_SIZE;
  key_queue_count--;
  return 1;
}

uint32_t DG_GetTickCount(void) { return SDL_GetTicks(); }

uint32_t DG_GetTicksMs(void) { return SDL_GetTicks(); }

void DG_SleepMs(uint32_t ms) { SDL_Delay(ms); }

void DG_SetWindowTitle(const char* title) {
  if (window != NULL) {
    /* Prepend agent info to window title. */
    char buf[256];
    snprintf(buf, sizeof(buf), "cnidoom — %s", title);
    SDL_SetWindowTitle(window, buf);
  }
}

/* ------------------------------------------------------------------ */
/* main() — parse --model, then hand off to doomgeneric's D_DoomMain  */
/* ------------------------------------------------------------------ */

extern void D_DoomMain(void);
extern void doomgeneric_Create(int argc, char** argv);

int main(int argc, char** argv) {
  /* Extract --model from argv before passing to doomgeneric. */
  char* filtered_argv[64];
  int filtered_argc = 0;

  for (int i = 0; i < argc && filtered_argc < 63; i++) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else {
      filtered_argv[filtered_argc++] = argv[i];
    }
  }
  filtered_argv[filtered_argc] = NULL;

  doomgeneric_Create(filtered_argc, filtered_argv);
  D_DoomMain();

  /* Cleanup. */
  golden_log_close();
  agent_destroy_host();
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
