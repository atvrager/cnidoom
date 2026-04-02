/*
 * cnidoom.c — Entry point + DG_* bridge for libcnidoom.a.
 *
 * Extracted from doomgeneric_agent_rv32.c.  Bridges the cnidoom_*
 * platform callbacks to doomgeneric's DG_* API so the Doom engine
 * runs portably on any platform that links this library.
 */

#include "cnidoom.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "doom_agent.h"
#include "doomgeneric.h"
#include "doomkeys.h"

/* Access engine internals for the game state vector. */

#include "d_player.h"
#include "doomstat.h" /* players[], consoleplayer */

#include <stdbool.h>
/* ------------------------------------------------------------------ */
/* Global config (set by cnidoom_run, read by platform defaults)      */
/* ------------------------------------------------------------------ */

cnidoom_config_t g_cnidoom_config;

/* ------------------------------------------------------------------ */
/* Action repeat & key event queue                                    */
/* ------------------------------------------------------------------ */

#define INFER_INTERVAL 4
#define KEY_QUEUE_SIZE 16

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

static uint8_t prev_actions = 0;
static int frames_since_infer = 0;

static const unsigned char action_to_key[AGENT_NUM_ACTIONS] = {
    KEY_UPARROW,    /* AGENT_ACT_FORWARD  */
    KEY_DOWNARROW,  /* AGENT_ACT_BACKWARD */
    KEY_LEFTARROW,  /* AGENT_ACT_TURN_L   */
    KEY_RIGHTARROW, /* AGENT_ACT_TURN_R   */
    KEY_FIRE,       /* AGENT_ACT_FIRE     */
    KEY_USE,        /* AGENT_ACT_USE      */
};

/* ------------------------------------------------------------------ */
/* Game state extraction                                              */
/* ------------------------------------------------------------------ */

static void fill_game_state(agent_game_state_t* gs) {
  memset(gs, 0, sizeof(*gs));
  player_t* p = &players[consoleplayer];

  gs->health = (float)p->health / 200.0f;
  gs->armor = (float)p->armorpoints / 200.0f;

  for (int i = 0; i < 4; i++) {
    gs->ammo[i] = (float)p->ammo[i] / 200.0f;
  }

  const int w = p->readyweapon;
  if (w >= 0 && w < 9) {
    gs->weapon_onehot[w] = 1.0f;
  }

  if (p->mo != NULL) {
    gs->velocity_xy[0] = (float)p->mo->momx / (float)FRACUNIT / 30.0f;
    gs->velocity_xy[1] = (float)p->mo->momy / (float)FRACUNIT / 30.0f;
  }
}

/* ------------------------------------------------------------------ */
/* FPS / inference timing                                             */
/* ------------------------------------------------------------------ */

static uint32_t fps_last_time = 0;
static int fps_frame_count = 0;
static int total_frames = 0;
static uint32_t last_inference_ms = 0;

/* ------------------------------------------------------------------ */
/* doomgeneric platform API (bridges to cnidoom_* callbacks)          */
/* ------------------------------------------------------------------ */

void DG_Init(void) {
  cnidoom_platform_init();
  agent_init();
}

#define STARTUP_GRACE_PERIOD 100
void DG_DrawFrame(void) {
  /* Display via platform callback. */
  cnidoom_draw(DG_ScreenBuffer, DOOMGENERIC_RESX, DOOMGENERIC_RESY);

  /* FPS counter: print every ~1 second. */
  total_frames++;
  fps_frame_count++;
  uint32_t now = cnidoom_get_ticks_ms();
  if (fps_last_time == 0) fps_last_time = now;
  uint32_t elapsed = now - fps_last_time;
  if (elapsed >= 1000) {
    printf("[FPS: %d | Inference: %ums | Frame: %d]\n", fps_frame_count,
           (unsigned)last_inference_ms, total_frames);
    fps_frame_count = 0;
    fps_last_time = now;
  }
  /*
   * SAFE START GATE:
   * Only allow inference if we are past the grace period
   * AND we are in a real level (not a demo playback).
   */
  bool is_real_gameplay = (gamestate == GS_LEVEL && !demoplayback);

  if (total_frames < STARTUP_GRACE_PERIOD || !is_real_gameplay) {
    /* Mash Enter to clear title screens and skip demos.
     * We hold the key for 5 frames to ensure it's registered. */
    static int autostart_timer = 0;
    autostart_timer++;
    if (autostart_timer == 15) {
      key_queue_push(KEY_ENTER, 1);  // Press
    } else if (autostart_timer >= 20) {
      key_queue_push(KEY_ENTER, 0);  // Release
      autostart_timer = 0;
    }
    return;  // SHUT DOWN INFERENCE FOR THIS FRAME
  }

  /* Run agent inference at the configured interval. */
  frames_since_infer++;
  if (frames_since_infer < INFER_INTERVAL) return;
  frames_since_infer = 0;

  agent_game_state_t gs;
  fill_game_state(&gs);

  uint32_t t0 = cnidoom_get_ticks_ms();
  const uint8_t actions =
      agent_infer(DG_ScreenBuffer, DOOMGENERIC_RESX, DOOMGENERIC_RESY, &gs);
  last_inference_ms = cnidoom_get_ticks_ms() - t0;

  /* Generate key events by diffing against previous actions. */
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

uint32_t DG_GetTicksMs(void) { return cnidoom_get_ticks_ms(); }

void DG_SleepMs(uint32_t ms) { cnidoom_sleep_ms(ms); }

void DG_SetWindowTitle(const char* title) { (void)title; }

/* ------------------------------------------------------------------ */
/* Default config                                                     */
/* ------------------------------------------------------------------ */

cnidoom_config_t cnidoom_default_config(void) {
  cnidoom_config_t cfg;
  cfg.wad_path = NULL;
  cfg.clint_mtime_base = 0x200BFF8;
  cfg.clint_mtime_freq = 10000000; /* 10 MHz */
  return cfg;
}

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

void cnidoom_run(const cnidoom_config_t* cfg) {
  if (cfg != NULL) {
    g_cnidoom_config = *cfg;
  } else {
    g_cnidoom_config = cnidoom_default_config();
  }

  const char* wad =
      g_cnidoom_config.wad_path ? g_cnidoom_config.wad_path : "wads/DOOM1.WAD";

  char* argv[] = {"doom", "-iwad", (char*)wad};
  doomgeneric_Create(3, argv);

  for (;;) {
    doomgeneric_Tick();
  }
}
