/*
 * doomgeneric_agent.c — doomgeneric platform layer driven by the RL agent.
 *
 * Implements the doomgeneric platform API (DG_Init, DG_DrawFrame, DG_GetKey,
 * DG_GetTickCount, DG_SetWindowTitle). Instead of reading keyboard input,
 * DG_DrawFrame runs the agent every INFER_INTERVAL tics and DG_GetKey
 * returns the agent's actions as synthetic key events.
 */

#include <string.h>
#include <time.h>

#include "doom_agent.h"
#include "doomgeneric.h"
#include "doomkeys.h"

/* Access engine internals for the game state vector. */
#include "d_player.h"
#include "doomstat.h" /* players[], consoleplayer */

/* Action repeat: infer every N tics (~8.6 decisions/sec at 35 tics/sec). */
#define INFER_INTERVAL 4

/* ------------------------------------------------------------------ */
/* Key event queue                                                    */
/* ------------------------------------------------------------------ */

/* We need up to 12 events per inference (6 key-down + 6 key-up). */
#define KEY_QUEUE_SIZE 16

typedef struct {
  unsigned char key;
  int pressed; /* 1 = down, 0 = up */
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

/* Previous action bitfield — used to generate key-up events. */
static uint8_t prev_actions = 0;
static int frames_since_infer = 0;

/* Map from action bit index → doomgeneric key code. */
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

  /* Velocity from mobj momentum, normalized. */
  if (p->mo != NULL) {
    gs->velocity_xy[0] = (float)p->mo->momx / (float)FRACUNIT / 30.0f;
    gs->velocity_xy[1] = (float)p->mo->momy / (float)FRACUNIT / 30.0f;
  }
}

/* ------------------------------------------------------------------ */
/* doomgeneric platform API                                           */
/* ------------------------------------------------------------------ */

void DG_Init(void) { agent_init(); }

void DG_DrawFrame(void) {
  frames_since_infer++;
  if (frames_since_infer < INFER_INTERVAL) return;
  frames_since_infer = 0;

  /* Build game state from engine internals. */
  agent_game_state_t gs;
  fill_game_state(&gs);

  /* Run inference. */
  const uint8_t actions =
      agent_infer(DG_ScreenBuffer, DOOMGENERIC_RESX, DOOMGENERIC_RESY, &gs);

  /* Generate key events by diffing against previous actions. */
  for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
    const uint8_t mask = 1u << i;
    const int was_pressed = (prev_actions & mask) != 0;
    const int is_pressed = (actions & mask) != 0;

    if (is_pressed && !was_pressed) {
      key_queue_push(action_to_key[i], 1); /* key down */
    } else if (!is_pressed && was_pressed) {
      key_queue_push(action_to_key[i], 0); /* key up */
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

uint32_t DG_GetTickCount(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

uint32_t DG_GetTicksMs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

void DG_SleepMs(uint32_t ms) {
  struct timespec ts;
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (ms % 1000) * 1000000L;
  nanosleep(&ts, NULL);
}

void DG_SetWindowTitle(const char* title) {
  (void)title; /* No window in headless agent mode. */
}

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
  doomgeneric_Create(argc, argv);

  for (;;) {
    doomgeneric_Tick();
  }

  return 0;
}
