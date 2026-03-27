/*
 * doomgeneric_agent_rv32.c — Bare-metal RISC-V platform layer.
 *
 * Implements the doomgeneric DG_* API for QEMU virt (riscv32):
 *   - Display via ramfb (QEMU renders in its SDL window)
 *   - Timing via CLINT mtime register
 *   - Agent inference via the static code-generated engine
 *   - WAD file I/O via semihosting (handled by syscalls.c)
 *
 * No OS, no libc beyond what syscalls.c provides.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "doom_agent.h"
#include "doomgeneric.h"
#include "doomkeys.h"
#include "platform/rv32/ramfb.h"

/* Access engine internals for the game state vector. */
#include "d_player.h"
#include "doomstat.h" /* players[], consoleplayer */

/* ------------------------------------------------------------------ */
/* CLINT timer — QEMU virt riscv32 (wall clock for game timing)       */
/* ------------------------------------------------------------------ */

/*
 * CLINT mtime register: 64-bit free-running counter at 10 MHz.
 * Address: 0x200BFF8 on QEMU virt.
 *
 * On RV32, mtime is read as two 32-bit halves.  We loop to handle
 * the case where the low word wraps between reads.
 */
#define CLINT_MTIME_LO ((volatile uint32_t*)0x200BFF8)
#define CLINT_MTIME_HI ((volatile uint32_t*)0x200BFFC)
#define MTIME_FREQ 10000000ULL /* 10 MHz */

static uint64_t read_mtime(void) {
  uint32_t hi1, lo, hi2;
  do {
    hi1 = *CLINT_MTIME_HI;
    lo = *CLINT_MTIME_LO;
    hi2 = *CLINT_MTIME_HI;
  } while (hi1 != hi2);
  return ((uint64_t)hi1 << 32) | lo;
}

static uint32_t mtime_to_ms(uint64_t ticks) {
  return (uint32_t)(ticks / (MTIME_FREQ / 1000));
}

/* Note: mcycle/minstret CSRs are NOT useful under QEMU — it inflates
 * minstret by vl for each vector instruction, making RVV appear 10×
 * worse than scalar.  Use mtime (wall clock) for QEMU, and real HW
 * counters only on physical silicon. */

/* ------------------------------------------------------------------ */
/* Action repeat & key event queue (same pattern as doomgeneric_agent) */
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
/* ramfb display state                                                */
/* ------------------------------------------------------------------ */

static int use_ramfb = 0;

/* FPS / inference timing. */
static uint64_t fps_last_time = 0;
static int fps_frame_count = 0;
static int total_frames = 0;
static uint64_t last_inference_ticks = 0;

/* ------------------------------------------------------------------ */
/* doomgeneric platform API                                           */
/* ------------------------------------------------------------------ */

void DG_Init(void) {
  /* Initialize UART (for printf). */
  extern void uart_init(void);
  uart_init();

  /* Try to initialize ramfb display. */
  if (ramfb_init() == 0) {
    use_ramfb = 1;
  }

  /* Initialize inference engine. */
  agent_init();
}

void DG_DrawFrame(void) {
  /* Copy Doom's framebuffer to ramfb for display. */
  if (use_ramfb) {
    uint32_t* fb = ramfb_get_buffer();
    memcpy(fb, DG_ScreenBuffer,
           DOOMGENERIC_RESX * DOOMGENERIC_RESY * sizeof(uint32_t));
  }

  /* FPS counter: print every 1 second. */
  total_frames++;
  fps_frame_count++;
  uint64_t now = read_mtime();
  if (fps_last_time == 0) fps_last_time = now;
  uint64_t elapsed = now - fps_last_time;
  if (elapsed >= MTIME_FREQ) { /* >= 1 second */
    unsigned long inference_us =
        (unsigned long)(last_inference_ticks / (MTIME_FREQ / 1000000));
    printf("[FPS: %d | Inference: %luus | Frame: %d]\n", fps_frame_count,
           inference_us, total_frames);
    fps_frame_count = 0;
    fps_last_time = now;
  }

  /* Auto-start: mash Enter during title/demo/intermission screens
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

  /* Run agent inference at the configured interval. */
  frames_since_infer++;
  if (frames_since_infer < INFER_INTERVAL) return;
  frames_since_infer = 0;

  agent_game_state_t gs;
  fill_game_state(&gs);

  /* Time the inference. */
  uint64_t t0 = read_mtime();
  const uint8_t actions =
      agent_infer(DG_ScreenBuffer, DOOMGENERIC_RESX, DOOMGENERIC_RESY, &gs);
  last_inference_ticks = read_mtime() - t0;

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

uint32_t DG_GetTicksMs(void) { return mtime_to_ms(read_mtime()); }

void DG_SleepMs(uint32_t ms) {
  uint64_t target = read_mtime() + (uint64_t)ms * (MTIME_FREQ / 1000);
  while (read_mtime() < target) {
    __asm__ volatile("nop");
  }
}

void DG_SetWindowTitle(const char* title) {
  (void)title; /* No way to set QEMU window title from guest. */
}

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

/*
 * Doom expects argc/argv for -iwad path.  On bare-metal, we hardcode
 * the WAD path.  Semihosting opens files on the host filesystem, so
 * the path is relative to QEMU's working directory.
 */
static char* rv32_argv[] = {"doom", "-iwad", "wads/DOOM1.WAD"};

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  doomgeneric_Create(3, rv32_argv);

  for (;;) {
    doomgeneric_Tick();
  }

  return 0;
}
