/*
 * cnidoom.h — Public API for the cnidoom embedded library (libcnidoom.a).
 *
 * Links the Doom engine + RL agent inference into a single static library.
 * Downstream users provide main(), call cnidoom_run(), and optionally
 * override weak platform callbacks with strong symbols for their hardware.
 */

#ifndef CNIDOOM_H
#define CNIDOOM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Configuration                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
  const char* wad_path;       /* NULL → "DOOM1.WAD" via semihosting */
  uintptr_t clint_mtime_base; /* 0 → semihosting timing fallback */
  uint32_t clint_mtime_freq;  /* Hz, default 10 MHz */
} cnidoom_config_t;

/* Return default config for QEMU virt (riscv32). */
cnidoom_config_t cnidoom_default_config(void);

/* Run the Doom engine with agent inference.  Does not return. */
void cnidoom_run(const cnidoom_config_t* cfg);

/* ------------------------------------------------------------------ */
/* Platform callbacks — weak defaults, override with strong symbols   */
/* ------------------------------------------------------------------ */

/* Display: called each frame with the XRGB8888 framebuffer. */
void cnidoom_draw(const uint32_t* fb, int w, int h);

/* Timing: return monotonic millisecond tick count. */
uint32_t cnidoom_get_ticks_ms(void);

/* Sleep: busy-wait for the given number of milliseconds. */
void cnidoom_sleep_ms(uint32_t ms);

/* Console output: write a single character (for printf via _write). */
void cnidoom_putc(char c);

/* Platform init: called once before the engine starts. */
void cnidoom_platform_init(void);

#ifdef __cplusplus
}
#endif

#endif /* CNIDOOM_H */
