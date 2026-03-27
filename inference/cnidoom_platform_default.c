/*
 * cnidoom_platform_default.c — Weak fallback platform implementations.
 *
 * All functions here are __attribute__((weak)).  Link strong overrides
 * to replace any or all of them for your target hardware.
 */

#include <stdint.h>

#include "cnidoom.h"

/* Config set by cnidoom_run(). */
extern cnidoom_config_t g_cnidoom_config;

/* ------------------------------------------------------------------ */
/* Semihosting (used as timing fallback)                              */
/* ------------------------------------------------------------------ */

#define SYS_ELAPSED 0x30
#define SYS_TICKFREQ 0x31
#define SYS_WRITEC 0x03

static inline long semihosting_call(long op, long arg) {
  register long a0 __asm__("a0") = op;
  register long a1 __asm__("a1") = arg;
  __asm__ volatile(
      ".option push\n"
      ".option norvc\n"
      "slli x0, x0, 0x1f\n"
      "ebreak\n"
      "srai x0, x0, 7\n"
      ".option pop\n"
      : "+r"(a0)
      : "r"(a1)
      : "memory");
  return a0;
}

/* ------------------------------------------------------------------ */
/* CLINT mtime reader (when base address is configured)               */
/* ------------------------------------------------------------------ */

static uint64_t read_clint_mtime(uintptr_t base) {
  volatile uint32_t* lo = (volatile uint32_t*)base;
  volatile uint32_t* hi = (volatile uint32_t*)(base + 4);
  uint32_t hi1, lo_val, hi2;
  do {
    hi1 = *hi;
    lo_val = *lo;
    hi2 = *hi;
  } while (hi1 != hi2);
  return ((uint64_t)hi1 << 32) | lo_val;
}

/* ------------------------------------------------------------------ */
/* Weak fallback: cnidoom_draw — no-op                                */
/* ------------------------------------------------------------------ */

__attribute__((weak)) void cnidoom_draw(const uint32_t* fb, int w, int h) {
  (void)fb;
  (void)w;
  (void)h;
}

/* ------------------------------------------------------------------ */
/* Weak fallback: cnidoom_get_ticks_ms — CLINT or semihosting         */
/* ------------------------------------------------------------------ */

__attribute__((weak)) uint32_t cnidoom_get_ticks_ms(void) {
  if (g_cnidoom_config.clint_mtime_base != 0) {
    uint64_t ticks = read_clint_mtime(g_cnidoom_config.clint_mtime_base);
    uint32_t freq = g_cnidoom_config.clint_mtime_freq;
    if (freq == 0) freq = 10000000;
    return (uint32_t)(ticks / (freq / 1000));
  }
  /* Semihosting fallback: SYS_ELAPSED returns a 64-bit tick count. */
  uint64_t ticks = 0;
  semihosting_call(SYS_ELAPSED, (long)&ticks);
  /* SYS_TICKFREQ returns the tick frequency. */
  long freq = semihosting_call(SYS_TICKFREQ, 0);
  if (freq <= 0) freq = 1000; /* assume ms if unknown */
  return (uint32_t)(ticks / ((uint64_t)freq / 1000));
}

/* ------------------------------------------------------------------ */
/* Weak fallback: cnidoom_sleep_ms — busy-wait                        */
/* ------------------------------------------------------------------ */

__attribute__((weak)) void cnidoom_sleep_ms(uint32_t ms) {
  uint32_t target = cnidoom_get_ticks_ms() + ms;
  while (cnidoom_get_ticks_ms() < target) {
    __asm__ volatile("nop");
  }
}

/* ------------------------------------------------------------------ */
/* Weak fallback: cnidoom_putc — semihosting SYS_WRITEC               */
/* ------------------------------------------------------------------ */

__attribute__((weak)) void cnidoom_putc(char c) {
  semihosting_call(SYS_WRITEC, (long)&c);
}

/* ------------------------------------------------------------------ */
/* Weak fallback: cnidoom_platform_init — no-op                       */
/* ------------------------------------------------------------------ */

__attribute__((weak)) void cnidoom_platform_init(void) {}
