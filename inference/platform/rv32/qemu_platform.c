/*
 * qemu_platform.c — Strong platform overrides for QEMU virt (riscv32).
 *
 * Provides concrete implementations of cnidoom_platform_init,
 * cnidoom_putc, and cnidoom_draw using the QEMU-specific UART
 * and ramfb drivers.
 */

#include <stdint.h>
#include <string.h>

#include "cnidoom.h"
#include "ramfb.h"

/* Defined in uart.c */
extern void uart_init(void);
extern void uart_putc(char c);

static int use_ramfb = 0;

void cnidoom_platform_init(void) {
  uart_init();
  if (ramfb_init() == 0) {
    use_ramfb = 1;
  }
}

void cnidoom_putc(char c) { uart_putc(c); }

void cnidoom_draw(const uint32_t* fb, int w, int h) {
  if (!use_ramfb) return;
  uint32_t* ramfb_buf = ramfb_get_buffer();
  memcpy(ramfb_buf, fb, (size_t)w * (size_t)h * sizeof(uint32_t));
}
