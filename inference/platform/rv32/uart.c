/*
 * uart.c — NS16550a UART driver for QEMU virt (riscv32).
 *
 * QEMU virt maps UART0 at 0x10000000.  We use polled I/O (no IRQs).
 * This provides the _write() syscall stub so printf works.
 */

#include <stdint.h>

#define UART0_BASE ((volatile uint8_t*)0x10000000)

/* NS16550a register offsets. */
#define UART_THR 0         /* Transmit Holding Register (write) */
#define UART_LSR 5         /* Line Status Register (read) */
#define UART_LSR_THRE 0x20 /* THR Empty */

void uart_init(void) {
  /* QEMU virt UART is pre-configured — no init needed. */
}

void uart_putc(char c) {
  /* Wait for transmit holding register to be empty. */
  while ((UART0_BASE[UART_LSR] & UART_LSR_THRE) == 0) {
  }
  UART0_BASE[UART_THR] = (uint8_t)c;
}

void uart_puts(const char* s) {
  while (*s) {
    if (*s == '\n') uart_putc('\r');
    uart_putc(*s++);
  }
}
