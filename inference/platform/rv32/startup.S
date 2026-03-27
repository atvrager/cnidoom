/*
 * startup.S — Bare-metal RISC-V 32-bit reset handler for QEMU virt.
 *
 * Executed at the reset vector (0x80000000 via -kernel loading).
 * Sets up mtvec trap vector, enables the vector unit, zeroes BSS,
 * sets up the stack, and jumps to main().
 */

    .section .text.init, "ax", @progbits
    .global _start
    .type _start, @function

_start:
    /* Disable interrupts, enable FPU + vector unit in mstatus.
     * FS (bits [14:13]) = 01 (Initial) — enable floating-point.
     * VS (bits [10:9])  = 01 (Initial) — enable vector. */
    csrw    mie, zero
    li      t0, (1 << 13) | (1 << 9)
    csrw    mstatus, t0

    /* Set trap vector so illegal instructions etc. produce diagnostics. */
    la      t0, _trap_handler
    csrw    mtvec, t0

    /* Set global pointer (linker-relaxation anchor). */
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    /* Set stack pointer to top of stack (defined in linker script). */
    la      sp, __stack_top

    /* Zero the BSS section. */
    la      t0, __bss_start
    la      t1, __bss_end
.Lbss_loop:
    bge     t0, t1, .Lbss_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       .Lbss_loop
.Lbss_done:

    /* Call main(argc=0, argv=NULL).  If main returns, spin. */
    li      a0, 0
    li      a1, 0
    call    main

.Lhalt:
    wfi
    j       .Lhalt

    .size _start, . - _start

/* ------------------------------------------------------------------ */
/* Trap handler — prints mcause, mepc, mtval to UART and halts.       */
/* ------------------------------------------------------------------ */

    .balign 4
    .global _trap_handler
    .type _trap_handler, @function
_trap_handler:
    /* Save minimal context on the current stack. */
    addi    sp, sp, -16
    sw      ra, 12(sp)
    sw      a0, 8(sp)
    sw      a1, 4(sp)

    /* Print "TRAP mcause=" */
    la      a0, .Ltrap_msg
    call    uart_puts

    /* Print mcause as hex. */
    csrr    a0, mcause
    call    _print_hex

    /* Print " mepc=" */
    la      a0, .Lmepc_msg
    call    uart_puts
    csrr    a0, mepc
    call    _print_hex

    /* Print " mtval=" */
    la      a0, .Lmtval_msg
    call    uart_puts
    csrr    a0, mtval
    call    _print_hex

    /* Newline. */
    li      a0, '\n'
    call    uart_putc

    /* Halt. */
.Ltrap_halt:
    wfi
    j       .Ltrap_halt

    .size _trap_handler, . - _trap_handler

/* ------------------------------------------------------------------ */
/* _print_hex — print a0 as 8-digit hex via uart_putc.                */
/* ------------------------------------------------------------------ */

    .type _print_hex, @function
_print_hex:
    addi    sp, sp, -16
    sw      ra, 12(sp)
    sw      s0, 8(sp)
    sw      s1, 4(sp)
    mv      s0, a0          /* value to print */
    li      s1, 28          /* shift = 28 (start from MSB) */
.Lhex_loop:
    bltz    s1, .Lhex_done
    srl     a0, s0, s1
    andi    a0, a0, 0xf
    li      t0, 10
    blt     a0, t0, .Lhex_digit
    addi    a0, a0, ('a' - 10)
    j       .Lhex_print
.Lhex_digit:
    addi    a0, a0, '0'
.Lhex_print:
    call    uart_putc
    addi    s1, s1, -4
    j       .Lhex_loop
.Lhex_done:
    lw      s1, 4(sp)
    lw      s0, 8(sp)
    lw      ra, 12(sp)
    addi    sp, sp, 16
    ret

    .size _print_hex, . - _print_hex

/* ------------------------------------------------------------------ */
/* String constants                                                   */
/* ------------------------------------------------------------------ */

    .section .rodata.trap, "a", @progbits
.Ltrap_msg:
    .asciz  "\r\nTRAP mcause="
.Lmepc_msg:
    .asciz  " mepc="
.Lmtval_msg:
    .asciz  " mtval="
