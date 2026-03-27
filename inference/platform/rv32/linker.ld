/*
 * linker.ld — Memory layout emulating a RISC-V embedded SoC.
 *
 * Target memory map:
 *   ITCM  0x00000000  1 MB  RX   — instruction TCM (zero wait state)
 *   DTCM  0x10000000  1 MB  RW   — data TCM (zero wait state)
 *   SRAM  0x20000000  4 MB  RWX  — external SRAM (moderate latency)
 *   DDR   0x80000000  2 GB  RWX  — external DDR (highest latency)
 *
 * For QEMU virt (RAM fixed at 0x80000000), we pack the regions
 * contiguously starting at the QEMU RAM base.  The sizes and
 * permissions match the real target — only base addresses differ.
 *
 * Section placement:
 *   ITCM: .text                 (inference kernels + Doom engine code)
 *   DTCM: .data, .sdata, stack  (hot working memory)
 *   SRAM: .rodata, .bss         (model weights, engine state)
 *   DDR:  .wad, heap             (embedded WAD, Z_Malloc zone)
 */

ENTRY(_start)

MEMORY
{
    ITCM (rx)  : ORIGIN = 0x80000000, LENGTH = 1M
    DTCM (rw)  : ORIGIN = 0x80100000, LENGTH = 1M
    SRAM (rwx) : ORIGIN = 0x80200000, LENGTH = 4M
    DDR  (rwx) : ORIGIN = 0x80600000, LENGTH = 128M - 6M
}

SECTIONS
{
    /* ── ITCM: code ─────────────────────────────────────────────── */

    .text : {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > ITCM

    /* ── SRAM: read-only data (model weights, LUTs, strings) ───── */

    .rodata : ALIGN(4) {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > SRAM

    /* ── SRAM: cnidoom model weights (const, from codegen) ────────── */

    .cnidoom.weights : ALIGN(4) {
        *(.cnidoom.weights)
    } > SRAM

    /* ── DTCM: read-write initialized data + stack ─────────────── */

    .data : ALIGN(4) {
        __global_pointer$ = . + 0x800;
        *(.data .data.*)
        *(.sdata .sdata.*)
    } > DTCM

    /* Stack at top of DTCM, grows downward.  64 KB. */
    __stack_size = 0x10000;
    __stack_top = ORIGIN(DTCM) + LENGTH(DTCM);
    __stack_bottom = __stack_top - __stack_size;

    /* ── SRAM: uninitialized data ──────────────────────────────── */

    .bss (NOLOAD) : ALIGN(4) {
        __bss_start = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > SRAM

    /* ── SRAM: cnidoom scratch buffer (RW, hot activations) ───────── */

    .cnidoom.scratch (NOLOAD) : ALIGN(4) {
        *(.cnidoom.scratch)
    } > SRAM

    /* ── DDR: embedded WAD + heap ──────────────────────────────── */

    .wad : ALIGN(4) {
        KEEP(*(.wad .wad.*))
        KEEP(*(.cnidoom.wad))
    } > DDR

    /* Heap lives in DDR after the WAD — plenty of room for Doom's
     * Z_Malloc zone (~6 MB) and other runtime allocations. */
    . = ALIGN(8);
    __heap_start = .;
    __heap_end = ORIGIN(DDR) + LENGTH(DDR);

    /DISCARD/ : {
        *(.eh_frame .eh_frame_hdr)
        *(.comment)
        *(.note .note.*)
    }
}
