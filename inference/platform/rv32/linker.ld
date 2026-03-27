/*
 * linker.ld — Memory layout for QEMU virt machine (riscv32).
 *
 * QEMU virt RAM starts at 0x80000000.  We use 128 MB (-m 128M).
 * Layout:
 *   .text / .rodata / .data at RAM base
 *   .bss after data
 *   heap grows upward from end of BSS
 *   stack at top of RAM (grows downward)
 */

ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
    . = ORIGIN(RAM);

    .text : {
        KEEP(*(.text.init)) /* Reset vector — must be first, never GC'd */
        *(.text .text.*)
    } > RAM

    .rodata : ALIGN(4) {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > RAM

    .data : ALIGN(4) {
        __global_pointer$ = . + 0x800;
        *(.data .data.*)
        *(.sdata .sdata.*)
    } > RAM

    .bss (NOLOAD) : ALIGN(4) {
        __bss_start = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > RAM

    . = ALIGN(8);
    __heap_start = .;

    /* 256 KB stack at top of RAM. */
    __stack_size = 0x40000;
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
    __stack_bottom = __stack_top - __stack_size;
    __heap_end = __stack_bottom;

    /DISCARD/ : {
        *(.eh_frame .eh_frame_hdr)
        *(.comment)
        *(.note .note.*)
    }
}
