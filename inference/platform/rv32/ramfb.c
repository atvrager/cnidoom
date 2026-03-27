/*
 * ramfb.c — ramfb display driver for QEMU.
 *
 * ramfb is QEMU's simplest framebuffer device:
 *   1. Guest writes a RAMFBCfg struct to fw_cfg selector 0x0100
 *   2. QEMU reads pixel data from the address specified in the struct
 *   3. QEMU displays it in its SDL window
 *
 * We use 32-bit XRGB8888 at 640×400 (matching Doom's DOOMGENERIC_RESX/Y).
 *
 * fw_cfg device on QEMU virt: MMIO at 0x10100000.
 *   0x10100000: data register (1 byte read/write)
 *   0x10100008: selector register (2 bytes, big-endian write)
 *   0x10100010: DMA address register (8 bytes, big-endian write)
 */

#include "ramfb.h"

#include <stdint.h>
#include <string.h>

/* fw_cfg MMIO registers (QEMU virt riscv32). */
#define FW_CFG_BASE 0x10100000
#define FW_CFG_DATA ((volatile uint8_t*)FW_CFG_BASE)
#define FW_CFG_SEL ((volatile uint16_t*)(FW_CFG_BASE + 8))
#define FW_CFG_DMA ((volatile uint64_t*)(FW_CFG_BASE + 16))

/* fw_cfg selectors. */
#define FW_CFG_FILE_DIR 0x0019

/* DMA control bits. */
#define FW_CFG_DMA_CTL_SELECT (1u << 3)
#define FW_CFG_DMA_CTL_WRITE (1u << 4)
#define FW_CFG_DMA_CTL_READ (1u << 1)
#define FW_CFG_DMA_CTL_ERROR (1u << 0)

/* DRM fourcc for XRGB8888. */
#define DRM_FORMAT_XRGB8888 0x34325258

/* RAMFBCfg structure — must match QEMU's expectation (all big-endian). */
typedef struct __attribute__((packed)) {
  uint64_t addr;   /* Physical address of framebuffer */
  uint32_t fourcc; /* Pixel format (DRM fourcc) */
  uint32_t flags;  /* 0 */
  uint32_t width;  /* Pixels */
  uint32_t height; /* Pixels */
  uint32_t stride; /* Bytes per row */
} ramfb_cfg_t;

/* fw_cfg DMA access structure (all big-endian). */
typedef struct __attribute__((packed)) {
  uint32_t control;
  uint32_t length;
  uint64_t address;
} fw_cfg_dma_access_t;

/* Framebuffer memory — statically allocated. */
#define RAMFB_WIDTH 640
#define RAMFB_HEIGHT 400
static uint32_t ramfb_buffer[RAMFB_WIDTH * RAMFB_HEIGHT]
    __attribute__((aligned(4096)));

/* Byte-swap helpers (RISC-V is little-endian, fw_cfg wants big-endian). */
static inline uint16_t bswap16(uint16_t x) {
  return (uint16_t)((x >> 8) | (x << 8));
}

static inline uint32_t bswap32(uint32_t x) {
  return ((x >> 24) & 0x000000FFu) | ((x >> 8) & 0x0000FF00u) |
         ((x << 8) & 0x00FF0000u) | ((x << 24) & 0xFF000000u);
}

static inline uint64_t bswap64(uint64_t x) {
  return ((uint64_t)bswap32((uint32_t)x) << 32) |
         (uint64_t)bswap32((uint32_t)(x >> 32));
}

/*
 * Find a fw_cfg file by name. Returns the selector, or -1 on failure.
 */
static int fw_cfg_find_file(const char* name) {
  /* Select the file directory. */
  *FW_CFG_SEL = bswap16(FW_CFG_FILE_DIR);

  /* Read the file count (big-endian uint32). */
  uint32_t count_be = 0;
  for (int i = 0; i < 4; i++) {
    ((uint8_t*)&count_be)[i] = *FW_CFG_DATA;
  }
  uint32_t count = bswap32(count_be);

  /* Iterate through file entries. */
  for (uint32_t i = 0; i < count; i++) {
    /* Each entry: uint32 size, uint16 select, uint16 reserved, char name[56] */
    uint32_t fsize_be = 0;
    for (int b = 0; b < 4; b++) ((uint8_t*)&fsize_be)[b] = *FW_CFG_DATA;

    uint16_t fsel_be = 0;
    for (int b = 0; b < 2; b++) ((uint8_t*)&fsel_be)[b] = *FW_CFG_DATA;

    /* Skip reserved. */
    (void)*FW_CFG_DATA;
    (void)*FW_CFG_DATA;

    /* Read name. */
    char fname[56];
    for (int b = 0; b < 56; b++) {
      fname[b] = (char)*FW_CFG_DATA;
    }

    if (strcmp(fname, name) == 0) {
      return (int)bswap16(fsel_be);
    }
  }
  return -1;
}

/*
 * Write data to a fw_cfg file using DMA.
 */
static void fw_cfg_dma_write(uint16_t selector, const void* data,
                             uint32_t len) {
  fw_cfg_dma_access_t dma __attribute__((aligned(4)));
  dma.control = bswap32(FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_WRITE |
                        ((uint32_t)selector << 16));
  dma.length = bswap32(len);
  dma.address = bswap64((uint64_t)(uintptr_t)data);

  /* Write the DMA access struct address to the DMA register. */
  uint64_t dma_addr = bswap64((uint64_t)(uintptr_t)&dma);
  *FW_CFG_DMA = dma_addr;

  /* Spin until DMA completes (control field zeroed by QEMU). */
  while (dma.control != 0) {
    __asm__ volatile("" ::: "memory");
  }
}

int ramfb_init(void) {
  /* Find the etc/ramfb fw_cfg file. */
  int selector = fw_cfg_find_file("etc/ramfb");
  if (selector < 0) {
    return -1; /* ramfb not available */
  }

  /* Build the RAMFBCfg in big-endian. */
  ramfb_cfg_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.addr = bswap64((uint64_t)(uintptr_t)ramfb_buffer);
  cfg.fourcc = bswap32(DRM_FORMAT_XRGB8888);
  cfg.flags = 0;
  cfg.width = bswap32(RAMFB_WIDTH);
  cfg.height = bswap32(RAMFB_HEIGHT);
  cfg.stride = bswap32(RAMFB_WIDTH * 4);

  /* Write config to the ramfb selector. */
  fw_cfg_dma_write((uint16_t)selector, &cfg, sizeof(cfg));

  /* Clear framebuffer to black. */
  memset(ramfb_buffer, 0, sizeof(ramfb_buffer));

  return 0;
}

uint32_t* ramfb_get_buffer(void) { return ramfb_buffer; }

int ramfb_get_width(void) { return RAMFB_WIDTH; }

int ramfb_get_height(void) { return RAMFB_HEIGHT; }
