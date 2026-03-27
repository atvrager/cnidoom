/*
 * ramfb.h — ramfb display driver interface.
 */

#ifndef RAMFB_H
#define RAMFB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the ramfb device.
 * Configures a 640×400 XRGB8888 framebuffer via QEMU fw_cfg.
 * Returns 0 on success, -1 if ramfb is not available.
 */
int ramfb_init(void);

/*
 * Get a pointer to the framebuffer memory.
 * Pixels are XRGB8888 (same format as DG_ScreenBuffer).
 * Writing to this buffer is immediately visible in QEMU's display.
 */
uint32_t* ramfb_get_buffer(void);

int ramfb_get_width(void);
int ramfb_get_height(void);

#ifdef __cplusplus
}
#endif

#endif /* RAMFB_H */
