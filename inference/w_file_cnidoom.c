/*
 * w_file_cnidoom.c — Unified WAD backend for the cnidoom library.
 *
 * Declares _embedded_wad_start/_embedded_wad_end as weak externs.
 * At runtime:
 *   - If non-NULL → memory-mapped zero-copy (embedded WAD blob)
 *   - If NULL     → stdc fopen/fread via semihosting
 *
 * Replaces both w_file_embedded.c and w_file_stdc.c for library users.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "w_file.h"
#include "z_zone.h"

/* Weak externs — provided by objcopy when a WAD blob is linked in. */
extern const uint8_t _embedded_wad_start[] __attribute__((weak));
extern const uint8_t _embedded_wad_end[] __attribute__((weak));

/* ------------------------------------------------------------------ */
/* Embedded WAD backend (zero-copy from linked blob)                  */
/* ------------------------------------------------------------------ */

typedef struct {
  wad_file_t wad;
} embedded_wad_file_t;

/* Forward declaration. */
extern wad_file_class_t stdc_wad_file;

static wad_file_t* W_Embedded_OpenFile(char* path) {
  (void)path;
  unsigned int size = (unsigned int)(_embedded_wad_end - _embedded_wad_start);

  embedded_wad_file_t* result =
      Z_Malloc(sizeof(embedded_wad_file_t), PU_STATIC, 0);
  result->wad.file_class = &stdc_wad_file;
  result->wad.mapped = (byte*)_embedded_wad_start;
  result->wad.length = size;

  return &result->wad;
}

static void W_Embedded_CloseFile(wad_file_t* wad) { Z_Free(wad); }

static size_t W_Embedded_Read(wad_file_t* wad, unsigned int offset,
                              void* buffer, size_t buffer_len) {
  if (offset >= wad->length) return 0;
  if (offset + buffer_len > wad->length) {
    buffer_len = wad->length - offset;
  }
  memcpy(buffer, _embedded_wad_start + offset, buffer_len);
  return buffer_len;
}

/* ------------------------------------------------------------------ */
/* Stdc WAD backend (fopen/fread via semihosting)                     */
/* ------------------------------------------------------------------ */

typedef struct {
  wad_file_t wad;
  FILE* fstream;
} stdc_wad_file_t;

static wad_file_t* W_Stdc_OpenFile(char* path) {
  FILE* fstream = fopen(path, "rb");
  if (fstream == NULL) return NULL;

  /* Get file length. */
  fseek(fstream, 0, SEEK_END);
  long length = ftell(fstream);
  fseek(fstream, 0, SEEK_SET);

  stdc_wad_file_t* result = Z_Malloc(sizeof(stdc_wad_file_t), PU_STATIC, 0);
  result->wad.file_class = &stdc_wad_file;
  result->wad.mapped = NULL;
  result->wad.length = (unsigned int)length;
  result->fstream = fstream;

  return &result->wad;
}

static void W_Stdc_CloseFile(wad_file_t* wad) {
  stdc_wad_file_t* stdc = (stdc_wad_file_t*)wad;
  fclose(stdc->fstream);
  Z_Free(stdc);
}

static size_t W_Stdc_Read(wad_file_t* wad, unsigned int offset, void* buffer,
                          size_t buffer_len) {
  stdc_wad_file_t* stdc = (stdc_wad_file_t*)wad;
  fseek(stdc->fstream, (long)offset, SEEK_SET);
  return fread(buffer, 1, buffer_len, stdc->fstream);
}

/* ------------------------------------------------------------------ */
/* Dispatch: pick embedded or stdc at runtime                         */
/* ------------------------------------------------------------------ */

static wad_file_t* W_Cnidoom_OpenFile(char* path) {
  if (&_embedded_wad_start != NULL) {
    return W_Embedded_OpenFile(path);
  }
  return W_Stdc_OpenFile(path);
}

static void W_Cnidoom_CloseFile(wad_file_t* wad) {
  if (&_embedded_wad_start != NULL) {
    W_Embedded_CloseFile(wad);
  } else {
    W_Stdc_CloseFile(wad);
  }
}

static size_t W_Cnidoom_Read(wad_file_t* wad, unsigned int offset, void* buffer,
                             size_t buffer_len) {
  if (&_embedded_wad_start != NULL) {
    return W_Embedded_Read(wad, offset, buffer, buffer_len);
  }
  return W_Stdc_Read(wad, offset, buffer, buffer_len);
}

wad_file_class_t stdc_wad_file = {
    W_Cnidoom_OpenFile,
    W_Cnidoom_CloseFile,
    W_Cnidoom_Read,
};
