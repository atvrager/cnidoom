/*
 * doom_agent_preprocess.c — Pure preprocessing and quantization functions.
 *
 * No external dependencies — safe to link into test binaries without
 * pulling in TFLite Micro or doomgeneric.
 */

#include <math.h>
#include <stdint.h>

#include "doom_agent.h"

void agent_preprocess_frame(const uint32_t* rgba, int w, int h,
                            int8_t out[AGENT_FRAME_H][AGENT_FRAME_W]) {
  const int sx = w / AGENT_FRAME_W;
  const int sy = h / AGENT_FRAME_H;
  const int area = sx * sy;

  for (int y = 0; y < AGENT_FRAME_H; y++) {
    for (int x = 0; x < AGENT_FRAME_W; x++) {
      int sum = 0;
      for (int dy = 0; dy < sy; dy++) {
        const int src_y = y * sy + dy;
        if (src_y >= h) break;
        for (int dx = 0; dx < sx; dx++) {
          const int src_x = x * sx + dx;
          if (src_x >= w) break;
          const uint32_t pixel = rgba[src_y * w + src_x];
          const uint8_t r = (pixel >> 16) & 0xFF;
          const uint8_t g = (pixel >> 8) & 0xFF;
          const uint8_t b = pixel & 0xFF;
          sum += (r * 77 + g * 150 + b * 29) >> 8;
        }
      }
      const uint8_t gray = (uint8_t)(sum / area);
      out[y][x] = (int8_t)(gray - 128);
    }
  }
}

void agent_quantize_state(const agent_game_state_t* gs, int8_t out[20],
                          float scale, int32_t zero_point) {
  const float* raw = (const float*)gs;
  for (int i = 0; i < 20; i++) {
    int32_t q = (int32_t)roundf(raw[i] / scale) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    out[i] = (int8_t)q;
  }
}
