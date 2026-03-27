/*
 * tanh_lut.c — INT8 tanh via 256-entry lookup table (generic C).
 *
 * The codegen precomputes the LUT from the layer's quantization params:
 *   for each uint8 index i (representing signed int8 value i-128):
 *     float x = (i - 128 - in_zp) * in_scale;
 *     float y = tanh(x);
 *     lut[i] = clamp(round(y / out_scale) + out_zp, -128, 127);
 */

#include "../kernel_ops.h"

void kernel_tanh_int8(const int8_t* in, int count, const int8_t lut[256],
                      int8_t* out) {
  for (int i = 0; i < count; i++) {
    out[i] = lut[(uint8_t)in[i]];
  }
}
