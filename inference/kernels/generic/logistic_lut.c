/*
 * logistic_lut.c — INT8 logistic (sigmoid) via 256-entry LUT (generic C).
 *
 * Same principle as tanh_lut.c but using sigmoid(x) = 1/(1+exp(-x)).
 */

#include "../kernel_ops.h"

void kernel_logistic_int8(const int8_t* in, int count, const int8_t lut[256],
                          int8_t* out) {
  for (int i = 0; i < count; i++) {
    out[i] = lut[(uint8_t)in[i]];
  }
}
