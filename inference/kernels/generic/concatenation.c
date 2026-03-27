/*
 * concatenation.c — INT8 concatenation of two vectors (generic C).
 *
 * Assumes both inputs share the same quantization domain (scale and
 * zero_point match), so no requantization is needed — just memcpy.
 */

#include <string.h>

#include "../kernel_ops.h"

void kernel_concatenation_int8(const int8_t* a, int a_len, const int8_t* b,
                               int b_len, int8_t* out) {
  /* Handle overlap: if out == a, skip the first copy. */
  if (out != a) {
    memcpy(out, a, (size_t)a_len);
  }
  memcpy(out + a_len, b, (size_t)b_len);
}
