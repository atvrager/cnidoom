/*
 * logistic_lut.c — RVV-optimized INT8 logistic (sigmoid) via 256-entry LUT.
 *
 * Same vector gather pattern as tanh_lut.c.
 * For count=6 (our logistic layer): 1 iteration with vl=6 via vsetvl.
 *
 * Target: RISC-V RV32IMF_Zve32x_Zve32f, VLEN=128.
 */

#include <riscv_vector.h>

#include "../kernel_ops.h"

void kernel_logistic_int8(const int8_t* in, int count, const int8_t lut[256],
                          int8_t* out) {
  int i = 0;
  for (; i < count;) {
    size_t vl = __riscv_vsetvl_e8m1((size_t)(count - i));

    vint8m1_t vin = __riscv_vle8_v_i8m1(in + i, vl);
    vuint8m1_t vidx = __riscv_vreinterpret_v_i8m1_u8m1(vin);
    vuint8m1_t vout_u = __riscv_vluxei8_v_u8m1((const uint8_t*)lut, vidx, vl);
    vint8m1_t vout = __riscv_vreinterpret_v_u8m1_i8m1(vout_u);

    __riscv_vse8_v_i8m1(out + i, vout, vl);
    i += (int)vl;
  }
}
