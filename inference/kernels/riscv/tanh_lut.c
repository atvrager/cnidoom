/*
 * tanh_lut.c — RVV-optimized INT8 tanh via 256-entry lookup table.
 *
 * Uses vluxei8 (byte-indexed vector gather) to look up 16 LUT entries
 * per iteration.  For count=64 (our tanh layer): 4 iterations of vl=16.
 *
 * Target: RISC-V RV32IMF_Zve32x_Zve32f, VLEN=128.
 */

#include <riscv_vector.h>

#include "../kernel_ops.h"

void kernel_tanh_int8(const int8_t* in, int count, const int8_t lut[256],
                      int8_t* out) {
  int i = 0;
  for (; i < count;) {
    size_t vl = __riscv_vsetvl_e8m1((size_t)(count - i));

    /* Load input elements. */
    vint8m1_t vin = __riscv_vle8_v_i8m1(in + i, vl);

    /* Reinterpret signed int8 as unsigned for LUT indexing.
     * This maps [-128..127] → [0..255], covering all 256 LUT entries. */
    vuint8m1_t vidx = __riscv_vreinterpret_v_i8m1_u8m1(vin);

    /* Gather: out[lane] = lut[vidx[lane]].
     * vluxei8 with byte offsets indexes the full 256-entry table. */
    vuint8m1_t vout_u = __riscv_vluxei8_v_u8m1((const uint8_t*)lut, vidx, vl);
    vint8m1_t vout = __riscv_vreinterpret_v_u8m1_i8m1(vout_u);

    __riscv_vse8_v_i8m1(out + i, vout, vl);
    i += (int)vl;
  }
}
