/*
 * test_rvv_generic_kernels.c — Generic C kernel implementations under ref_*
 * names.
 *
 * This translation unit re-exports the generic (scalar) kernels with a "ref_"
 * prefix so that the bit-accuracy test can call both the reference and the
 * target-optimized (RVV) implementations in the same binary.
 *
 * The #define trick renames every kernel_* symbol before #including the
 * generic .c source files.  The kernel_ops.h header declarations also get
 * renamed, which is fine — we only need the ref_* function definitions
 * from this TU.
 */

/* Rename all kernel functions to ref_* before including generic sources. */
#define kernel_depthwise_conv2d_int8 ref_depthwise_conv2d_int8
#define kernel_conv2d_int8 ref_conv2d_int8
#define kernel_fully_connected_int8 ref_fully_connected_int8
#define kernel_tanh_int8 ref_tanh_int8
#define kernel_logistic_int8 ref_logistic_int8
#define kernel_concatenation_int8 ref_concatenation_int8

#include "kernels/generic/conv2d.c"           /* NOLINT(build/include) */
#include "kernels/generic/depthwise_conv2d.c" /* NOLINT(build/include) */
#include "kernels/generic/fully_connected.c"  /* NOLINT(build/include) */
#include "kernels/generic/logistic_lut.c"     /* NOLINT(build/include) */
#include "kernels/generic/tanh_lut.c"         /* NOLINT(build/include) */
