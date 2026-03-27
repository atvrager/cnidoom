/*
 * test_x86_generic_kernels.c — Generic C kernel implementations under ref_*
 * names for the x86 bit-accuracy test.
 *
 * Same approach as test_rvv_generic_kernels.c — renames generic kernels
 * to ref_* so both reference and AVX2-optimized versions coexist.
 */

/* Rename all kernel functions to ref_* before including generic sources. */
#define kernel_mean_int8 ref_mean_int8

#include "kernels/generic/mean.c" /* NOLINT(build/include) */
