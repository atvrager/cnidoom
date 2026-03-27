/*
 * doom_agent_static.c — Static inference bridge.
 *
 * Implements the tflm_init / tflm_get_state_quant / tflm_infer /
 * tflm_destroy interface expected by doom_agent.c, using the
 * code-generated graph instead of the TFLM interpreter.
 *
 * All weights are const arrays in .rodata.  Activations live in a
 * statically-sized scratch buffer (~14 KB vs TFLM's 64 KB arena).
 */

#include <string.h>

#include "doom_agent.h"
#include "generated/doom_agent_graph.h"

int tflm_init(void) {
  /* Nothing to initialize — all state is static/const. */
  return 0;
}

void tflm_get_state_quant(float* scale, int32_t* zero_point) {
  graph_get_state_quant(scale, zero_point);
}

uint8_t tflm_infer(
    const int8_t frame_nhwc[AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK],
    const int8_t state_q[20]) {
  int8_t output[AGENT_NUM_ACTIONS];

  run_graph(frame_nhwc, state_q, output);

  /* Dequantize output and threshold at 0.5 → action bitfield. */
  float out_scale;
  int32_t out_zp;
  graph_get_output_quant(&out_scale, &out_zp);

  uint8_t actions = 0;
  for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
    float prob = (output[i] - out_zp) * out_scale;
    if (prob > 0.5f) {
      actions |= (1u << i);
    }
  }
  return actions;
}

void tflm_destroy(void) { /* Nothing to free — all static. */ }
