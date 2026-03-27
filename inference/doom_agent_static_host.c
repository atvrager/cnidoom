/*
 * doom_agent_static_host.c — Host backend using the static inference engine.
 *
 * Drop-in replacement for doom_agent_host.cc: implements the same
 * agent_init_host / agent_infer_host / agent_destroy_host API so that
 * doomgeneric_agent_sdl.c works unchanged.
 *
 * Accepts float inputs, quantizes them, runs the code-generated graph,
 * dequantizes outputs.  No TFLM dependency.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "doom_agent.h"
#include "doom_agent_graph.h"

static float vis_scale, st_scale, out_scale;
static int32_t vis_zp, st_zp, out_zp;

static int64_t time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

int agent_init_host(const char* model_path) {
  (void)model_path; /* Model is compiled in — path ignored. */

  graph_get_visual_quant(&vis_scale, &vis_zp);
  graph_get_state_quant(&st_scale, &st_zp);
  graph_get_output_quant(&out_scale, &out_zp);

  fprintf(stderr, "Static host backend initialized (model compiled in)\n");
  return 0;
}

uint8_t agent_infer_host(
    const float frame_nhwc[AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK],
    const float state[20], float out_probs[AGENT_NUM_ACTIONS],
    int64_t* out_inference_us) {
  const int vis_size = AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK;

  /* Quantize float inputs → INT8. */
  int8_t vis_q[AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK];
  for (int i = 0; i < vis_size; i++) {
    int32_t q = (int32_t)roundf(frame_nhwc[i] / vis_scale) + vis_zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    vis_q[i] = (int8_t)q;
  }

  int8_t state_q[20];
  for (int i = 0; i < 20; i++) {
    int32_t q = (int32_t)roundf(state[i] / st_scale) + st_zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    state_q[i] = (int8_t)q;
  }

  /* Run static graph. */
  int8_t output_q[AGENT_NUM_ACTIONS];
  int64_t t0 = time_us();
  run_graph(vis_q, state_q, output_q);
  int64_t t1 = time_us();

  if (out_inference_us != NULL) {
    *out_inference_us = t1 - t0;
  }

  /* Dequantize output → float probabilities. */
  float probs[AGENT_NUM_ACTIONS];
  for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
    probs[i] = (output_q[i] - out_zp) * out_scale;
  }

  if (out_probs != NULL) {
    memcpy(out_probs, probs, AGENT_NUM_ACTIONS * sizeof(float));
  }

  /* Threshold → action bitfield. */
  uint8_t actions = 0;
  for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
    if (probs[i] > 0.5f) {
      actions |= (1u << i);
    }
  }

  return actions;
}

void agent_destroy_host(void) { /* Nothing to free — all static. */ }
