/*
 * test_static_vs_tflm.cc — Verify static engine matches TFLM byte-for-byte.
 *
 * Loads the .tflite model via TFLM, runs inference through both backends
 * on random INT8 inputs, and checks that raw INT8 outputs are identical.
 *
 * Build: cmake -B build -S inference -DDOOM_AGENT_STATIC=ON
 * Run:   build/test_static_vs_tflm --model models/doom_agent_int8.tflite
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "doom_agent.h"
#include "generated/doom_agent_graph.h"

/* TFLM headers for reference backend. */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static constexpr int kVisualSize =
    AGENT_FRAME_H * AGENT_FRAME_W * AGENT_FRAME_STACK;
static constexpr int kStateSize = 20;
static constexpr int kOutputSize = AGENT_NUM_ACTIONS;

/* ------------------------------------------------------------------ */
/* TFLM reference backend                                              */
/* ------------------------------------------------------------------ */

static constexpr int kArenaSize = 1024 * 1024;
static uint8_t tensor_arena[kArenaSize];

static uint8_t* load_file(const char* path, size_t* out_len) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) return nullptr;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  auto* buf = static_cast<uint8_t*>(malloc(len));
  if (buf == nullptr) {
    fclose(f);
    return nullptr;
  }
  if (fread(buf, 1, len, f) != static_cast<size_t>(len)) {
    free(buf);
    fclose(f);
    return nullptr;
  }
  fclose(f);
  *out_len = static_cast<size_t>(len);
  return buf;
}

/* ------------------------------------------------------------------ */
/* PRNG (xoshiro128+)                                                  */
/* ------------------------------------------------------------------ */

static uint32_t rng_s[4];

static void rng_seed(uint32_t seed) {
  rng_s[0] = seed;
  rng_s[1] = seed ^ 0x12345678;
  rng_s[2] = seed ^ 0xDEADBEEF;
  rng_s[3] = seed ^ 0xCAFEBABE;
}

static uint32_t rng_next(void) {
  uint32_t t = rng_s[1] << 9;
  rng_s[2] ^= rng_s[0];
  rng_s[3] ^= rng_s[1];
  rng_s[1] ^= rng_s[2];
  rng_s[0] ^= rng_s[3];
  rng_s[2] ^= t;
  rng_s[3] = (rng_s[3] << 11) | (rng_s[3] >> 21);
  return rng_s[0] + rng_s[3];
}

static int8_t rng_int8(void) { return static_cast<int8_t>(rng_next() & 0xFF); }

/* ------------------------------------------------------------------ */
/* Test driver                                                         */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
  /* Parse --model arg. */
  const char* model_path = nullptr;
  int num_trials = 1000;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
      num_trials = atoi(argv[++i]);
    }
  }
  if (model_path == nullptr) {
    fprintf(stderr, "Usage: %s --model <path.tflite> [--trials N]\n", argv[0]);
    return 1;
  }

  /* Load model. */
  size_t model_len = 0;
  uint8_t* model_data = load_file(model_path, &model_len);
  if (model_data == nullptr) {
    fprintf(stderr, "ERROR: Failed to load model: %s\n", model_path);
    return 1;
  }
  fprintf(stderr, "Loaded model: %s (%zu bytes)\n", model_path, model_len);

  /* Initialize TFLM. */
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model == nullptr) {
    fprintf(stderr, "ERROR: Invalid model\n");
    return 1;
  }

  static tflite::MicroMutableOpResolver<16> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddLogistic();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddTranspose();
  resolver.AddTanh();

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "ERROR: AllocateTensors failed\n");
    return 1;
  }

  /* Bind TFLM tensors. */
  TfLiteTensor* tflm_state = interpreter.input(0);
  TfLiteTensor* tflm_visual = interpreter.input(1);
  TfLiteTensor* tflm_output = interpreter.output(0);

  /* Auto-swap if indices are reversed (visual is 4D). */
  if (tflm_state->dims->size == 4) {
    TfLiteTensor* tmp = tflm_state;
    tflm_state = tflm_visual;
    tflm_visual = tmp;
  }

  fprintf(stderr, "TFLM arena used: %zu bytes\n",
          interpreter.arena_used_bytes());

  /* Run comparison. */
  rng_seed(42);
  int pass = 0;
  int fail = 0;

  for (int trial = 0; trial < num_trials; trial++) {
    /* Generate random INT8 inputs. */
    int8_t visual[kVisualSize];
    int8_t state[kStateSize];
    for (int i = 0; i < kVisualSize; i++) visual[i] = rng_int8();
    for (int i = 0; i < kStateSize; i++) state[i] = rng_int8();

    /* Run TFLM. */
    std::memcpy(tflm_visual->data.int8, visual, kVisualSize);
    std::memcpy(tflm_state->data.int8, state, kStateSize);
    if (interpreter.Invoke() != kTfLiteOk) {
      fprintf(stderr, "ERROR: TFLM invoke failed at trial %d\n", trial);
      return 1;
    }
    int8_t tflm_out[kOutputSize];
    std::memcpy(tflm_out, tflm_output->data.int8, kOutputSize);

    /* Run static engine. */
    int8_t static_out[kOutputSize];
    run_graph(visual, state, static_out);

    /* Compare byte-for-byte. */
    bool match = true;
    for (int i = 0; i < kOutputSize; i++) {
      if (tflm_out[i] != static_out[i]) {
        match = false;
        break;
      }
    }

    if (match) {
      pass++;
    } else {
      fail++;
      if (fail <= 5) {
        fprintf(stderr, "MISMATCH at trial %d: TFLM=[", trial);
        for (int i = 0; i < kOutputSize; i++)
          fprintf(stderr, "%d%s", tflm_out[i], i < kOutputSize - 1 ? "," : "");
        fprintf(stderr, "] STATIC=[");
        for (int i = 0; i < kOutputSize; i++)
          fprintf(stderr, "%d%s", static_out[i],
                  i < kOutputSize - 1 ? "," : "");
        fprintf(stderr, "]\n");
      }
    }
  }

  fprintf(stderr, "\nResults: %d/%d passed", pass, num_trials);
  if (fail > 0) {
    fprintf(stderr, " (%d FAILED)\n", fail);
  } else {
    fprintf(stderr, " — BYTE-IDENTICAL\n");
  }

  free(model_data);

  /* Use _Exit to avoid TFLM static destructor ordering issues. */
  _Exit(fail > 0 ? 1 : 0);
}
