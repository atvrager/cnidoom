/*
 * golden_log.c — CSV logging implementation.
 */

#include "golden_log.h"

#include <stdio.h>
#include <stdlib.h>

static FILE* log_file = NULL;

void golden_log_init(void) {
  const char* path = getenv("DOOM_GOLDEN_LOG");
  if (path == NULL || path[0] == '\0') return;

  log_file = fopen(path, "w");
  if (log_file == NULL) {
    fprintf(stderr, "WARNING: Could not open golden log: %s\n", path);
    return;
  }
  fprintf(log_file, "frame,action_bits,p0,p1,p2,p3,p4,p5,inference_us\n");
  fprintf(stderr, "Golden log: %s\n", path);
}

void golden_log_write(int frame, uint8_t actions,
                      const float probs[AGENT_NUM_ACTIONS],
                      int64_t inference_us) {
  if (log_file == NULL) return;

  fprintf(log_file, "%d,%u", frame, (unsigned)actions);
  for (int i = 0; i < AGENT_NUM_ACTIONS; i++) {
    fprintf(log_file, ",%.4f", probs != NULL ? probs[i] : 0.0f);
  }
  fprintf(log_file, ",%ld\n", (long)inference_us);
}

void golden_log_close(void) {
  if (log_file != NULL) {
    fclose(log_file);
    log_file = NULL;
  }
}
