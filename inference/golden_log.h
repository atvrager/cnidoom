/*
 * golden_log.h — CSV logging of per-frame agent decisions.
 *
 * When enabled (via DOOM_GOLDEN_LOG env var), logs each inference
 * step to a CSV file for offline comparison between model variants.
 *
 * Format: frame,action_bits,p0,p1,p2,p3,p4,p5,inference_us
 */

#ifndef GOLDEN_LOG_H
#define GOLDEN_LOG_H

#include <stdint.h>

#include "doom_agent.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize golden logging. Checks DOOM_GOLDEN_LOG env var.
 * If set, opens the file for writing. Safe to call even if
 * logging is not desired (becomes a no-op).
 */
void golden_log_init(void);

/*
 * Log one inference step.
 * frame: frame counter
 * actions: action bitfield
 * probs: 6 action probabilities (may be NULL — logs 0.0 for each)
 * inference_us: inference time in microseconds
 */
void golden_log_write(int frame, uint8_t actions,
                      const float probs[AGENT_NUM_ACTIONS],
                      int64_t inference_us);

/*
 * Flush and close the log file.
 */
void golden_log_close(void);

#ifdef __cplusplus
}
#endif

#endif /* GOLDEN_LOG_H */
