# Project Status

## Current state
- Conservative Evaluator remains unchanged; current explicit queue term is still `queue_overhead_s=0`.
- HELIX fixed-pipeline Reference stays on draft PR #6. Queue diagnostic PR #8 and blocking-sweep PR #9 are research-only; do not merge.
- HELIX is used as a relative execution reference, not absolute real-system ground truth.

## Experiment log
- **E7 / capacity comparison**: Conservative ranks Fast > Slow but estimates much higher safe request rates than HELIX under the current strict SLA criterion.
- **E8 / metric semantics**: HELIX Decode Increment latency is valid for strong per-token TPOT; aligned TTFT is still Prefill completion rather than true first-token TTFT.
- **E9 / frontier queue diagnostic**: the safe→unsafe TPOT cliff is caused by a later Prefill blocking an active Decode. Safe frontier points have 0 GPU queue wait; first unsafe points add ~0.67–0.71 s Prefill-attributed GPU queue wait.
- **E10 / controlled two-query blocking sweep**: one Decode victim plus one later Prefill blocker; prompt lengths {64,256,512,1024}, four arrival offsets, both Slow/Fast fixed pipelines (32 cases total). Isolated Decode service is ~0.1121 s. 64-token Prefill causes no measured Decode queueing in these offsets; larger Prefills create discrete queue delay that grows with prompt length and timing. All measured GPU queue delay is attributed to the Prefill; Decode/mixed blockers and service-time inflation are zero. In every case, measured Decode queue wait is <= the blocker Prefill's full measured pipeline compute service. Worst observed queue/full-Prefill-service ratio is ~0.949 on Fast and ~0.954 on Slow. Thus `full active Prefill compute backlog` is a viable conservative queue-upper-bound candidate for this one-blocker setting. Slow additionally shows network residual, which remains a separate term already handled by the network model.

## Candidate model (not yet adopted)
For an active Decode request at event time t, test a simple bound of the form
`T_queue,D^UB(t) = sum_{r in active Prefill} T_prefill,r^UB`, initially using each active Prefill's full pipeline compute time rather than remaining time. This deliberately over-counts already-executed Prefill work but is simple and conservative. It must still be validated with multiple simultaneous Prefills before modifying Conservative.

## Next
Run a small multi-Prefill additivity test. If the summed Prefill-compute backlog still upper-bounds HELIX Decode queue delay, implement it as a controlled Conservative ablation/new queue model while leaving the SLA-to-network red-line equations unchanged.
