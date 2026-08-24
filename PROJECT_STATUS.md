# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged; it still has no explicit GPU queue model (`queue_overhead_s=0` in the current experiment).
- Hand-written Reference schedulers: frozen; no longer the validation path.
- HELIX fixed-pipeline Reference remains on draft PR #6 and must not be merged yet. Current diagnostic branch is `exp/helix-queue-overlap-diagnostic`; draft PR #8 is research-only and must not be merged.
- Reference reuses pinned HELIX public runtime with an externally fixed Layer-Level pipeline; HELIX placement/max-flow optimization is not used.
- HELIX simulator is a validated execution reference, not ground truth. Decode Increment latency matches per-token TPOT semantics; current aligned TTFT remains Prefill completion rather than true first-token TTFT.

## Experiment log
- **E6 / HELIX fixed-pipeline integration**: smoke + unit/regression CI PASS; no new serving simulator/scheduler was implemented.
- **E7 / 30 s, 17-request capacity**: Conservative Slow = 1.2031 rps safe (1.2211 unsafe); Fast >= 10.3434 rps (right-censored). HELIX Slow capacity is in [0.008663, 0.008792) rps; Fast is in [0.008792, 0.008921) rps. Fast > Slow is resolved pairwise, while absolute HELIX capacity is not treated as real-system ground truth.
- **E8 / Reference metric semantics**: HELIX represents each Decode token as one Increment and issues the next Increment exactly when the previous one reaches the sink, so Increment source-to-sink latency equals token-to-token TPOT.
- **E9 / passive HELIX queue-overlap diagnostic**: all four known safe/unsafe frontier points reproduced without changing HELIX scheduling. At the safe points, the worst Decode token has zero GPU queue wait and TPOT is ~0.117 s including fixed overhead. At the first unsafe points, `azure-00014` is blocked by the Prefill of exactly one later request, `azure-00015`. Slow unsafe: TPOT 1.1037 s, measured GPU queue wait 0.6692 s, all queue wait attributed to Prefill, plus 0.3174 s network/other residual. Fast unsafe: TPOT 0.8369 s, queue wait 0.7140 s, all queue wait attributed to Prefill, residual only 0.0058 s. Decode/mixed blockers and unexplained queue time are zero. This confirms that the HELIX TPOT cliff is a discrete Prefill-blocks-Decode effect, not a smooth increase in Decode compute time.

## Next
Do not modify Conservative yet. Run a minimal controlled two-request experiment: one Decode victim plus one arriving Prefill blocker. Sweep blocker prompt length and arrival offset, use the same passive tracer, and test whether a simple deterministic Prefill-backlog / blocking upper bound can explain or upper-bound Decode queue delay. Keep the existing SLA-to-network red-line formulation unchanged.
