# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current candidate keeps the original event trajectory and network red-line and charges profiled Prefill blocking-service debt inside Decode's SLA remaining-time budget.
- No scheduler or production Evaluator semantic change has been merged.

## Experiment log
- **E31 candidate**: on the original seed-7 workload, candidate frontier is `0.0131` safe / `0.0132` first-unsafe for both fixed pipelines.
- **E33 direct HELIX refinement**: sampled HELIX feasibility from intensity `0.01320` to `0.01390` in `0.00005` steps; sampled feasibility is monotone for both pipelines, then the first safe/unsafe bracket is bisected to width `3.125e-6`.
- **Slow pipeline HELIX frontier**: safe lower bound `0.013578125`, unsafe upper bound `0.01358125`; corresponding request-rate bracket is `0.00877778–0.00877980 rps`. Relative to candidate safe `0.0131`, HELIX safe lower bound is `+3.6498%` higher.
- **Fast pipeline HELIX frontier**: safe lower bound `0.01370625`, unsafe upper bound `0.013709375`; request-rate bracket `0.00886061–0.00886263 rps`. Relative to candidate safe `0.0131`, HELIX safe lower bound is `+4.6279%` higher.
- At the HELIX frontier, failure is TPOT, not TTFT. The transition is discrete: just below the bracket max TPOT remains about `0.117 s`, while just above it jumps to roughly `0.8–1.18 s`, consistent with the previously observed Prefill/Decode overlap cliff.
- **E34 multi-workload** (separate branch): candidate-safe remains HELIX-safe in 8/8 pipeline×workload checks; candidate first-unsafe remains HELIX-safe in 8/8. +5% above candidate unsafe is HELIX-unsafe in 6/8 and still safe in seed11 Fast / seed19 Slow.
- **E27 transfer** (separate branch): stage-additive Prefill blocking overhead matches direct HELIX service across 7/8/10 stages × 256/512/1024 prompts with max absolute relative error `0.0698798%`.

## Interpretation
E33 quantifies the original seed-7 conservatism instead of only bracketing it qualitatively. The current scheduler-free candidate underestimates the pinned HELIX safe frontier by about `3.65%` (Slow) and `4.63%` (Fast) relative to its own safe intensity on this workload. This is a modest, systematic safety margin rather than the orders-of-magnitude mismatch seen before the compute-side corrections. The remaining gap is tied to coarse hidden execution phase / overlap onset, not to an obviously wrong network red-line or blocking-service magnitude.

## Next
1. Refine the two E34 cases that remained HELIX-safe at +5% (seed11 Fast, seed19 Slow) to measure the largest observed conservatism; E35 is running on a separate draft branch.
2. Use E33+E35 to report a range of observed safety margins across workloads rather than fitting a global discount.
3. Keep the full blocking-service envelope as the leading scheduler-free candidate unless an additional cheap event-level state variable is independently justified.
Do not merge and do not change default Evaluator semantics yet.
