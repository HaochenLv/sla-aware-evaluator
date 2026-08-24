# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current candidate keeps the original event trajectory and network red-line and charges profiled Prefill blocking-service debt inside Decode's SLA remaining-time budget.
- No scheduler, routing, replica, or production Evaluator semantic change has been merged.

## Experiment log
- **E27 transfer**: stage-additive Prefill blocking overhead matches direct HELIX service across 7/8/10 stages × 256/512/1024 prompts; max absolute relative error `0.0698798%`.
- **E31 baseline**: original seed-7 candidate frontier is `0.0131` safe / `0.0132` first-unsafe for both fixed pipelines.
- **E33 exact seed-7 HELIX frontier**: Slow `0.013578125` safe / `0.01358125` unsafe; Fast `0.01370625` / `0.013709375`. Relative to candidate safe `0.0131`, the HELIX safe lower bound is `+3.6498%` Slow and `+4.6279%` Fast. Sampled feasibility is monotone; failure is TPOT and remains cliff-like.
- **E34 multi-workload validation**: seed3, seed11, seed19, seed7 offset200 × two pipelines = 8 checks. Candidate-safe is HELIX-safe in **8/8**; candidate first-unsafe is still HELIX-safe in **8/8**. +5% above candidate unsafe is HELIX-unsafe in **6/8** and still safe in two cases: seed11 Fast and seed19 Slow.
- **E35 focused worst-case refinement**: keep the E31 candidate unchanged and refine only those two +5%-still-safe E34 cases. Probe a 1.00–1.30 factor ladder above candidate first-unsafe, check sampled monotonicity, then bisect the first observed safe/unsafe HELIX bracket to `5e-6` intensity width. Run `32707761379` is currently executing both cases in parallel.

## Interpretation
Completed evidence shows no observed optimism across the tested workloads, while E33 quantifies a modest `3.65–4.63%` safety margin on the original seed-7 workload. E35 is designed to measure the largest observed margin rather than inventing a global discount. This keeps safety evidence and tightness evidence separate.

## Next
1. Finish E35 and report the observed safety-margin range across E33/E35.
2. If E35 remains within a moderate margin, treat the full blocking-service envelope as the leading scheduler-free Evaluator-v1 candidate and move toward broader configuration validation rather than coefficient tuning.
3. Do not merge and do not change default Evaluator semantics yet.
