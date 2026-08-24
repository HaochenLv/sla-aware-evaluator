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
- **E35 focused worst-case refinement**: seed11 Fast candidate first-unsafe `0.0156`; HELIX frontier `0.01648359375` safe / `0.016486640625` unsafe. HELIX safe lower bound is `+5.6641%` above candidate first-unsafe and about `+6.3458%` above candidate safe `0.0155`.
- **E35 seed19 Slow**: candidate first-unsafe `0.0153`; HELIX frontier `0.0183779296875` safe / `0.01838091796875` unsafe. HELIX safe lower bound is `+20.1172%` above candidate first-unsafe and about `+20.9074%` above candidate safe `0.0152`. Sampled feasibility is monotone and first HELIX failure remains TPOT; aligned TTFT stays below `2.0 s` (`1.88964 s`).

## Interpretation
E35 preserves the safety story but reveals a material tightness problem in seed19 Slow. Across the exact E33/E35 frontier measurements, observed conservatism now ranges from roughly `3.65%` to `20.91%`. The full blocking-service envelope remains the supported scheduler-free safety baseline, but seed19 Slow should be diagnosed before any tightening rule is proposed. A global discount is not justified because prior hidden-phase evidence reaches `95.3143%` of full blocking-service debt.

## Next
1. Diagnose the seed19 Slow first-unsafe overlap and compare it with actual HELIX blocking/queue exposure near the candidate and true frontier.
2. Search only for cheap event-level observables already compatible with the Evaluator abstraction; do not reconstruct the HELIX scheduler.
3. Do not merge and do not change default Evaluator semantics yet.
