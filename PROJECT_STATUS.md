# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current candidate keeps the original event trajectory and network red-line and charges profiled Prefill blocking-service debt inside Decode's SLA remaining-time budget.
- No scheduler, routing, replica, or production Evaluator semantic change has been merged.

## Experiment log
- **E27 transfer**: stage-additive Prefill blocking overhead matches direct HELIX service across 7/8/10 stages × 256/512/1024 prompts; max absolute relative error `0.0698798%`.
- **E31 baseline**: original seed-7 candidate frontier is `0.0131` safe / `0.0132` first-unsafe for both fixed pipelines.
- **E33 exact seed-7 HELIX frontier**: Slow `0.013578125` safe / `0.01358125` unsafe; Fast `0.01370625` / `0.013709375`. Relative to candidate safe `0.0131`, HELIX safe lower bound is `+3.6498%` Slow and `+4.6279%` Fast. Failure is TPOT.
- **E34 multi-workload validation**: seed3, seed11, seed19, seed7 offset200 × two pipelines = 8 checks. Candidate-safe is HELIX-safe in **8/8**; candidate first-unsafe is still HELIX-safe in **8/8**. +5% above candidate unsafe is HELIX-unsafe in **6/8** and still safe in seed11 Fast and seed19 Slow.
- **E35 seed11 Fast**: candidate first-unsafe is `0.0156`; HELIX refined frontier is `0.01648359375` safe / `0.016486640625` unsafe, width `3.046875e-6`. HELIX safe lower bound is `+5.6641%` above candidate first-unsafe and about `+6.3458%` above candidate safe `0.0155`. First HELIX failure is TPOT; sampled feasibility is monotone.
- **E35 seed19 Slow**: candidate first-unsafe is `0.0153`; HELIX refined frontier is `0.0183779296875` safe / `0.01838091796875` unsafe, width `2.98828125e-6`. HELIX safe lower bound is `+20.1172%` above candidate first-unsafe and about `+20.9074%` above candidate safe `0.0152`. First HELIX failure is TPOT; sampled feasibility is monotone. Aligned TTFT remains below the `2.0 s` SLA (`1.88964 s`).

## Interpretation
E35 confirms safety but exposes a real tightness problem. Across completed exact refinements, the scheduler-free full blocking-service envelope is conservative by roughly `3.65%–20.91%` relative to candidate-safe intensity, depending on workload/pipeline. The seed11 Fast case remains moderate, but seed19 Slow is materially over-conservative (~21%), so it is premature to call the current candidate tight enough for Evaluator-v1. The evidence still does **not** support a global discount: E28–E30 showed hidden phase can consume `95.3143%` of full debt and simple slack/discount rules are not safe. The next question is therefore diagnostic: why does the same coarse `N_P/N_D` state produce a ~21% frontier gap in seed19 Slow, and can any additional cheap event-level observable explain it without reconstructing scheduler/stage phase?

## Next
1. Diagnose seed19 Slow around candidate first-unsafe versus HELIX first-unsafe: compare request/event states, prompt sizes, Decode contexts, and active Prefill blocking-service debt, while keeping scheduler phase hidden.
2. Test only cheap event-level observables already available to the Evaluator before proposing any tightening rule.
3. Keep full blocking-service debt as the safety baseline; do not introduce a fitted global multiplier or modify default Evaluator semantics yet.
4. Do not merge.
