# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current candidate keeps the original event trajectory and network red-line, but charges profiled Prefill blocking-service debt inside Decode's SLA remaining-time budget.
- No scheduler, routing, replica, or production Evaluator semantic change has been merged.

## Experiment log
- **E23/E26 magnitude**: compute-only Prefill debt covers 42/44 controlled E10/E11 cases; blocking-service debt covers 44/44. Independently calibrated E22 overhead reconstructs E10 Prefill compute-node service within `0.05710%` maximum relative error.
- **E27 transfer**: stage-additive Prefill blocking overhead was tested directly on 7/8/10 stages × 256/512/1024 prompts. All 9/9 cases closely match; max absolute relative error is `0.0698798%`, including held-out 7- and 10-stage configurations. This supports a stage-local profiling interface within the pinned HELIX configuration, not a universal coefficient.
- **E31 seed-7 baseline**: candidate frontier is `0.0131` safe / `0.0132` unsafe for both fixed pipelines. Candidate-safe is HELIX-safe; HELIX is still safe at candidate first-unsafe and unsafe by `0.01386` (+5%). First candidate violation is interpretable `sla_time` exhaustion at `N_P=1, N_D=1`, not a network-only artifact.
- **E34 multi-workload validation**: held-out workloads seed3, seed11, seed19, and seed7 offset200 completed for both fixed pipelines (8 pipeline×workload checks). Candidate frontiers are: seed3 `0.0152/0.0153`, seed11 `0.0155/0.0156`, seed19 `0.0152/0.0153`, seed7 offset200 `0.0087/0.0088` (safe/first-unsafe intensity).
- **E34 safety result**: candidate-safe point is HELIX-safe in **8/8** checks; no observed optimism. Candidate first-unsafe point is still HELIX-safe in **8/8**, confirming systematic conservatism at the coarse overlap-onset frontier.
- **E34 +5% probe**: HELIX is unsafe in **6/8** checks and still safe in 2/8 (seed11 Fast, seed19 Slow). Thus the candidate is conservative but generally within a small workload-dependent margin of an observed HELIX failure; exact frontier tightness is not yet quantified.
- **E33 frontier refinement**: direct HELIX refinement for the original seed-7 workload is currently running on branch `exp/e31-helix-frontier-refinement` / draft PR #36.

## Interpretation
The current evidence now separates three issues cleanly. (1) The blocking-service debt magnitude has direct controlled support. (2) Its profiling overhead transfers across tested stage counts as a stage-local term. (3) When inserted into the original Decode SLA budget, the resulting capacity candidate remains conservative across four workload variants and two fixed pipelines, with 0/8 observed optimism. The remaining weakness is tightness: the coarse scheduler-free state creates an overlap-onset cliff, so candidate first-unsafe occurs before HELIX actually violates SLA. E28–E30 already show that simple global discounts/slacks are not defensible fixes.

## Next
1. Finish E33 and quantify the exact HELIX frontier gap for the original seed-7 workload.
2. If E33 confirms a modest but systematic gap, run one focused frontier-refinement experiment on the two E34 cases where +5% remained HELIX-safe (seed11 Fast, seed19 Slow) to measure the worst observed conservatism.
3. Keep the full blocking-service envelope as the leading scheduler-free candidate unless an additional cheap event-level state variable is independently justified.
Do not merge and do not change default Evaluator semantics yet.
