# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Best compute-side candidate remains exact HELIX singleton Decode alignment + unit full active-Prefill interference debt. Network red-line equations remain unchanged.
- E24 budget-consistent debt-in-Delta ablation is prepared separately on draft PR #27 but direct execution is currently blocked by GitHub Actions runner failure before any step starts.

## Experiment log
- **E15–E16 / candidate frontier**: exact singleton Decode + full active-Prefill debt stayed conservative across tested workload variants; no candidate optimism was observed, but Slow/Fast tied at the candidate frontier.
- **E23 / total Decode compute-side excess**: full active-Prefill debt upper-bounds queue wait + positive Decode service inflation in all 44/44 controlled HELIX cases. Worst observed excess/debt ratio is `0.953687`, so a global debt discount below 1 is not supported.
- **E25 / frontier-cliff audit**: re-read the successful E15/E16 artifacts for 5 workload variants x 2 pipelines (10 first-unsafe cases). Every first-unsafe candidate state is exactly `N_P=1, N_D=1`, with exact-singleton Decode compute `0.112 s` and fixed overhead `0.005 s`, leaving only `0.033 s` of the 0.150-s TPOT before any Prefill debt. The smallest observed active-Prefill debt is `0.11584 s` and the largest is `0.68192 s`, i.e. `3.51x–20.66x` the no-debt residual budget. Consequently the first-unsafe compute-side requirement exceeds TPOT by `82.84–648.92 ms`, rather than crossing it marginally. Slow/Fast first-unsafe states are identical in all 5 workloads.

## Interpretation
E25 identifies why the current candidate is safe but coarse on finite traces: its capacity frontier is a discrete **Prefill+Decode overlap-onset cliff**, not a smooth resource-budget crossing. Once the first relevant Prefill becomes active beside a singleton Decode, charging its full compute debt immediately overwhelms the remaining TPOT budget. This explains the repeated Slow/Fast ties without requiring any change to the network equation. It does not invalidate the unit full-debt upper bound itself: E23 shows that discounting the debt globally would break observed controlled cases. The unresolved issue is instead **when and how much of an active Prefill's debt should be exposed to a Decode at a given event state**.

## Next
Do not fit a multiplicative discount and do not reconstruct HELIX scheduling. The next useful controlled test is a remaining-exposure variant: compare full active-Prefill debt against a simple event-available remaining-Prefill debt (or equivalent remaining blocking exposure) using the archived E10/E11 tracer data where possible. If the existing artifacts are insufficient to reconstruct remaining exposure exactly, add only the missing timestamps/remaining-service fields to the passive tracer and run it when Actions runners recover. Keep E24 pending in parallel.
