# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current magnitude candidate is **Prefill blocking-service debt = profiled Prefill compute + profiled/bounded blocking runtime overhead**. Network red-line equations remain unchanged.
- E24 budget-consistent debt-in-Delta ablation and E27 stage-count transfer are prepared separately, but direct execution is currently blocked by GitHub Actions runner failure before any step starts.

## Experiment log
- **E15–E16 / historical candidate frontier**: exact singleton Decode + full active-Prefill compute debt stayed conservative across tested workload variants; no candidate optimism was observed, but Slow/Fast tied at the candidate frontier.
- **E23 correction / E26 follow-up**: profiler-only Prefill compute covers 42/44 controlled E10/E11 interference cases. Adding the independently calibrated E22 blocking/runtime overhead gives 44/44 coverage; the tightest service-debt ratio is `0.953143`.
- **E25 / frontier-cliff audit**: re-read the successful E15/E16 artifacts for 5 workload variants x 2 pipelines (10 first-unsafe cases). Every first-unsafe historical-candidate state is exactly `N_P=1, N_D=1`, with exact-singleton Decode compute `0.112 s` and fixed overhead `0.005 s`, leaving only `0.033 s` of the 0.150-s TPOT before any Prefill debt. The smallest historical compute debt is `0.11584 s` and the largest is `0.68192 s`, i.e. `3.51x–20.66x` the no-debt residual budget. Consequently the first-unsafe compute-side requirement exceeds TPOT by `82.84–648.92 ms`, rather than crossing it marginally. Slow/Fast first-unsafe states are identical in all 5 workloads.

## Interpretation
E25 identifies a **timing/exposure** problem, not a magnitude-calibration result. Even the smaller historical compute-only debt produces a discrete Prefill+Decode overlap-onset cliff when charged immediately. The corrected E23/E26 result separately says the magnitude bound should use blocking-service debt, not compute-only debt; that larger magnitude would not remove the cliff. Therefore keep the two questions separate: (1) how large can Prefill-induced blocking be, and (2) when/how much of that upper bound should be exposed to an active Decode in a scheduler-free event state.

## Next
Do not fit a multiplicative discount and do not reconstruct HELIX scheduling. Use archived E10 timing sweeps to test whether exposure behaves monotonically with simple time-since-overlap / remaining-service notions. If it is phase-sensitive, record that as evidence that exact exposure timing is scheduler/stage-phase dependent and keep the Conservative Evaluator as a worst-case screening bound rather than forcing a fragile timing heuristic.
