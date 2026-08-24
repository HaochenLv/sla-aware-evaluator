# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free safety candidate. E39 is rejected; E40 directly stress-tests unchanged E31 against arrival-phase variation.

## Experiment log
- **E31/E34**: prior candidate-safe checks were HELIX-safe in all tested workload×pipeline points, but E39 showed HELIX finite-trace feasibility is nonmonotone in intensity, so binary-search “exact frontier” claims are invalid.
- **E36–E38**: seed19 Slow conservatism comes from hidden phase-dependent Prefill exposure plus trajectory mismatch. Cheap request/event state cannot determine exposure; simply separating fixed SLA overhead from physical progression over-corrects.
- **E39 rejected**: split semantics makes seed19 Slow candidate-safe at `0.0159` while HELIX is already TPOT-unsafe (`0.168847 s > 0.150 s`).
- **E40 setup**: keep original E31 at seed19 Slow candidate-safe intensity `0.0152`; jitter only `azure-00010` arrival from `-120 ms` to `+120 ms` in `10 ms` steps (25 cases).
- **E40 result**: `24/25` cases are both safe; `1/25` is candidate-conservative; `0/25` are candidate-optimistic; HELIX is unsafe in `0/25`. Maximum HELIX TPOT is `0.117426029 s < 0.150 s`.
- **E40 conservative case**: jitter `-120 ms` creates an E31 `N_P=1,N_D=1` Arrival overlap; full Prefill debt is `0.126587275 s`, so required time becomes `0.243587275 s > 0.150 s`, while HELIX remains safe.

## Judgment
E40 is positive local safety evidence for E31: within ±120 ms phase perturbation of the known seed19 interferer, no `E31 safe / HELIX unsafe` counterexample appears. The only mismatch is conservative. This supports the scheduler-free worst-case envelope, but is not universal proof because it covers one workload, one pipeline, one base intensity, one interferer, and one jitter window.

## Next
1. Extend the same phase/prefix safety stress to remaining E34 candidate-safe workload×pipeline points.
2. Prioritize searching for any `candidate safe / HELIX unsafe` counterexample; reject tightening rules if found.
3. Keep E31 unchanged; do not merge and do not modify main.
