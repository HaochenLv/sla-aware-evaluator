# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free safety candidate. E39 is rejected; E40 is a direct phase-jitter safety stress test of unchanged E31.

## Experiment log
- **E31/E34 pointwise safety evidence**: previously tested candidate-safe points were HELIX-safe in all 8 workload×pipeline checks. E39 showed finite-trace HELIX feasibility is not globally monotone in intensity, so these are pointwise safety checks rather than proof of one unique HELIX frontier.
- **E36/E37**: seed19 Slow conservatism combines hidden phase-dependent Prefill exposure and trajectory mismatch. Cheap event/request observables cannot uniquely determine exposure.
- **E38**: accumulating 5 ms fixed overhead as physical time on each Decode token contributes 2.075 s over the 415-token victim and can create a false overlap, but simply removing it over-corrects relative to HELIX.
- **E39 rejected**: split progression/accounting semantics moves the candidate to 0.0159 safe but creates Slow optimism: HELIX is TPOT-unsafe at that candidate-safe point. E39 also exposes nonmonotone HELIX safe/unsafe pockets under finite-trace intensity scaling.
- **E40 setup**: keep original E31 unchanged at seed19 Slow candidate-safe intensity `0.0152`. Jitter only `azure-00010` arrival from `-120 ms` to `+120 ms` in `10 ms` steps (25 cases). For every phase perturbation compare E31 feasibility with pinned HELIX.
- **E40 result**: `24/25` cases are `both_safe`; `1/25` is `candidate_conservative` (E31 unsafe while HELIX safe); `0/25` are `candidate_optimism`; HELIX is unsafe in `0/25` cases. Maximum HELIX TPOT over the full jitter grid is `0.1174260288 s < 0.150 s`.
- **E40 conservative case**: at jitter `-120 ms`, E31 sees an Arrival-triggered `N_P=1,N_D=1` overlap and charges full Prefill blocking-service debt `0.126587275 s`, giving required compute-side time `0.243587275 s > 0.150 s`; HELIX remains safe with max TPOT `0.117426029 s`.

## Interpretation
E40 is positive safety evidence for the original E31 envelope on the diagnosed seed19 Slow safe point. Across a ±120 ms phase perturbation of the known interfering request, no case was found where E31 stayed safe while HELIX violated TPOT. The only disagreement is in the conservative direction. This supports the intended scheduler-free worst-case role of full Prefill blocking-service debt against local arrival-phase variation.

The result is not a proof of universal safety: it covers one workload, one pipeline, one base intensity, one known interferer, and a ±120 ms jitter window. Because HELIX feasibility is phase-sensitive and nonmonotone in intensity, validation should continue with broader phase/prefix stress rather than relying on a single binary-search frontier.

## Next
1. Extend the phase-jitter/prefix safety audit to the remaining E34 workload×pipeline candidate-safe points, prioritizing seed11 Fast and the other Slow/Fast cases.
2. Search for any `candidate safe / HELIX unsafe` counterexample; reject any tightening rule immediately if one appears.
3. Keep E31 unchanged as the safety baseline; do not revive E39 split semantics.
4. Do not merge and do not modify main.
