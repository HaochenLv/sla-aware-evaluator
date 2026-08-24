# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free safety candidate. E39 is rejected; E40/E41 stress the unchanged E31 candidate rather than tightening it.

## Experiment log
- **E31/E34**: candidate-safe points were HELIX-safe in all 8 workload×pipeline checks, but E39 showed finite-trace HELIX feasibility can be nonmonotone in intensity; earlier binary-search “exact frontier” language is therefore invalid.
- **E36–E38**: seed19 Slow conservatism comes from hidden phase-dependent Prefill exposure plus trajectory mismatch. Cheap request/event state cannot determine exposure, and simply separating fixed SLA overhead from physical progression over-corrects.
- **E39 rejected**: split semantics creates candidate optimism on seed19 Slow (`0.0159` candidate-safe while HELIX max TPOT is `0.168847 s > 0.150 s`).
- **E40 local phase stress**: seed19 Slow at original candidate-safe `0.0152`, jitter `azure-00010` by ±120 ms in 10 ms steps. `24/25` both-safe, `1/25` candidate-conservative, `0/25` candidate-optimistic, HELIX unsafe `0/25`; max HELIX TPOT `0.117426029 s`.
- **E41 moderate cross-workload stress**: extend to the remaining seven E34 workload×pipeline candidate-safe points. For each case, identify the Prefill that triggers the first E31 unsafe overlap at the known first-unsafe intensity, then at the known safe intensity jitter only that Prefill by ±80 ms in 20 ms steps. This window is intentionally moderate and below one isolated Decode-token interval (~112 ms) in the pinned setup.
- **E41 execution**: Actions run `32714461575` completed all 7 matrix jobs successfully and produced 7 artifacts.
- **E41 result**: 7 workload×pipeline cases × 9 jitters = `63` checks. `61/63` are both-safe, `2/63` are candidate-conservative, `0/63` are candidate-optimistic, and HELIX is unsafe in `0/63`. Maximum HELIX TPOT across E41 is `0.117426029 s < 0.150 s`.
- **E41 conservative cases**: both occur for seed11 at jitter `-80 ms` (Slow and Fast). E31 creates an Arrival-triggered `N_P=1,N_D=1` overlap and charges `0.141973573 s` full Prefill blocking debt, giving required time `0.258973573 s > 0.150 s`, while HELIX remains safe.
- **Combined E40+E41 phase stress**: `88` targeted, reasonable arrival-phase perturbations around known candidate-safe points; `85` both-safe, `3` candidate-conservative, `0` candidate-optimistic. This is strong empirical safety evidence, not a proof.

## Judgment
E41 did not break the unchanged E31 candidate under moderate phase perturbations across the remaining E34 workload/pipeline cases. The observed mismatches remain one-sided: E31 sometimes rejects a HELIX-safe case, but no tested case has E31 saying safe while HELIX violates SLA. This strengthens the robustness case for the full blocking-service envelope while preserving the known tightness cost.

## Next
1. Run a small second-order stress on a few representative candidate-safe cases by perturbing more than one nearby arrival within realistic tens-of-milliseconds noise, rather than expanding to extreme timing shifts.
2. Keep the primary falsification criterion `E31 safe / HELIX unsafe`; stop and diagnose immediately if one appears.
3. Keep E31 unchanged; do not merge and do not modify main.
