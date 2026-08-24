# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free safety candidate. E39 is rejected; E40–E42 stress the unchanged E31 candidate rather than tightening it.

## Experiment log
- **E31/E34**: candidate-safe points were HELIX-safe in all 8 workload×pipeline checks, but E39 showed finite-trace HELIX feasibility can be nonmonotone in intensity; earlier binary-search “exact frontier” language is therefore invalid.
- **E36–E38**: seed19 Slow conservatism comes from hidden phase-dependent Prefill exposure plus trajectory mismatch. Cheap request/event state cannot determine exposure, and simply separating fixed SLA overhead from physical progression over-corrects.
- **E39 rejected**: split semantics creates candidate optimism on seed19 Slow (`0.0159` candidate-safe while HELIX max TPOT is `0.168847 s > 0.150 s`).
- **E40 local phase stress**: seed19 Slow at original candidate-safe `0.0152`, jitter one critical Prefill by ±120 ms in 10 ms steps. `24/25` both-safe, `1/25` candidate-conservative, `0/25` candidate-optimistic, HELIX unsafe `0/25`; max HELIX TPOT `0.117426029 s`.
- **E41 moderate cross-workload stress**: remaining seven E34 workload×pipeline candidate-safe points, one critical Prefill jittered by ±80 ms in 20 ms steps. `63` checks: `61/63` both-safe, `2/63` candidate-conservative, `0/63` candidate-optimistic, HELIX unsafe `0/63`; max HELIX TPOT `0.117426029 s`.
- **Combined E40+E41**: `88` targeted reasonable single-arrival phase perturbations: `85` both-safe, `3` candidate-conservative, `0` candidate-optimistic.
- **E42 paired moderate-arrival stress**: four representative candidate-safe cases (seed3 Slow, seed11 Fast, seed19 Slow, seed7-offset200 Fast). For each, identify the first-unsafe critical Prefill and the nearest other request at the candidate-safe intensity; independently perturb both arrivals by `{-40,0,+40} ms`, giving `9` combinations per case and `36` checks total. This is second-order timing noise without extreme phase shifts.
- **E42 execution**: Actions run `32715571178` completed all four matrix jobs successfully and produced four artifacts.
- **E42 result**: `35/36` both-safe, `1/36` candidate-conservative, `0/36` both-unsafe, `0/36` candidate-optimistic; HELIX unsafe in `0/36`. Maximum HELIX TPOT is `0.117426029 s < 0.150 s`.
- **E42 only mismatch**: seed11 Fast with critical Prefill jitter `-40 ms` and neighboring victim-request arrival jitter `+40 ms`. E31 creates an Arrival-triggered `N_P=1,N_D=1` overlap and charges the unchanged `0.141973573 s` blocking-service debt, yielding `0.258973573 s > 0.150 s`, while HELIX remains safe.
- **Combined E40–E42 targeted phase stress**: `124` reasonable perturbation checks around known candidate-safe points; `120` both-safe, `4` candidate-conservative, `0` candidate-optimistic. This is strong empirical safety evidence, not a proof.

## Judgment
E42 did not break unchanged E31 even when two request arrival times are perturbed simultaneously within ±40 ms. Across E40–E42, all disagreements remain one-sided in the conservative direction. The current evidence therefore supports the full blocking-service envelope as a robust scheduler-free safety screen on the tested pinned HELIX configuration, while confirming a small but real false-rejection cost.

## Next
1. Do not keep widening timing jitter; the current phase-stress evidence is already broad enough for this configuration.
2. Next smallest informative stress should change one orthogonal but still realistic dimension—e.g. a modest SLA or prompt/output-length regime—while keeping the fixed pipeline and E31 semantics unchanged.
3. Keep the primary falsification criterion `E31 safe / HELIX unsafe`; stop and diagnose immediately if one appears.
4. Keep E31 unchanged; do not merge and do not modify main.
