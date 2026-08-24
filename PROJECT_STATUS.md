# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free safety candidate. E39 is rejected; E40–E43 stress the unchanged E31 candidate rather than tightening it.

## Experiment log
- **E31/E34**: candidate-safe points were HELIX-safe in all 8 workload×pipeline checks, but E39 showed finite-trace HELIX feasibility can be nonmonotone in intensity; earlier binary-search “exact frontier” language is therefore invalid.
- **E36–E38**: seed19 Slow conservatism comes from hidden phase-dependent Prefill exposure plus trajectory mismatch. Cheap request/event state cannot determine exposure, and simply separating fixed SLA overhead from physical progression over-corrects.
- **E39 rejected**: split semantics creates candidate optimism on seed19 Slow (`0.0159` candidate-safe while HELIX max TPOT is `0.168847 s > 0.150 s`).
- **E40–E42 timing robustness**: `124` targeted reasonable timing perturbations around known candidate-safe points (single-arrival and paired-arrival jitter); `120` both-safe, `4` candidate-conservative, `0` candidate-optimistic.
- **E43 moderate-TPOT robustness design**: orthogonal SLA stress with TPOT `130 ms` and `180 ms` around the `150 ms` baseline; TTFT remains `2 s`, fixed overhead `5 ms`. Test seed7 and seed19 on Slow/Fast fixed pipelines. Candidate edge is defined as the largest contiguous safe prefix on intensity grid `0.006–0.022` with step `0.0001`, followed by first unsafe; this avoids assuming HELIX monotonicity.
- **E43 execution**: Actions run `32716674203` completed all four matrix jobs successfully and produced four artifacts.
- **E43 safety result**: `8/8` TPOT×workload×pipeline candidate-safe edge checks are HELIX-safe; `0/8` candidate optimism. HELIX is also still safe at all `8/8` candidate first-unsafe points, so every tested edge remains conservative.
- **E43 candidate edges**: seed7 stays `0.0131 safe / 0.0132 unsafe` for both Slow/Fast and for both TPOT `130/180 ms`; seed19 stays `0.0152 / 0.0153` for both pipelines and both TPOT values.
- **E43 HELIX margins at candidate-safe points**: max TPOT is `0.117426029 s` on Slow and `0.117150744 s` on Fast, hence below both `130 ms` and `180 ms` SLAs.

## Judgment
E43 strengthens safety generalization but exposes a tightness/sensitivity limitation: within a realistic TPOT range `130–180 ms`, the E31 capacity edge does not move at all on these workloads because the first active Prefill full blocking-service debt is already much larger than the additional SLA slack. Thus the scheduler-free envelope remains safe, but its capacity can be dominated by the discrete onset of any Prefill/Decode overlap rather than smoothly reflecting moderate TPOT changes. This is not a safety failure, but it is important for how the eventual capacity estimate should be interpreted.

## Next
1. Do not tune E31 to force TPOT sensitivity; that would risk the safety failures already seen in E39.
2. Next smallest orthogonal falsification should vary request size distribution moderately (e.g. prompt/output lengths ±20%) on a small representative subset while keeping arrival timing, fixed pipelines, and E31 semantics unchanged.
3. Keep the primary falsification criterion `E31 safe / HELIX unsafe`; stop and diagnose immediately if one appears.
4. Keep E31 unchanged; do not merge and do not modify main.
