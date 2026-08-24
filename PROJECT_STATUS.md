# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free candidate, but E44 produced a genuine candidate-optimism counterexample under a moderate request-size shift. E45 now localizes that failure to an omitted intrinsic Prefill service overhead in the Evaluator's own TTFT accounting, not cross-request queueing and not the network red-line.

## Experiment log
- **E31/E34**: candidate-safe points were HELIX-safe in all 8 workload×pipeline checks, but E39 showed finite-trace HELIX feasibility can be nonmonotone in intensity; earlier binary-search “exact frontier” language is therefore invalid.
- **E36–E38**: seed19 Slow conservatism comes from hidden phase-dependent Prefill exposure plus trajectory mismatch. Cheap request/event state cannot determine exposure, and simply separating fixed SLA overhead from physical progression over-corrects.
- **E39 rejected**: split semantics creates candidate optimism on seed19 Slow (`0.0159` candidate-safe while HELIX max TPOT is `0.168847 s > 0.150 s`).
- **E40–E42 timing robustness**: `124` targeted reasonable timing perturbations around known candidate-safe points; `120` both-safe, `4` candidate-conservative, `0` candidate-optimistic.
- **E43 moderate-TPOT robustness**: TPOT `130 ms` and `180 ms`, seed7/seed19 × Slow/Fast. `8/8` candidate-safe checks are HELIX-safe and `0/8` candidate optimism. The candidate edge does not move across `130–180 ms`, exposing a sensitivity/tightness limitation.
- **E44 moderate request-size robustness**: multiply prompt/output lengths by `0.8×` or `1.2×`, keep arrival times, TTFT=`2 s`, TPOT=`150 ms`, fixed overhead=`5 ms`, fixed pipeline, E31 semantics, and network red-line unchanged. E44 falsifies unchanged E31 on seed7 Slow at `1.2×`, intensity `0.0109`: E31 says safe, pinned HELIX gives aligned TTFT `2.015105077 s > 2.0 s` on `azure-00008`; TPOT remains safe.
- **E45 focused diagnosis**: Actions run `32720229427` completed successfully. Target `azure-00008` has input length `2051` tokens, output `143`, and arrives with Evaluator state `N_P=1, N_D=0`.
- **E45 full-vs-isolated test**: HELIX full-workload raw Prefill time is `2.010105077008 s`; isolated replay is `2.010105077004 s`. Difference is only `3.9e-12 s`. Compute+queue and network differences are also effectively zero. Therefore this counterexample is **not caused by cross-request queueing/interference**.
- **E45 component decomposition**: Evaluator profile compute=`1.136320 s`; ideal internal-network service=`0.752720282 s`; fixed=`0.005 s`, leaving `0.105959718 s` TTFT margin. HELIX isolated compute+queue=`1.257292902 s`, i.e. `0.120972902 s` above profile compute. HELIX network exceeds ideal-network algebra by only `0.000091893 s` (`0.092 ms`). Thus almost the entire optimism comes from omitted intrinsic Prefill compute-node service overhead, not network mismatch.
- **E45 stage localization**: on each of the 8 ten-layer stages, profile pure compute is `0.142040 s`, while isolated HELIX compute-node residence is `0.157161613 s`, an extra `0.015121613 s` per stage. Summed over 8 stages this is `0.120972902 s`. No overlapping requests are observed on any target stage.
- **E45 link to E22**: independently calibrated E22 Prefill blocking-overhead predicts `0.121782655 s` for this 2051-token request, only about `0.000810 s` above the HELIX isolated intrinsic overhead. The exact pinned-HELIX CPU-buffer coefficient previously identified (`58.9824 us/token` across 8 stages) yields the observed `~0.120973 s` essentially exactly. E22 was already used when charging Prefill blocking debt to Decode, but unchanged E31 did not charge the same Prefill intrinsic overhead to the Prefill request's own TTFT budget.

## Judgment
E45 explains the E44 counterexample cleanly. The failure is not evidence that the network red-line or the overall conservative-envelope idea is broken. Instead, E31 is asymmetric: it recognizes Prefill service overhead when estimating how Prefill can block Decode, but omits that overhead when deciding whether Prefill itself fits inside TTFT. At 2051 prompt tokens the omitted term is about `121 ms`; the Evaluator had only about `106 ms` residual TTFT margin, producing the observed `15.1 ms` optimism. This is a specific modeling omission with a plausible existing profiled/bounded term, not a reason to add a global safety factor.

## Next
1. Build a research-only candidate that charges a profiled/bounded Prefill intrinsic service-overhead term in the Prefill TTFT budget, reusing the E22-style provenance rather than hard-coding HELIX CPU-buffer constants as universal theory.
2. First test only the E44 counterexample: the fix must make seed7 Slow `1.2×`, intensity `0.0109` unsafe/conservative without changing the network red-line.
3. Then rerun the earlier candidate-safe points to check that the TTFT fix removes optimism without excessive new false rejection.
4. Keep unchanged E31 as baseline/provenance; no merge and no main modification.
