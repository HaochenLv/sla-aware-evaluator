# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free candidate, but E44 has now produced a genuine candidate-optimism counterexample under a moderate request-size shift. Do not claim cross-regime conservative safety for unchanged E31.

## Experiment log
- **E31/E34**: candidate-safe points were HELIX-safe in all 8 workload×pipeline checks, but E39 showed finite-trace HELIX feasibility can be nonmonotone in intensity; earlier binary-search “exact frontier” language is therefore invalid.
- **E36–E38**: seed19 Slow conservatism comes from hidden phase-dependent Prefill exposure plus trajectory mismatch. Cheap request/event state cannot determine exposure, and simply separating fixed SLA overhead from physical progression over-corrects.
- **E39 rejected**: split semantics creates candidate optimism on seed19 Slow (`0.0159` candidate-safe while HELIX max TPOT is `0.168847 s > 0.150 s`).
- **E40–E42 timing robustness**: `124` targeted reasonable timing perturbations around known candidate-safe points (single-arrival and paired-arrival jitter); `120` both-safe, `4` candidate-conservative, `0` candidate-optimistic.
- **E43 moderate-TPOT robustness**: TPOT `130 ms` and `180 ms`, seed7/seed19 × Slow/Fast. `8/8` candidate-safe checks are HELIX-safe and `0/8` candidate optimism. The candidate edge does not move across `130–180 ms`, exposing a sensitivity/tightness limitation.
- **E44 moderate request-size robustness design**: multiply every request's prompt and output lengths together by `0.8×` or `1.2×`, keep arrival times, TTFT=`2 s`, TPOT=`150 ms`, fixed overhead=`5 ms`, fixed pipeline, E31 semantics, and network red-line unchanged. Test seed7/seed19 × Slow/Fast.
- **E44 rerun**: Actions run `32718676248` used an expanded candidate scan `[0.001, 0.040]` with step `0.0001`. seed7 Fast, seed7 Slow, and seed19 Fast completed successfully; seed19 Slow still had no contiguous E31 safe→unsafe prefix edge within the expanded range.
- **E44 falsification**: seed7 Slow at `1.2×` request size is a genuine candidate-optimism case. E31 marks intensity `0.0109` safe, but pinned HELIX is unsafe due to TTFT on request `azure-00008`: max aligned TTFT `2.015105077 s > 2.0 s`. Max TPOT remains `0.117426029 s < 0.150 s`, so this is specifically a Prefill/TTFT-side failure rather than the known Decode TPOT blocking issue.
- **E44 other completed cases**: seed7 Fast (`0.8×`: `0.0164/0.0165`; `1.2×`: `0.0109/0.0110`) and seed19 Fast (`0.8×`: `0.0190/0.0191`; `1.2×`: `0.0127/0.0128`) are conservative at the candidate prefix edge. seed7 Slow `0.8×` (`0.0164/0.0165`) is also conservative.

## Judgment
E44 breaks the unchanged E31 safety story across moderate request-size regimes. The counterexample is small in numerical margin (~`15.1 ms` TTFT overshoot) but conceptually important because the Evaluator says safe while the execution reference violates the hard SLA. The failure is on TTFT/Prefill semantics, not TPOT. Therefore the next step is not broader robustness testing and not more seed19 Slow boundary hunting; it is a focused diagnosis of why E31 underestimates TTFT cost when prompt/output sizes are scaled up by 20% on the Slow pipeline.

## Next
1. Stop broad E31 robustness expansion for now; the primary falsification criterion has been met.
2. Diagnose the seed7 Slow `1.2×`, intensity `0.0109`, request `azure-00008` case: decompose Evaluator TTFT budget versus HELIX Prefill completion/queue/network/runtime components and identify the missing or underestimated term.
3. Do not tune a global safety factor. Any fix must target the identified TTFT mechanism and preserve the existing network red-line structure.
4. Keep E31 unchanged until the mechanism is understood; do not merge and do not modify main.
