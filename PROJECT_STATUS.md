# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- E21 heterogeneous-link HELIX validation is prepared on draft PR #23 but is runner-blocked: even a one-line Actions smoke job still fails before any step starts.
- Network red-line equations remain unchanged. The best compute-side candidate remains exact HELIX singleton Decode profile alignment + full active-Prefill compute debt.

## Experiment log
- **E15–E16 / Prefill guard**: exact singleton Decode + active-Prefill debt stayed conservative across tested workloads; no candidate optimism was observed.
- **E17–E18 / bandwidth diagnostics**: once network-feasible, 2.5→20 Gbps changes HELIX safe request-rate capacity by only ~1.3–1.5%; below threshold TTFT dominates, above it TPOT/Prefill interference dominates.
- **E19–E20 / isolated red-line validation**: for 512/1024/1536/1821 prompts, the original evaluator underpredicts HELIX's required bandwidth increasingly with prompt length. Keeping the red-line algebra unchanged but accounting for HELIX's known runtime overhead and activation+token payload puts all four predictions inside the direct HELIX bisection brackets.
- **E21 / heterogeneous-link validation**: two unequal-link patterns and prompts 1024/1709 are prepared; direct HELIX execution is still blocked by GitHub Actions runner failure, so no E21 runtime result is claimed yet.
- **E22 / overhead calibration from E20 thresholds**: infer a generic Prefill overhead law `T_ovhd^P(L)=h_P L` directly from the four measured HELIX threshold brackets, without using the HELIX source constants as fit inputs. The four constraints have a non-empty joint interval `h_P in [58.891, 59.863] us/token`; the HELIX source CPU-buffer coefficient is `58.9824 us/token` and lies inside it. Using the joint midpoint `59.377 us/token` predicts all four bandwidth thresholds inside their observed HELIX brackets. Leave-one-prompt-out calibration also predicts the held-out threshold inside its HELIX bracket for all 4/4 prompts.

## Interpretation
For isolated fixed-path Prefill, the network red-line itself continues to hold up: the observed mismatch is explained by time-budget inputs, especially runtime overhead. E22 strengthens the generic modeling direction: do not hard-code HELIX's 4/5-GBps constants; instead expose Prefill runtime overhead as a profiled or conservatively bounded function of request size. This is still simulator-relative evidence under aligned Prefill-completion TTFT and one model/hardware/runtime, so it is not yet a universal law.

## Next
Do not revise the network equation and do not add a scheduler. Keep E21 pending until a runner is available. Meanwhile test whether the simple profiled-overhead interface remains valid when the overhead is estimated from one subset of prompt lengths and used on unseen lengths / different pipeline stage counts; after that, return to the main unresolved request-rate issue: a simple conservative Prefill/Decode queue-interference bound.