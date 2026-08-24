# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- E21 heterogeneous-link HELIX validation is prepared on draft PR #23 but remains runner-blocked: a fresh Actions rerun on 2026-08-24 again failed before any workflow step started.
- Network red-line equations remain unchanged. The best compute-side candidate remains exact HELIX singleton Decode profile alignment + full active-Prefill compute debt.

## Experiment log
- **E15–E16 / Prefill guard**: exact singleton Decode + active-Prefill debt stayed conservative across tested workloads; no candidate optimism was observed.
- **E17–E18 / bandwidth diagnostics**: once network-feasible, 2.5→20 Gbps changes HELIX safe request-rate capacity by only ~1.3–1.5%; below threshold TTFT dominates, above it TPOT/Prefill interference dominates.
- **E19–E20 / isolated red-line validation**: for 512/1024/1536/1821 prompts, the original evaluator underpredicts HELIX's required bandwidth increasingly with prompt length. Keeping the red-line algebra unchanged but accounting for HELIX's known runtime overhead and activation+token payload puts all four predictions inside the direct HELIX bisection brackets.
- **E21 / heterogeneous-link validation**: two unequal-link patterns and prompts 1024/1709 are prepared; direct HELIX execution is still blocked by GitHub Actions runner failure, so no E21 runtime result is claimed yet.
- **E22 / generic Prefill overhead calibration**: using only the four E20 measured HELIX aligned-TTFT threshold brackets, the inferred linear law `T_ovhd^P(L)=h_P L` gives per-prompt admissible intervals (us/token): 512 `[19.375, 62.489]`, 1024 `[56.004, 62.556]`, 1536 `[58.784, 60.585]`, 1821 `[58.891, 59.863]`. Their joint intersection is `[58.891, 59.863] us/token`; midpoint `h_P=59.377 us/token`. The HELIX source CPU-buffer coefficient `58.9824 us/token` lies inside this interval and was not used for fitting.
- **E22 predictions**: the joint midpoint predicts thresholds `0.286343/0.730374/1.392365/1.895750` Gbps for prompts 512/1024/1536/1821, all 4/4 inside the E20 HELIX brackets. Leave-one-prompt-out predicts `0.286343/0.730374/1.392365/1.896956` Gbps respectively, again 4/4 inside the held-out brackets.

## Interpretation
For isolated fixed-path Prefill, the network red-line itself continues to hold up: the observed mismatch is explained by time-budget inputs, especially runtime overhead. E22 supports a generic profiling interface rather than HELIX-specific constants: expose Prefill runtime overhead as a profiled or conservatively bounded function of request size. This is simulator-relative evidence under aligned Prefill-completion TTFT and one model/hardware/runtime, so it is not a universal linear-overhead law.

## Next
Do not revise the network equation and do not add a scheduler. Keep E21 pending until a runner is available. After the network side is externally closed, return to the main request-rate issue: test whether `sum(active Prefill full compute)` also upper-bounds total Prefill-induced Decode compute-side excess (queue wait + service-time inflation), not only queue wait.