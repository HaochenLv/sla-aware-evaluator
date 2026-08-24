# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- E27 is a research-only out-of-configuration profiling transfer test. Network red-line equations, serving scheduler, and Evaluator state-machine semantics remain unchanged.
- Public Actions run `32706628352` completed successfully and produced artifact `9512491598` for the unchanged nine-case E27 experiment.

## Experiment log
- **E23 correction**: profiler-only Prefill compute covers 42/44 controlled E10/E11 cases; the two 1024-token E10 cases at offset 0.08 exceed compute-only debt by up to `4.2621%`.
- **E26**: the independently calibrated E22 overhead midpoint `59.377209 us/token` on the 8-stage pipeline reconstructs independent E10 measured Prefill compute-node service within `0.05710%` and restores 44/44 interference-debt coverage.
- **E27 setup / stage-count transfer**: keep LLaMA-2-70B, A100-40GB, prompt profile, HELIX runtime, direct stage links, and 80 total layers fixed; vary only the continuous layer partition from 8 stages to 7/8/10 stages (`12+12+12+11+11+11+11`, `8x10`, `10x8`). Test prompts 256/512/1024. Use the 8-stage E22 coefficient through the per-stage form `T_ovhd^P = (h_8/8) * N_stage * L`, with `h_8/8 = 7.422151 us/(token·stage)`.
- **E27 result — 7 stages (held out)**: relative prediction error for measured Prefill compute-node service is `0.04995%`, `0.05050%`, `0.05050%` for prompts 256/512/1024.
- **E27 result — 8 stages (control)**: relative error is `0.05648%`, `0.05710%`, `0.05710%`.
- **E27 result — 10 stages (held out)**: relative error is `0.06915%`, `0.06988%`, `0.06988%`.
- **E27 summary**: all 9/9 cases closely match the stage-scaled prediction; maximum absolute relative error across all cases and across held-out 7/10-stage cases is `0.0698798%`.

## Interpretation
E27 provides direct out-of-configuration evidence, within the pinned HELIX runtime and the tested 7/8/10-stage LLaMA-2-70B A100 setup, that the E22-calibrated Prefill blocking overhead behaves almost exactly as a stage-additive term. This supports a profiling interface of the form `T_P^blocking = T_P^compute + sum_stage T_P,stage^blocking-overhead` rather than one opaque fixed 8-stage pipeline constant. It still does not justify a universal coefficient across different runtimes, GPU types, or model families; the coefficient remains a profiled/bounded input.

## Next
1. Refine the E31 HELIX frontier to quantify conservatism on the seed-7 workload.
2. Repeat E31 on multiple workload seeds/windows.
3. Keep stage-local blocking overhead as the leading profiling interface candidate, but do not merge or change default Evaluator semantics yet.
