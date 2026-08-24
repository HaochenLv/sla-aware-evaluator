# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- E27 is a research-only out-of-configuration profiling transfer test. Network red-line equations, serving scheduler, and Evaluator state-machine semantics remain unchanged.
- Direct E27 execution is currently blocked by the same GitHub Actions runner failure affecting E21/E24.

## Experiment log
- **E23 correction**: profiler-only Prefill compute covers 42/44 controlled E10/E11 cases; the two 1024-token E10 cases at offset 0.08 exceed compute-only debt by up to `4.2621%`.
- **E26**: the independently calibrated E22 overhead midpoint `59.377209 us/token` on the 8-stage pipeline reconstructs independent E10 measured Prefill compute-node service within `0.05710%` and restores 44/44 interference-debt coverage.
- **E27 setup / stage-count transfer**: keep LLaMA-2-70B, A100-40GB, prompt profile, HELIX runtime, direct stage links, and 80 total layers fixed; vary only the continuous layer partition from 8 stages to 7/8/10 stages (`12+12+12+11+11+11+11`, `8x10`, `10x8`). Test prompts 256/512/1024. Use the 8-stage E22 coefficient only through a per-stage form: `T_ovhd^P = (h_8/8) * N_stage * L`. The 7- and 10-stage cases are held-out stage-count tests; 8-stage is the in-configuration control.
- **E27 execution status**: Actions run `32701638835` failed before any workflow step started; job `97354208569` has `steps=null` and no log URL. Therefore E27 has no HELIX runtime result yet. Do not count per-stage scaling as validated evidence.

## Interpretation
E27 is designed to test whether the generic overhead should be exposed as a stage-local profiling term rather than one fixed pipeline constant. If the per-stage scaling transfers, the Evaluator interface can use a structural/profile input such as `T_P^service = T_P^compute + sum_stage T_P,stage^overhead` without encoding HELIX's 4/5-GBps source constants. If it does not transfer, keep overhead as a pipeline-level profiled/bounded function and do not force stage additivity. The source algebra alone is not sufficient evidence.

## Next
Keep E27 pending and rerun the nine direct HELIX cases when Actions runners recover. Until then, the strongest completed evidence is E26's in-configuration cross-experiment transfer; do not extrapolate it to stage counts 7/10.
