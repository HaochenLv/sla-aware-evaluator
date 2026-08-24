# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Queue/interference diagnostics are research-only. Do not merge PRs #8–#11.

## Experiment log
- **E7**: Conservative and HELIX both rank Fast > Slow, but Conservative capacity is far higher.
- **E9–E11**: HELIX diagnostics isolate Prefill→Decode interference. Across 44 controlled one-/multi-Prefill cases, sum of full profiled Prefill compute upper-bounds measured Decode GPU queue; multi-Prefill cases can also inflate Decode service.
- **E12 / observational Prefill interference guard**: on the unchanged Conservative trajectory, reject states where `base Decode + sum(active Prefill full compute) + fixed > TPOT`. The guard is monotonic on the scanned grid but first becomes unsafe only at intensity 0.025 (~0.01616 rps), whereas HELIX already becomes unsafe near 0.0136 intensity. Thus the guard alone does not recover the HELIX frontier because the Conservative trajectory itself reaches Prefill/Decode overlap too late.
- **New root-cause clue**: at E12's overlap, Conservative uses base Decode compute 0.056 s. HELIX isolated Decode layer service measured in E9–E11 is ~0.11206 s. Pinned HELIX `ModelLayer.get_inference_statistics` explicitly multiplies Decode inference time by 2 when the Decode batch contains exactly one token. `HelixA100Llama2Profiler.decode_time_per_token` currently sums the raw profile points and does not apply this runtime singleton-Decode rule. This is a profile-semantics mismatch, not a queue-model effect.

## Next
Before integrating any queue guard, test a runtime-aligned conservative Decode profile that applies the HELIX singleton-Decode ×2 rule. Re-run the 30 s capacity/overlap comparison, then reassess how much Prefill-interference debt is still needed. Keep the SLA-to-network red-line equations unchanged.
