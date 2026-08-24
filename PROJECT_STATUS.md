# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- Current magnitude candidate is **Prefill blocking-service debt = profiled Prefill compute + profiled/bounded blocking runtime overhead**. Network red-line equations remain unchanged.
- E27 stage-count transfer remains infrastructure-blocked: Actions still fail before any workflow step, so there is no out-of-configuration HELIX result yet.

## Experiment log
- **E23 correction / E26**: profiler-only Prefill compute covers 42/44 archived E10/E11 interference cases. Adding independently calibrated E22 blocking/runtime overhead gives 44/44 coverage; tightest service-debt ratio is `0.953143`. E22 also predicts independent E10 measured Prefill compute-node service within `0.05710%` maximum relative error.
- **E25 / frontier cliff**: charging the full historical debt immediately at the first `N_P=1, N_D=1` overlap creates a coarse finite-trace unsafe cliff. Magnitude and exposure timing are separate questions.
- **E28 / timing-phase sensitivity audit**: re-analyzed the successful E10 one-Decode/one-Prefill timing sweep at offsets `{0.01,0.04,0.08,0.12}` with fixed Prefill service magnitude for each prompt. For all 6 nonzero pipeline×prompt groups (Fast/Slow × 256/512/1024), Decode queue wait is neither monotone increasing nor monotone decreasing with the offset. All 6 peak at `0.08 s`, then at `0.12 s` return to the same queue wait as `0.01 s` within floating-point precision (maximum absolute difference `<4e-15 s`). The `0.12-0.01 = 0.11 s` separation is close to one isolated Decode token interval: `0.112151 s` Fast and `0.112426 s` Slow (about 1.92% and 2.16% difference). The 64-token Prefill produces zero measured Decode queue wait at all four offsets on both pipelines despite nonzero Prefill service (`0.044735 s`).

## Interpretation
E28 is direct evidence that **actual Prefill exposure to Decode is stage/execution-phase sensitive**, not a simple monotone function of Prefill size, time since overlap, or a naive remaining-service clock. For a fixed prompt, the full Prefill service magnitude is unchanged while the observed queue exposure changes with relative timing and nearly resets after one Decode-token-scale shift. This does not prove an exact universal period; it shows that a one-dimensional monotone timing heuristic is not justified by E10. Therefore do not replace the coarse full-debt rule with an unvalidated `remaining debt` or fixed-delay gate just to improve the frontier. If exact phase timing requires reconstructing HELIX scheduling, stop: the Evaluator should remain a cheap conservative screening model.

## Next
Keep E27 pending until runners recover. The next decisive model experiment should compare two choices explicitly: (A) full blocking-service debt as a scheduler-free worst-case envelope, versus (B) one minimal event-level exposure rule that uses only information already present in the Evaluator state, not stage scheduling. Any rule B must first survive the archived E10/E11 bounds and then be tested against HELIX across workload variants before adoption. Do not fit a global multiplier.
