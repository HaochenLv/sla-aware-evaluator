# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- Current magnitude candidate remains **Prefill blocking-service debt = profiled Prefill compute + profiled/bounded blocking runtime overhead**. Network red-line equations remain unchanged.
- E27 and E31 direct HELIX runs remain infrastructure-blocked before workflow steps.

## Experiment log
- **E23 correction / E26**: profiler-only Prefill compute covers 42/44 archived E10/E11 interference cases. Adding independently calibrated E22 blocking/runtime overhead gives 44/44 coverage; tightest service-debt ratio is `0.953143`. E22 predicts independent E10 measured Prefill compute-node service within `0.05710%` maximum relative error.
- **E25 / frontier cliff**: full immediate debt creates a coarse first-overlap unsafe cliff because singleton Decode + fixed overhead leaves only `33 ms` of the 150-ms TPOT budget.
- **E28–E30**: hidden execution phase matters; a large global multiplier discount is unsupported, and the largest archived-compatible constant slack (`32.995 ms`) still removes none of the E25 cliffs.
- **E32 / prompt-dependent envelope training**: train an intentionally optimistic scheduler-free exposure factor on E10 one-Prefill cases, `alpha(L)=max_offset,pipeline observed_excess / T_P^blocking(L)`. Anchors are `alpha(64)=0`, `alpha(256)=0.793444`, `alpha(512)=0.906857`, `alpha(1024)=0.953143`.
- **E32 / held-out multi-Prefill test**: apply the additive prompt envelope `sum alpha(L_i) T_P^blocking(L_i)` to all 12 E11 multi-Prefill cases, which were not used to fit the anchors. Result: 0/12 violations; worst observed excess/candidate ratio is `0.661865`.
- **E32 / E25 stress**: invert the pinned HELIX prompt profile to recover the historical first-unsafe Prefill prompt lengths from their compute debts: `0.11584 s -> 181 tokens`, `0.12992 s -> 203`, `0.66336 s -> 1073`, `0.68192 s -> 1131`. Use piecewise-linear interpolation between E10 alpha anchors and conservatively set `alpha=1` above 1024. Even this fitted prompt-dependent rule leaves all 10 E25 pipeline×workload first-unsafe cases above the `33 ms` residual. The most reduced case is the 181-token Prefill: `61.206 ms`, still `1.855x` the residual; 203 tokens gives `81.553 ms` (`2.471x`).

## Interpretation
E32 is another negative-but-informative result. A more flexible prompt-length-dependent exposure rule can be made to fit E10 and transfers safely to the held-out E11 multi-Prefill cases, but it still does not solve the actual finite-trace cliff that motivated tightening. Adopting such a fitted alpha curve would therefore add empirical parameters and complexity without changing the E25 qualitative capacity behavior. This strengthens the case for keeping full blocking-service debt as the clean scheduler-free worst-case envelope rather than tuning magnitude to imitate hidden execution phase.

## Next
Do not integrate the E32 alpha curve; it is diagnostic and fitted. Keep the implementation target as E31: blocking-service debt inside the original Decode SLA budget, with unchanged event trajectory and network red line. When runners recover, first execute E27 stage-count transfer and E31 frontier/HELIX probes. Until then, further offline work should focus only on falsifying the current envelope, not inventing more fitted discounts.
