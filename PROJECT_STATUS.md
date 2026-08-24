# Project Status

## Current state
- Conservative Evaluator remains unchanged; explicit queue overhead is still zero in the current production model.
- HELIX fixed-pipeline Reference stays on draft PR #6 and is a relative execution reference, not ground truth.
- The current compute-side candidate is a deterministic **Prefill interference debt**, not merely a queue-time term. Network red-line equations remain unchanged.

## Experiment log
- **E7–E9**: the large Conservative/HELIX capacity gap was localized to runtime compute contention: a later Prefill can block active Decode and create large TPOT delay.
- **E10 / one-Prefill sweep**: 32 controlled Slow/Fast cases. Full blocker Prefill compute upper-bounds total observed Decode compute-side excess in all 32 because Decode service inflation was zero; worst queue/debt ratio was ~0.954.
- **E11 / multi-Prefill additivity**: 12 cases with 2–3 Prefills. Sum of full profiled Prefill compute upper-bounds Decode queue wait in all 12. Heavy 3x512 overlap also inflates Decode layer service from ~0.11206 s to ~0.15632 s, showing queue wait alone is incomplete.
- **E23 / total Decode compute-side excess re-analysis**: re-analyzed the exact E11 measured observations using `E_D = T_queue + max(0, T_service - T_service,isolated)`. The full active-Prefill debt upper-bounds this total excess in all 12/12 multi-Prefill cases. Worst excess/debt ratio is 0.678904 (Slow, 512+1024). For the two material service-inflation cases (3x512), total excess/debt ratios are 0.371766 Slow and 0.354910 Fast. Combined with E10, there are now 44/44 controlled HELIX cases with no observed violation of the full-Prefill-debt bound.

## Candidate model (not yet adopted)
For active Decode at event time `t`, define
`I_D^UB(t) = sum_{p in active Prefill} T_prefill,p^UB`.
Use it as a **total compute-side interference debt**:
`Delta_D = tau_D - T_decode,base^UB - I_D^UB(t) - T_fix`,
then leave the existing SLA-to-network red-line unchanged. This deliberately over-counts already-executed Prefill work and is intended as a cheap conservative bound rather than a scheduler reconstruction.

## Interpretation
The distinction matters: the empirical support is now stronger for `Prefill interference debt` than for a literal `queue-time estimator`, because the same debt covers both measured GPU queue delay and the observed Decode service inflation under heavier overlap. This is still HELIX-relative empirical evidence, not a mathematical proof or a universal runtime law. The one-Prefill worst ratio near 0.95 also argues against introducing an aggressive multiplicative discount factor below 1 if conservatism is required.

## Next
Do not integrate the candidate into production Conservative yet and do not add a scheduler. The next controlled step is to test the candidate as an Evaluator-side ablation together with the already-established exact singleton Decode profiling and compare request-rate frontiers against HELIX across the existing workload variants. Separately keep E21 heterogeneous-network validation pending until GitHub Actions runners recover.