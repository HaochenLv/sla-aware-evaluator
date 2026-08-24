# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#17 are research-only; do not merge.
- Best current candidate remains: exact HELIX singleton Decode profile alignment + full active-Prefill compute debt. It is a conservative screening-bound candidate, not yet a production revision.

## Experiment log
- **E7**: original Conservative capacity was far above HELIX despite the same fixed Pipeline/workload/SLA.
- **E9–E16**: HELIX diagnostics identify Prefill→Decode interference; exact singleton Decode alignment + full active-Prefill compute debt is the simplest useful conservative guard. Across four extra workloads, no candidate optimism was observed at the predicted frontier.
- **E17 / bandwidth-only ordering**: across six Azure trace offsets, increasing all seven links from 2.5 to 10 Gbps never lowered the HELIX capacity frontier. The gain was only ~0.6–1.5%, so fine bandwidth-only ranking is weak and often unresolved at 1e-4 intensity tolerance.
- **E18 / bandwidth sweep**: two representative 30 s windows (17 and 50 requests) swept all seven inter-stage links across 1.25/2.5/5/10/20 Gbps with all other inputs fixed. At 1.25 Gbps, both windows found no positive SLA-safe HELIX intensity. From 2.5 to 20 Gbps, safe request rate rose only 0.008750→0.008864 rps (~1.30%) for the 17-request window and 0.013156→0.013354 rps (~1.50%) for the 50-request window; 5 and 10 Gbps share the same sampled frontier in both cases. In contrast, aligned TTFT drops strongly with bandwidth (17-request: 1.680 s at 2.5 Gbps to 1.131 s at 20 Gbps; 50-request: 1.786 s to 1.201 s). All finite positive-capacity frontiers still fail first by TPOT when intensity increases.

## Interpretation
E18 supports a two-regime picture under the current fixed pipeline/SLA: below a network feasibility threshold, the pipeline has no positive SLA-safe capacity; once bandwidth is sufficient, request-rate capacity rapidly becomes TPOT / Prefill-interference dominated and additional bandwidth yields only ~1–1.5% frontier improvement over an 8x bandwidth increase. The observed aligned-TTFT points are close to an `a + c/B` relation, implying a rough threshold near ~1.7–1.9 Gbps for these two windows, but this is a deduction from the sampled data, not yet a validated model law. This evidence supports keeping network as a hard red-line feasibility constraint rather than recreating HELIX network/event timing solely to match percent-level fine ranking.

## Next
Do not integrate E15 yet and do not add a scheduler. Use a cheap near-threshold bandwidth probe (roughly 1.5–2.0 Gbps) to validate whether the inferred network-feasibility threshold is real and whether `TTFT(B) ≈ a + c/B` is stable. Keep the network red-line equations unchanged; fine network-only ranking remains secondary to conservative screening / robust capacity.
