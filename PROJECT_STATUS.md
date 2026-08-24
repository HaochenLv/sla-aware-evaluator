# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#20 are research-only; do not merge.
- Best compute-side candidate remains exact HELIX singleton Decode profile alignment + full active-Prefill compute debt. Network red-line equations remain unchanged.

## Experiment log
- **E15–E16 / Prefill guard**: exact singleton Decode + active-Prefill debt stayed conservative across all tested workload variants; no candidate optimism was observed, but compute-driven progress ties bandwidth-only Slow/Fast pipelines.
- **E17–E18 / bandwidth diagnostics**: higher bandwidth weakly raises HELIX capacity, but 2.5→20 Gbps changes safe request-rate capacity by only ~1.3–1.5% once the pipeline is feasible. Below a threshold, TTFT makes the pipeline infeasible; above it, TPOT/Prefill interference dominates capacity.
- **E19 / isolated feasibility**: one-request probes give aligned-TTFT thresholds ~1.655 Gbps for 1709 prompt tokens and ~1.894 Gbps for 1821 tokens; TPOT remains ~117.5 ms, so the below-threshold failure is TTFT-only.
- **E20 / network-red-line source audit**: the current red-line prediction alone is ~1.496 Gbps for 1709 tokens and ~1.688 Gbps for 1821 tokens, about 9.6% and 10.9% below HELIX. HELIX source adds a first-local-layer CPU-buffer concat/transfer overhead on every compute node, `D/4Gbps + D/5Gbps`. Adding only this known runtime overhead to the available TTFT budget, while leaving the network red-line algebra unchanged, predicts ~1.655414506 and ~1.894205021 Gbps—numerically matching the HELIX fitted thresholds for the two observed prompts.

## Interpretation
For isolated Prefill under the current equal-link fixed pipeline, the observed HELIX mismatch is explained by upstream time-budget/overhead accounting rather than by the network red-line structure itself. This is evidence for keeping `SLA -> remaining time -> network red line`, while requiring compute/fixed overhead inputs to be conservative or profiled. The HELIX-specific 4/5-Gbps CPU-buffer constants must not be hard-coded as a universal evaluator rule. This result is simulator-relative and uses evaluator-aligned Prefill-completion TTFT, not standard true first-token TTFT.

## Next
Finish E20 HELIX bisection thresholds for 512/1024/1536/1821-token isolated prompts and compare them against both (a) the current evaluator prediction and (b) the source-aware same-red-line prediction. Do not change the red-line equation or add a scheduler.