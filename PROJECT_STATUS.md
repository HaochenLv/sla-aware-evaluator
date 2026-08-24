# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#20 are research-only; do not merge.
- Best compute-side candidate remains exact HELIX singleton Decode profile alignment + full active-Prefill compute debt. Network red-line equations remain unchanged.

## Experiment log
- **E15–E16 / Prefill guard**: exact singleton Decode + active-Prefill debt stayed conservative across tested workloads; no candidate optimism was observed.
- **E17–E18 / bandwidth diagnostics**: once the pipeline is network-feasible, 2.5→20 Gbps changes HELIX safe request-rate capacity by only ~1.3–1.5%; below the feasibility threshold TTFT dominates, above it TPOT/Prefill interference dominates.
- **E19 / isolated feasibility**: one-request probes gave aligned-TTFT thresholds ~1.655 Gbps for 1709 prompt tokens and ~1.894 Gbps for 1821 tokens; below-threshold failure is TTFT-only.
- **E20 / red-line validation**: HELIX bisection brackets are `(0.282813, 0.286621]`, `(0.728418, 0.732227]`, `(1.391113, 1.394922]`, `(1.893848, 1.897656]` Gbps for 512/1024/1536/1821 prompt tokens. Current evaluator predictions `0.281099/0.697328/1.277129/1.688402` Gbps lie below all four HELIX brackets, with optimism growing with prompt length. HELIX source includes an explicit first-local-layer CPU-buffer concat/transfer cost on every compute node, `D/4GBps + D/5GBps`. Accounting for only this known runtime overhead and HELIX's activation+token network payload, while leaving the red-line algebra unchanged, predicts `0.286307/0.730144/1.391531/1.894205` Gbps; all four lie inside the HELIX bisection brackets.

## Interpretation
For an isolated request, define `S = sum_e D_e/B_e`. The evaluator allocation `w_e=(D_e/B_e)/S`, `delta_e=w_e*Delta`, `b_req=D_e/delta_e` gives `b_req_e <= B_e` iff `S <= Delta`. Thus, under the fixed-path/no-contention assumptions, the current network red line is algebraically equivalent to requiring total serial network time to fit in the remaining SLA budget. E20 shows the equal-link HELIX mismatch came from upstream runtime-overhead accounting, not the red-line structure. The HELIX-specific 4/5-GBps constants are simulator/runtime details and must not be hard-coded as a universal evaluator rule; a generic evaluator should obtain such overhead from profiling or a conservative overhead bound. Results are simulator-relative and use aligned Prefill-completion TTFT, not standard true first-token TTFT.

## Next
Keep the network equation unchanged. Validate the same weighted red-line equivalence on an isolated **heterogeneous-link** pipeline, where the seven links have different capacities, before considering any evaluator revision. Separately, later decide how runtime overhead should enter the generic profiling/budget interface.