# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#20 plus E21 draft PR #23 are research-only; do not merge.
- Best compute-side candidate remains exact HELIX singleton Decode profile alignment + full active-Prefill compute debt. Network red-line equations remain unchanged.

## Experiment log
- **E15–E16 / Prefill guard**: exact singleton Decode + active-Prefill debt stayed conservative across tested workloads; no candidate optimism was observed.
- **E17–E18 / bandwidth diagnostics**: once network-feasible, 2.5→20 Gbps changes HELIX safe request-rate capacity by only ~1.3–1.5%; below threshold TTFT dominates, above it TPOT/Prefill interference dominates.
- **E19–E20 / isolated red-line validation**: for equal links, the existing normalized-cost red-line is equivalent to fitting total serial network time inside the remaining SLA budget. After accounting for HELIX's known CPU-buffer overhead and activation+token payload, the same red-line predicts all four HELIX aligned-TTFT thresholds inside their bisection brackets.
- **E21 / heterogeneous-link setup**: two seven-link ratio patterns (`0.7/1.0/1.4/0.8/1.8/1.1/2.2` and `0.5/2/2/2/2/2/2`) with prompts 1024 and 1709 are prepared. Current-evaluator predicted scale thresholds are `0.628796/0.498092` Gbps for 1024 and `1.348998/1.068590` Gbps for 1709 (mild/one-bottleneck). HELIX-source-aware same-red-line predictions are `0.658387/0.521532` and `1.492723/1.182439` Gbps respectively. Direct HELIX execution is not yet available: current GitHub Actions jobs fail before any runner step starts, and even a one-line `echo` runner smoke test fails with no steps/log blob, so this is an execution-infrastructure blocker rather than an E21-code result.

## Interpretation
The heterogeneous red-line equivalence is already an algebraic deduction: with `S=sum_e D_e/B_e`, `w_e=(D_e/B_e)/S`, `delta_e=w_e*Delta`, and `b_req=D_e/delta_e`, each link passes iff `S<=Delta`. E21 is intended only as external HELIX validation that pinned runtime semantics preserve that relation under unequal link capacities. Do not count the source-aware E21 numbers as experimental evidence until the HELIX jobs actually execute.

## Next
When a runner is available, execute the four E21 isolated HELIX cases and check whether the source-aware scale lies inside each direct HELIX threshold bracket. Keep the network equation unchanged and do not add a scheduler.