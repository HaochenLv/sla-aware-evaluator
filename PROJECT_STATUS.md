# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#15 are research-only; do not merge.
- Best current candidate remains: exact HELIX singleton Decode profile alignment + full active-Prefill compute debt. It is a screening-bound candidate, not yet a production revision.

## Experiment log
- **E7**: original Conservative capacity was far above HELIX despite the same fixed Pipeline/workload/SLA.
- **E9–E11**: HELIX diagnostics establish real Prefill→Decode queue/interference. One-/multi-Prefill controlled tests support full active Prefill compute as a simple conservative interference-debt candidate.
- **E12–E14**: full-Prefill debt alone rejects too late; unconditional 2× Decode is an artifact; the exact HELIX singleton Decode rule is semantically necessary but insufficient by itself.
- **E15 / exact singleton + active-Prefill debt**: original seed-7 workload gives candidate safe 0.0131 / unsafe 0.0132; HELIX safe frontiers are Slow 0.0134 and Fast 0.0136. Candidate is slightly conservative but ties Slow/Fast.
- **E16 / four additional workload variants**: three length-seed variants plus one different arrival window (41 requests) all keep the candidate conservative at its predicted frontier. Across 8 pipeline×workload checks, HELIX is feasible at every candidate-unsafe edge (8/8), so no candidate optimism was observed. At 5% above the candidate-unsafe edge, HELIX is already TPOT-unsafe in 6/8 checks; the two exceptions are seed11-Fast and seed19-Slow. Candidate safe/unsafe request-rate brackets are about 0.00983–0.00989 rps (seed3), 0.01002–0.01008 (seed11), 0.00983–0.00989 (seed19), and 0.01204–0.01218 (offset200).

## Interpretation
E16 strengthens the case that the simple Prefill-debt formulation is useful as a conservative screening bound: it stayed on the safe side of HELIX across all tested variants and was usually within a 5% intensity increase of a HELIX TPOT failure. However, it ties Slow/Fast in every tested workload because current Conservative progress is compute-driven and bandwidth affects red-line feasibility but not phase timing. HELIX can separate the pipelines through bandwidth-dependent timing; seed19 even has Slow safe and Fast unsafe at the same +5% probe, showing that strict max-TPOT ranking on a finite trace can be phase-sensitive rather than monotonically ordered by link speed. Some HELIX runs are aligned-TTFT feasible while true first-token TTFT exceeds 2 s (e.g. seed11/seed19 Slow), so external TTFT semantics remain a separate issue.

## Next
Do not integrate E15 yet and do not add a scheduler. First isolate the network-timing/ranking question with a controlled bandwidth-only experiment across several trace offsets: determine whether HELIX Slow/Fast ordering is itself stable and how often phase shifts change the strict max-TPOT boundary. This decides whether the Evaluator should target conservative screening/robust capacity rather than exact single-trace fine ranking. Keep the network red-line equations unchanged during this diagnostic.
