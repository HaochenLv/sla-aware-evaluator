# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#16 are research-only; do not merge.
- Best current candidate remains: exact HELIX singleton Decode profile alignment + full active-Prefill compute debt. It is a conservative screening-bound candidate, not yet a production revision.

## Experiment log
- **E7**: original Conservative capacity was far above HELIX despite the same fixed Pipeline/workload/SLA.
- **E9–E14**: HELIX diagnostics identify Prefill→Decode interference; exact singleton Decode alignment is semantically necessary, and full active-Prefill compute debt is the simplest useful conservative guard.
- **E15**: exact singleton + active-Prefill debt is only ~2–4% below HELIX safe points on the original workload, but ties Slow/Fast.
- **E16**: four additional workload variants keep the candidate conservative in 8/8 pipeline×workload frontier checks; HELIX is already TPOT-unsafe at +5% in 6/8 checks. No candidate optimism was observed.
- **E17 / bandwidth-only HELIX capacity ordering**: model, A100 nodes, 8x10-layer placement, node order, workload seed, SLA, and HELIX runtime were fixed; only all seven inter-stage links changed from 2.5 to 10 Gbps. Across six Azure trace offsets (0/50/100/150/200/250), Fast never had a lower full capacity frontier. Offsets 0 and 250 resolve `Fast > Slow`; offsets 50/100/150/200 have touching brackets at the 1e-4 intensity tolerance, with Fast's safe edge equal to Slow's unsafe edge. No `Slow > Fast` reversal and no sampled safe re-entry occurred. The Fast safe-intensity gain is tiny: roughly one or two refinement steps (~0.6–1.5%).

## Interpretation
E17 shows that the E16 single shared-intensity case where Slow was safe and Fast unsafe does not imply a reversed HELIX capacity frontier. In the controlled bandwidth-only experiment, higher bandwidth consistently shifts the sampled frontier weakly upward, but the effect is very small and often unresolved at the present search tolerance. This explains why the compute-driven Conservative candidate ties Slow/Fast: network bandwidth can alter HELIX phase timing and fine capacity, while current Conservative network equations act as red-line feasibility rather than event-time progression. Reproducing that tiny phase-sensitive separation exactly would risk recreating scheduler/runtime semantics and is not yet justified.

## Next
Do not integrate E15 yet and do not add a scheduler. Run a small HELIX bandwidth sweep on two representative trace offsets (one short 17-request window and one 40–50-request window) to test whether capacity changes monotonically and smoothly across several bandwidth levels. If the effect remains small, position the Evaluator primarily as a conservative screening / robust-capacity estimator and treat fine network-only ranking as secondary. Keep the network red-line equations unchanged during this diagnostic.
