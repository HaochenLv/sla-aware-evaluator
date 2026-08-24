# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#14 are research-only; do not merge.

## Experiment log
- **E7**: original Conservative capacity was far above HELIX despite the same fixed Pipeline/workload/SLA.
- **E9–E11**: HELIX diagnostics establish real Prefill→Decode queue/interference. One-/multi-Prefill controlled tests support full active Prefill compute as a simple conservative interference-debt candidate.
- **E12**: that guard on the original fast Conservative trajectory rejects too late.
- **E13**: unconditional 2× Decode looked close to HELIX but over-doubled n_decode>=2 and was therefore only a diagnostic artifact.
- **E14 / exact singleton rule**: 2× only for n_decode==1 is semantically correct but alone remains feasible beyond intensity 0.030, so it does not recover the HELIX frontier.
- **E15 / exact singleton + active-Prefill debt**: on a 0.0001 intensity grid, both Slow/Fast are safe at 0.0131 and unsafe at 0.0132, i.e. safe >=0.008469 rps and unsafe <=0.008533 rps. HELIX safe frontiers are higher: Slow safe at 0.0134 (0.008663 rps), Fast safe at 0.0136 (0.008792 rps). Thus this minimal pair is conservative relative to the current HELIX trace, only ~2–4% below HELIX's safe points, but it is too coarse to preserve the small Slow/Fast frontier separation.

## Interpretation
The best current explanation is now two-part: runtime-aligned Decode compute is required for semantic correctness, and a simple Prefill-blocking debt is required to prevent optimistic overlap. The network red-line equations do not need to change. A full active-Prefill debt effectively assumes an active Prefill can block a Decode across the whole Pipeline; this is simple and conservative, but intentionally ignores favorable stage phasing and therefore can reject some HELIX-safe overlap.

## Next
Do not integrate yet. Validate E15 on a few additional workload seeds / prompt-length mixes. If it stays conservative without becoming excessively pessimistic, promote the exact singleton rule plus Prefill interference debt as a controlled evaluator revision. If ranking collapses broadly, investigate a slightly tighter stage-aware/remaining-Prefill bound without recreating a scheduler.
