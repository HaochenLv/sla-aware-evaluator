# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#13 are research-only; do not merge.

## Experiment log
- **E7**: original Conservative capacity was far above HELIX despite the same fixed Pipeline/workload/SLA.
- **E9–E11**: HELIX diagnostics establish real Prefill→Decode queue/interference and motivate a simple active-Prefill interference debt.
- **E12**: that guard on the original Conservative trace rejects too late.
- **E13 / unconditional 2× Decode diagnostic**: forcing every Decode profile value to 2× moved the frontier close to HELIX, but this was intentionally stronger than HELIX semantics for n_decode>=2.
- **E14 / exact HELIX singleton rule**: applying 2× only when n_decode==1 and using the raw profile for n_decode>=2 leaves both Slow/Fast feasible through intensity 0.030 (>0.01939 rps), including every HELIX frontier probe 0.0132–0.0138. Therefore the near-match in E13 was an artifact of over-doubling multi-Decode states. Exact singleton profile alignment is necessary for semantic correctness but is not sufficient to explain the capacity gap.

## Revised diagnosis
Two distinct issues exist: (1) the profiler adapter underestimates singleton Decode service by 2× versus HELIX runtime; (2) the Conservative evaluator still lacks the discrete Prefill→Decode blocking observed in HELIX. Correcting only (1) does not recover HELIX capacity. The red-line equations remain structurally intact; the unresolved part is the compute-side interference/queue term supplied to the TPOT budget.

## Next
Combine the exact singleton Decode rule with the experimentally supported active-Prefill interference debt as a diagnostic, without changing the network equations or adding a scheduler. Test whether that minimal pair recovers the HELIX safe/unsafe frontier.
