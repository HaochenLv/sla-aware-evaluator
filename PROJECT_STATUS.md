# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- This branch is a historical research-only E24 ablation. It does not change state progression, HELIX scheduling, or the normalized-cost network red-line equation.
- IMPORTANT: E24 uses the older **compute-only Prefill debt** and is no longer the preferred magnitude candidate after the corrected E23/E26 result.

## Experiment log
- **E15–E16 baseline**: guard-only exact-singleton + full active-Prefill compute debt stayed conservative across tested workload variants, but repeatedly tied Slow/Fast.
- **E23 correction / E26**: compute-only Prefill debt covers 42/44 controlled E10/E11 cases, not 44/44. The preferred magnitude candidate is now blocking-service debt = profiled compute + independently profiled/bounded blocking overhead, which covers 44/44 archived cases.
- **E24 setup**: the branch demonstrates the mathematically consistent placement of a Prefill debt inside `Delta_D = tau_D - T_decode - I_prefill - T_queue - T_fix`, so the unchanged network red line sees the reduced Decode time budget. This structural idea remains relevant, but the branch's compute-only `I_prefill` input is obsolete as a final candidate.
- **E24 execution status**: Actions run `32700730311` failed before any workflow step (`steps=null`, no job log). No E24 frontier/result is claimed.
- **E25/E28 timing result**: immediate full-debt charging creates an overlap-onset cliff, and archived E10 timing sweeps show actual exposure is stage/execution-phase sensitive rather than monotone in a simple timing offset.

## Interpretation
Two pieces should be kept separate. The **structural placement** of interference inside Decode's remaining SLA budget is still correct for the current modeling logic. The **magnitude input** used by this historical E24 branch is not: compute-only debt misses 2/44 controlled cases. A successor ablation, if run, must use Prefill blocking-service debt and remain opt-in. E28 also warns that replacing full debt with a naive monotone timing heuristic is not justified.

## Next
Do not spend runner time re-validating this obsolete compute-only branch as a candidate. Keep it as provenance for the `Delta_D` integration structure. Any successor should use the E26 blocking-service debt and compare full scheduler-free worst-case exposure against only a separately justified event-level refinement. Do not merge.
