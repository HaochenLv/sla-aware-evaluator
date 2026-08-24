# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- This branch is a research-only E24 ablation. It does not change state progression, HELIX scheduling, or the normalized-cost network red-line equation.
- Candidate under test: exact HELIX singleton Decode alignment + full active-Prefill interference debt placed inside Decode's remaining SLA/network budget.

## Experiment log
- **E15–E16 baseline**: guard-only exact-singleton + full active-Prefill debt stayed conservative across tested workload variants, but repeatedly tied Slow/Fast.
- **E23 magnitude check**: full active-Prefill debt upper-bounds measured total Decode compute-side excess in 44/44 controlled HELIX cases; worst observed excess/debt ratio is `0.953687`, so a global multiplier below 1 is not supported.
- **E24 setup**: added an offline budget-consistent replay using `Delta_D = tau_D - T_decode - I_prefill - T_queue - T_fix`. It preserves the existing Conservative trajectory and snapshot Prefill network commitments, replacing only each active Decode's network contribution with the smaller debt-aware remaining budget. The script compares the resulting frontier with the earlier guard-only frontier and is prepared to probe pinned HELIX at its safe/unsafe edges.
- **E24 execution status**: Actions run `32700730311` failed before any workflow step started (`steps=null`, no job log). Therefore no E24 frontier/result is claimed. This is the same runner-infrastructure blocker affecting E21, not an experimental failure.
- **E25 related audit**: archived successful E15/E16 artifacts show every guard-only first-unsafe state is `N_P=1, N_D=1`; exact-singleton Decode + fixed overhead leaves only `0.033 s` TPOT budget, while the first active Prefill debt is `0.11584–0.68192 s`. Thus the guard-only frontier is already a discrete overlap-onset cliff rather than a marginal budget crossing.

## Interpretation
E24 remains useful because placing interference debt inside `Delta_D` is the mathematically consistent way to let the unchanged SLA-to-network red line see compute-side interference. However, E25 implies E24 can only differ from the guard-only frontier if debt-aware network reservation rejects a state that is still compute-time-feasible under the guard. We cannot say whether that happens until the workflow actually executes. Do not infer a result from the model structure alone.

## Next
Keep E24 pending and re-run when Actions runners recover. In parallel, investigate the more fundamental coarseness identified by E25: whether a simple remaining-Prefill exposure bound can preserve the conservative magnitude evidence of E23 without charging an entire Prefill immediately at the first overlap event. Do not fit a global debt discount and do not reconstruct a scheduler.
