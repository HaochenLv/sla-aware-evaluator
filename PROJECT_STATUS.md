# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 remains the leading scheduler-free safety candidate. E39 is a rejected experimental split-semantics ablation; no production semantics or network red-line changed.

## Experiment log
- **E31/E34 existing safety checks**: at the previously tested candidate-safe points, pinned HELIX was safe in all 8 workload×pipeline checks. This remains pointwise evidence only; E39 shows HELIX feasibility under finite-trace intensity scaling is not globally monotone, so these checks must not be overstated as proof of a single true HELIX capacity frontier.
- **E36/E37**: seed19 Slow over-conservatism combines hidden phase-dependent Prefill exposure and a trajectory-alignment mismatch. Cheap event/request observables are insufficient: identical visible Arrival state can produce exposure `3.4334%` versus `40.3133%`, and `87.4482%` versus `95.3687%`.
- **E38 trajectory audit**: current progression accumulates 5 ms fixed overhead on each of 415 Decode tokens, inserting `2.075 s` into the victim Decode span. Removing fixed from progression removes the specific false overlap at seed19 intensity `0.0153`, but also makes the trajectory earlier than raw HELIX because other end-to-end runtime components are not replayed.
- **E39 split-semantics setup**: generate state trajectory with `fixed_overhead_s=0`, but recompute TTFT/TPOT budgets and network reservations using the original 5 ms fixed overhead plus the unchanged full Prefill blocking-service debt. Seed19, both Slow/Fast, grid `0.010–0.022` step `0.0001`.
- **E39 candidate shift**: original E31 frontier is `0.0152 safe / 0.0153 unsafe` for both pipelines. Split semantics moves both to `0.0159 safe / 0.0160 unsafe` (+4.6% candidate-safe intensity), with first split violation still an Arrival-triggered `N_P=1,N_D=1` full-debt `sla_time_decode` event.
- **E39 falsification — Slow optimism**: pinned HELIX is already **unsafe** at the split candidate-safe point `0.0159` (`max TPOT 0.168847 s > 0.150 s`) and at `0.0160` (`0.211314 s`). Therefore the split fixed-overhead model violates the primary safety requirement and is rejected.
- **E39 Fast**: HELIX is safe at split `0.0159`, `0.0160`, and `0.0168`, but this does not rescue the model because Slow already exhibits candidate optimism.
- **E39 new HELIX finding — nonmonotone finite-trace feasibility**: for seed19 Slow, HELIX is unsafe at `0.0159` and `0.0160` but becomes safe again at `0.0168` (`max TPOT 0.117426 s`). Thus changing intensity shifts request phase alignment enough that finite-trace SLA feasibility is not a monotone function of intensity. The earlier E35 bracket around `0.01838` is therefore a **local sampled safe/unsafe crossing**, not a globally unique true capacity frontier. Sparse sampled monotonicity in E35 did not rule out narrower unsafe pockets between samples.

## Interpretation
E39 is a useful negative result. Separating fixed SLA overhead from physical progression improves the coarse candidate frontier numerically but breaks safety on seed19 Slow, so it must not replace E31. More importantly, the experiment exposes a deeper validation issue: the pinned HELIX finite-workload reference has phase-alignment-driven safe/unsafe pockets as arrival intensity changes. Therefore a single HELIX `frontier` obtained by binary search is not well-defined unless monotonicity is imposed by a stronger capacity definition or by robust aggregation over phases/workloads. This is consistent with E28/E36 phase sensitivity and strengthens the rationale for a conservative scheduler-free envelope, but it requires re-auditing how candidate safety is validated.

## Next
1. Map seed19 Slow HELIX feasibility around and below the original E31 candidate (`0.0152`) with a targeted intensity scan to determine whether any unsafe pocket exists inside the candidate-safe region.
2. If the original prefix is clean on a dense scan, extend the same safety-prefix audit to the other E34 workloads before making any capacity-accuracy claim.
3. Treat E33/E35 brackets as local crossings only; do not call them exact global HELIX capacity frontiers.
4. Keep E31 unchanged, reject E39 split semantics, and do not merge or modify main.
