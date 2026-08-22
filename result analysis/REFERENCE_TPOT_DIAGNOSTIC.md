# Reference v0 TPOT Diagnostic Checkpoint

Branch: `feat/reference-evaluator-v0`

## Why this diagnostic exists

The repaired 30-second HELIX smoke test successfully executed Reference v0, but exposed two signals that must be understood before any 20-30 Pipeline ranking experiment:

- Reference safe capacity was roughly `0.016 req/s` for both Slow and Fast, far below the Conservative result.
- Reference remained safe with `peak_decode=1`, then the first unsafe run appeared with `peak_decode=2` and a large TPOT jump (`~0.52 s` Slow, `~0.37 s` Fast vs `0.15 s` SLA).

This does **not** yet justify changing the scheduler. The next step is to identify what portion of the failing TPOT interval comes from explicit service versus waiting/synchronization under the current Reference v0 semantics.

## Added diagnostic path

`reference_diagnostics.py` wraps the existing Stage profiler without changing Reference v0 execution. For the request/token that first violates TPOT it reports:

```text
observed TPOT
= profiled GPU stage service
+ explicit D/B link service
+ configured overhead
+ residual waiting/synchronization
```

The residual is deliberately not named `GPU wait`. It can include:

- GPU queueing;
- link queueing;
- batching/synchronization effects not represented by the summed per-request profile calls.

For the HELIX adapter, Decode stage time does not depend on request context and all stages use the same A100 profile, so the recorded per-request stage service for a given microbatch matches the batch stage duration used by Reference v0.

## Network counterfactual

The diagnostic then replays the *same scaled finite workload at the same intensity* with every physical-link capacity multiplied by `1e6`.

This counterfactual is diagnostic only; it is not a new serving model or a result used for capacity ranking.

Interpretation:

- if the original TPOT violation disappears or drops sharply, network transfer/queueing is materially responsible;
- if the violation persists with similar magnitude, the dominant cause lies in GPU queueing / Decode batching / synchronization under Reference v0 rather than physical-link service.

## Run

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.reference_diagnostic_demo \
  > outputs/reference_v0_tpot_diagnostic.json
python3 -m json.tool outputs/reference_v0_tpot_diagnostic.json > /dev/null
```

The diagnostic demo intentionally reuses the same:

- HELIX LLaMA-2-70B / A100 profile;
- 30-second generated Azure-derived workload;
- TTFT `2.0 s`;
- TPOT `0.150 s`;
- fixed overhead `0.005 s`;
- Slow/Fast pipelines.

It first recovers each pipeline's Reference unsafe intensity and then diagnoses that exact first-unsafe point.

## Decision rule for the next code change

Do **not** enter the multi-pipeline ranking experiment yet.

First inspect:

1. `residual_wait_fraction` at the first TPOT failure;
2. `decode_batch_sizes` across the failing token's stages;
3. `gpu_profile_service_s` versus `explicit_link_service_s`;
4. whether the `1e6`-bandwidth counterfactual is still unsafe;
5. how much the counterfactual max TPOT changes.

If waiting remains dominant and the infinite-network replay still violates TPOT, then the next code change should focus on the Reference Decode batching/queueing policy. If the violation disappears under the network counterfactual, inspect explicit link contention before touching GPU scheduling.

This checkpoint deliberately changes diagnostics only. It does not modify Conservative semantics, Reference scheduling, workload, SLA, HELIX profiling, physical topology, or deployment search.
