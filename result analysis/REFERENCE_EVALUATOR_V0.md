# Reference Evaluator v0

Branch: `feat/reference-evaluator-v0`

This branch starts the independent serving-reference path used to test the core research hypothesis:

> Does the cheap Conservative SLA-Safe Evaluator preserve the ranking of fixed Layer-Level pipelines under a more explicit serving model?

It does **not** yet claim that the reference simulator reproduces vLLM, HELIX runtime scheduling, or real hardware.

## 1. Why this evaluator is independent

The Conservative Evaluator derives a minimum network commitment from remaining SLA time:

```text
profiled compute -> remaining TTFT/TPOT budget -> required bandwidth -> red line
```

Reference v0 intentionally does not use that formula. Instead it executes explicit services:

```text
Arrival
-> queued Stage compute
-> queued physical-link D/B transfer
-> next Stage
-> ...
-> Prefill complete
-> Decode token 0 through all Stages/links
-> Decode token 1 ...
-> Finish
```

This gives an independent target for ordering/capacity experiments.

## 2. GPU service semantics

Each physical GPU node is a queued server.

- Prefill operations are strict FIFO and served one request at a time.
- Decode operations use a deterministic microbatch rule.
- When a GPU becomes idle and the FIFO head is Decode, contiguous queued Decode operations for the same Layer stage are served together.
- HELIX `decode_bs2time.csv` is therefore queried with the actual queued microbatch size in Reference v0, rather than the Conservative Evaluator's active-Decode-count proxy.
- For heterogeneous analytical profiles, a Decode microbatch completes at the maximum per-request stage service time returned for the batch.

This scheduling rule is deliberately simple and reproducible. It is not claimed to be vLLM/HELIX scheduling.

## 3. Network semantics

Each physical link is an explicit FCFS server.

For every Stage boundary:

```text
service_time = activation_bytes / physical_link_capacity
```

Multi-hop routes traverse links sequentially. Transfers from different requests contend through the same physical-link queue.

There is no `remaining SLA -> required bandwidth` calculation in Reference v0.

## 4. SLA semantics

To keep the first comparison aligned with the current Conservative model:

- TTFT is measured at end of Prefill, after configured fixed/queue overhead.
- Strong TPOT is checked for every full end-to-end Decode-token interval.
- Actual GPU queueing and physical-link queueing are included in those observed times.

This TTFT definition is a deliberate alignment choice for v0. A future real-runtime comparison may instead measure first generated-token latency.

## 5. Memory semantics

Reference v0 keeps the current coarse memory convention:

- static model weights + workspace + margin;
- full prompt KV reserved while Prefill is active;
- Decode KV uses exact completed-token count rather than a Decode block upper bound;
- memory is released when the request finishes.

Therefore the initial decisive comparison should interpret memory ranking cautiously. The main independence gain in v0 is compute/network execution and measured SLA latency.

## 6. Shared profile source, different execution model

Reference and Conservative evaluators intentionally use the same underlying HELIX/analytical compute data. Otherwise profile differences would confound the experiment.

New stage-level profiler methods expose the same service data:

```text
prefill_stage_time(...)
decode_stage_time_per_token(...)
```

The existing pipeline-level Conservative methods remain unchanged in meaning and are sums of the stage-level values.

## 7. Capacity search

`find_reference_capacity()` uses the same workload-intensity scaling and sampled monotonicity discipline as the Conservative capacity search.

Matching the numerical search policy is intentional; the serving evaluator itself remains independent.

If sampled safe re-entry is observed, the scalar-capacity assumption must be revisited rather than hidden.

## 8. First experiment: smoke test only

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.reference_demo \
  > outputs/reference_v0_smoke.json
python3 -m json.tool outputs/reference_v0_smoke.json > /dev/null
```

The smoke experiment uses:

- HELIX LLaMA-2 70B / A100 profile;
- the existing Slow/Fast two-pipeline control;
- a 30-second generated Azure-derived finite workload;
- identical SLA and underlying compute data for both evaluators.

It reports Conservative and Reference capacity brackets side by side plus Reference TTFT/TPOT/memory/first-violation diagnostics and evaluator wall time.

Two pipelines are **not** enough to validate ranking correlation. This run only asks:

1. Does Reference v0 execute and drain correctly?
2. Does explicit service produce sensible TTFT/TPOT violations?
3. Does Slow/Fast ordering agree or disagree in this minimal case?
4. Is runtime acceptable before scaling to ~24 pipelines?
5. Does sampled feasibility remain monotonic under the reference microbatch model?

## 9. Do not do yet

Before the smoke result is reviewed, do not add:

- deployment search;
- replicas;
- dynamic routing;
- migration;
- preemption;
- vLLM-specific scheduler behavior;
- context-aware dimensions not observed by HELIX;
- large topology sweeps;
- 24-pipeline rank-correlation experiments.

The next branch decision should be based on the smoke result, not on adding more simulator features.
