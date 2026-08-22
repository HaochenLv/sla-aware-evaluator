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

## 7. Capacity search and right censoring

`find_reference_capacity()` uses the same workload-intensity scaling and sampled monotonicity discipline as the Conservative capacity search.

Matching the numerical search policy is intentional; the serving evaluator itself remains independent.

The first 30-second smoke attempt exposed an important finite-workload case: the Conservative evaluator remained feasible through the configured `max_intensity=16`, so the old search raised `no unsafe intensity found below max_intensity` before Reference execution began.

That is now treated as a valid **right-censored** observation, not as an exception. If the evaluated search limit is still feasible, both capacity APIs return:

```text
safe_intensity = max_intensity
unsafe_intensity = null
right_censored = true
representative_unsafe_run = null
```

Its interpretation is:

```text
C >= max_intensity
```

The code does not fabricate an unsafe point or change the workload merely to force a bracket. The smoke harness continues to the other evaluator even when one side is right-censored.

If a true safe/unsafe frontier is observed, the original bracket + sampled monotonicity + local refinement behavior is unchanged. If sampled safe re-entry is observed after an unsafe point, the scalar-capacity assumption must still be revisited rather than hidden.

## 8. First experiment: execution smoke test only

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
- identical SLA and underlying compute data for both evaluators;
- `max_intensity=16`, with explicit right-censoring if the frontier is not observed.

It reports Conservative and Reference capacity bounds side by side plus Reference TTFT/TPOT/memory/first-violation diagnostics and evaluator wall time.

Two pipelines are **not** enough to validate ranking correlation. This run only asks:

1. Does Reference v0 execute and drain correctly?
2. Does explicit service produce sensible TTFT/TPOT violations?
3. If both sides expose an ordering within the observed range, do Slow/Fast agree or disagree?
4. Is runtime acceptable before scaling to a larger pipeline set?
5. Does sampled feasibility remain monotonic under the reference microbatch model?
6. Does the harness correctly preserve right-censored capacity results instead of failing?

A censored 30-second capacity is not a reason to tune the workload. A later ranking-capacity experiment may use a longer workload, such as the already exercised 120-second trace, if an observed frontier is needed.

## 9. Result-driven repair checkpoint

The first Codex smoke run had the following outcome:

```text
21/21 tests passed
reference_demo terminated before Reference execution
cause: Conservative 30-second workload remained safe at max_intensity=16
```

The branch now contains only the minimal repair implied by that result:

- Conservative capacity search supports right-censored upper search ranges;
- Reference capacity search supports the same right-censoring semantics;
- `reference_demo.py` serializes `null` unsafe bounds/runs and continues both evaluators;
- `helix_demo.py` is compatible with the new result type;
- regression tests cover bracketed and right-censored outcomes for both evaluators.

No scheduler, profile, workload, SLA, bandwidth, routing, or deployment-search semantics were changed by this repair.

The next action is therefore **rerun the same smoke experiment**, not add more model features.

## 10. Do not do yet

Before the repaired smoke result is reviewed, do not add:

- deployment search;
- replicas;
- dynamic routing;
- migration;
- preemption;
- vLLM-specific scheduler behavior;
- context-aware dimensions not observed by HELIX;
- large topology sweeps;
- 24-pipeline rank-correlation experiments.

The next branch decision should be based on the repaired smoke result, not on adding more simulator features.
