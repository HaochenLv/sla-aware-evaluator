# Reference v0 TPOT Diagnostic Checkpoint

Branch: `feat/reference-evaluator-v0`

## Why this diagnostic exists

The repaired 30-second HELIX smoke successfully executed Reference v0, but the first unsafe run showed a large TPOT jump when Decode overlap appeared:

- Slow: about `0.52 s`;
- Fast: about `0.37 s`;
- SLA: `0.150 s`;
- safe run `peak_decode=1`, unsafe run `peak_decode=2`.

The first diagnostic then showed that direct per-token `D/B` transfer time was tiny, while more than 80% of the failing TPOT remained in a residual waiting/synchronization bucket. Multiplying link capacity by `1e6` did not remove the failure: both pipelines still had roughly `0.344 s` TPOT. The failing token also used Decode batch size 1 at every stage despite two active Decode requests.

That evidence localizes the problem to waiting/synchronization, but the old residual mixed GPU queue wait, link queue wait, and pipeline/batching effects. It is still not enough evidence to rewrite the scheduler.

## Exact observational tracing

Reference v0 now accepts an optional observational `trace_sink`. The scheduler is unchanged. The trace only records timestamps for:

- resource queue enqueue;
- resource service start;
- resource service completion;
- Decode token start;
- Decode token completion.

The trace hook does **not** modify queue order, microbatch formation, service durations, event ordering, capacity search, SLA checks, or feasibility.

For the first TPOT-violating token, the diagnostic now measures:

```text
observed TPOT
= GPU queue wait
+ GPU service elapsed
+ link queue wait
+ link service elapsed
+ configured overhead
+ unattributed
```

`unattributed` should be approximately zero if all simulated time is closed by the trace. This is stronger than the previous residual calculation because queue waiting is measured directly from enqueue/start timestamps instead of inferred as a remainder.

The diagnostic also reports:

- GPU queue wait per stage;
- GPU service elapsed per stage;
- link queue wait per physical link;
- link service elapsed per physical link;
- actual Decode batch size at each GPU service start;
- profiler-returned per-stage Decode service for comparison with elapsed batch service.

For backwards readability, `residual_wait_s` now means the **measured** sum:

```text
GPU queue wait + link queue wait
```

and `unattributed_s` is reported separately.

## Network counterfactual

The same scaled workload and same first-unsafe intensity are replayed with every physical-link capacity multiplied by `1e6`.

The counterfactual is traced with the same instrumentation and reports:

- GPU queue wait;
- GPU service elapsed;
- link queue wait;
- link service elapsed;
- unattributed time;
- Decode batch sizes.

This lets us distinguish two different network effects:

1. direct link service / link queueing;
2. network-induced changes in GPU/pipeline phase alignment that later appear as GPU queue wait.

Therefore a large TPOT reduction after increasing bandwidth is not automatically interpreted as direct transmission cost; the exact queue attribution must show where the removed time went.

## Regression tests

The diagnostic tests now cover three invariants:

1. adding a trace sink does not change the Reference evaluation result;
2. a link-driven toy TPOT failure closes as explicit GPU service + link service with zero queue/unattributed time;
3. a deliberately constructed Decode interference case attributes the failing token's delay to GPU queue wait and closes the TPOT exactly.

The existing infinite-network toy counterfactual remains as a control showing that a genuinely link-driven failure can disappear when link capacity is removed as a bottleneck.

## Run

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.reference_diagnostic_demo \
  > outputs/reference_v0_tpot_queue_attribution.json
python3 -m json.tool outputs/reference_v0_tpot_queue_attribution.json > /dev/null
```

The HELIX diagnostic intentionally reuses the same:

- LLaMA-2-70B / A100 profile;
- 30-second generated Azure-derived workload;
- TTFT `2.0 s`;
- TPOT `0.150 s`;
- fixed overhead `0.005 s`;
- Slow/Fast pipelines;
- Reference first-unsafe intensity search.

## Next decision

Do **not** start the multi-pipeline ranking experiment yet.

The next experiment should answer:

1. what fraction of failing TPOT is exact GPU queue wait;
2. what fraction is exact link queue wait;
3. whether `unattributed_s` is effectively zero;
4. which stages accumulate the GPU queue wait;
5. whether the `1e6`-bandwidth counterfactual mainly reduces link wait or changes downstream GPU queue wait;
6. whether the failing token remains batch size 1 at all stages.

If the trace shows that the dominant delay is GPU queue wait while active Decode requests fail to batch together, that is strong evidence to review the Reference Decode batching policy. If link queueing is dominant, network contention should be fixed or modeled before touching Decode scheduling.

This checkpoint adds measurement hooks only. It does not yet replace the Reference scheduler.
