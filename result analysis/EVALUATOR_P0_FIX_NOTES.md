# Evaluator P0 Semantic Fix Notes

Branch: `fix/evaluator-p0-semantics`

This branch applies the first correctness pass after the HELIX MVP review. It intentionally does **not** add the planned reference serving evaluator or deployment search.

## 1. Network commitment now affects active lifetime

Previous behavior:

- Prefill transitioned to Decode after compute time only.
- Decode equivalent-token progress advanced with compute time only.
- Network was checked as a conservative bandwidth commitment, but the request could release that commitment before the corresponding network service window had elapsed.

New default behavior (`EvaluatorConfig.conservative_network_lifetime=True`):

- If a Prefill/Decode phase has cross-node traffic and its SLA network budget is positive, the request retains its resource commitment for the full corresponding SLA service window.
- This is deliberately conservative reservation semantics, not a high-fidelity link scheduler.
- `conservative_network_lifetime=False` preserves the previous compute-only progress semantics for ablation/regression tests.

Expected consequence: the old HELIX capacity values are **not expected to remain unchanged**. Re-run the experiment and treat the new output as a new version of the evaluator.

## 2. Capacity search no longer silently relies on pure binary search

`find_capacity()` now:

1. explicitly finds one safe and one unsafe point;
2. samples a coarse grid inside the bracket;
3. probes higher intensities for obvious safe re-entry;
4. raises an error if sampled feasibility is non-monotonic;
5. performs local numerical refinement only after the sampled monotonicity check.

The returned `CapacityResult` records:

- `monotonicity_verified_on_samples`;
- `verification_probe_intensities`.

This is still sampled verification, not a mathematical proof of monotonicity.

## 3. Decode block boundary handling is unified

`RequestRuntime.block_index()` and `resource_context()` now use the same epsilon policy. Decode progress is snapped to a block boundary when it is within `progress_epsilon`.

This removes the previous edge case where the scheduler could treat a boundary as crossed while resource accounting still used the previous block.

## 4. Same-timestamp violations are retained

Resource accounting now collects the complete violation set at the event timestamp instead of returning immediately after the first object encountered.

`EvaluationResult` keeps:

- `first_violation` for compatibility;
- `first_violations` for the complete same-timestamp set.

This matters for the HELIX MVP because equal-capacity stage links can violate simultaneously; `slow-0` or `fast-0` should not automatically be interpreted as a unique physical bottleneck.

## 5. Diagnostics are richer

Trace snapshots now optionally record:

- event types in the atomic event batch;
- request phase;
- request resource context;
- Decode progress.

`helix_demo.py` now outputs:

- safe and unsafe run summaries;
- peak Prefill/Decode concurrency;
- minimum link headroom;
- peak memory and utilization;
- all first-event violations;
- monotonicity verification metadata;
- safe/unsafe arrival-rate bracket.

## 6. Small model/validation cleanup

- `activation_element_bytes` is separated from KV element precision while preserving the old fallback behavior.
- duplicate stage boundaries are rejected.
- workspace/margin are only added to nodes that actually host model layers.

## 7. Added regression tests

New tests cover:

- old continuous-progress behavior when conservative network lifetime is explicitly disabled;
- network commitment extending active lifetime under the new default;
- epsilon-consistent block context;
- simultaneous network violations being retained;
- sampled monotonicity verification metadata.

## 8. Commands to run in Codex/local environment

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.helix_demo | tee /tmp/helix_p0_result.json
```

Recommended comparison:

```bash
git diff main...fix/evaluator-p0-semantics
```

For a direct semantic ablation, run the same HELIX setup once with:

```python
EvaluatorConfig(decode_block_size=16, conservative_network_lifetime=False)
```

and once with the new default `True`.

## 9. What this branch does not claim

This branch does not establish that the evaluator is a correct predictor of real serving capacity. The next decisive step remains a multi-pipeline ranking comparison against an independent reference serving evaluator.
