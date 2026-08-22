# Evaluator Semantic Fix Notes

Branch: `fix/evaluator-p0-semantics`

This branch contains the first correctness pass after the HELIX MVP review plus the follow-up semantic ablation decision. It still does **not** add the planned reference serving evaluator or deployment search.

## 1. Default progress semantics: profiling-driven compute-only

The follow-up ablation showed that making every cross-node Prefill/Decode request remain active for the full TTFT/TPOT window changes the state trajectory substantially and is too strong to use as the default evaluator semantics.

Current default (`EvaluatorConfig.conservative_network_lifetime=False`):

- Prefill transitions according to profiled Prefill service time plus the configured fixed/queue overhead convention already used by the event engine.
- Decode equivalent-token progress is driven by the profiled Decode per-token service time.
- Network remains a conservative SLA-derived reservation/red-line check.
- Network reservation does **not** directly slow the state machine.

This preserves the original v2.1 modeling intent: profiling drives state evolution while SLA maps the current state to a conservative network requirement.

## 2. Full-SLA-window lifetime is retained only as an ablation

Setting:

```python
EvaluatorConfig(conservative_network_lifetime=True)
```

enables the stronger policy in which a cross-node phase/token remains active for the full corresponding SLA service window.

This policy is retained for research comparison only. It should be described as:

`full-SLA-window lifetime ablation`

and not as the default P0 correctness fix.

Controlled HELIX ablation results showed that the large capacity shift on the first P0 branch was primarily attributable to this lifetime policy. The compute-only variant largely restored the original Fast/Slow capacity ratio while keeping the other engineering fixes.

## 3. Capacity search no longer silently relies on pure binary search

`find_capacity()` now:

1. explicitly finds one safe and one unsafe point;
2. samples a coarse grid inside the bracket;
3. probes higher intensities for obvious safe re-entry;
4. raises an error if sampled feasibility is non-monotonic;
5. performs local numerical refinement only after the sampled monotonicity check.

The returned `CapacityResult` records:

- `monotonicity_verified_on_samples`;
- `verification_probe_intensities`.

This is sampled verification, not a mathematical proof of monotonicity.

## 4. Decode block boundary handling is unified

`RequestRuntime.block_index()` and `resource_context()` use the same epsilon policy. Decode progress is snapped to a block boundary when it is within `progress_epsilon`.

This removes the previous edge case where the scheduler could treat a boundary as crossed while resource accounting still used the previous block.

## 5. Same-timestamp violations are retained

Resource accounting collects the complete violation set at the event timestamp instead of returning immediately after the first object encountered.

`EvaluationResult` keeps:

- `first_violation` for compatibility;
- `first_violations` for the complete same-timestamp set.

This matters for the HELIX MVP because equal-capacity stage links can violate simultaneously; `slow-0` or `fast-0` should not be interpreted as a unique physical bottleneck.

## 6. Diagnostics are richer

Trace snapshots optionally record:

- event types in the atomic event batch;
- request phase;
- request resource context;
- Decode progress.

`helix_demo.py` outputs:

- an explicit `progress_policy` label;
- safe and unsafe run summaries;
- peak Prefill/Decode concurrency;
- minimum link headroom;
- peak memory and utilization;
- all first-event violations;
- monotonicity verification metadata;
- safe/unsafe arrival-rate bracket.

The default HELIX demo now uses `progress_policy = compute_only`.

## 7. Small model/validation cleanup

- `activation_element_bytes` is separated from KV element precision while preserving fallback behavior.
- duplicate stage boundaries are rejected.
- workspace/margin are only added to nodes that actually host model layers.

## 8. Regression tests

Tests cover:

- compute-only being the default progress policy;
- continuous Decode progress under the default policy;
- the full-SLA-window ablation extending request lifetime;
- epsilon-consistent block context;
- simultaneous network violations being retained;
- sampled monotonicity verification metadata.

## 9. Commands for Codex/local validation

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sla_aware_mvp.helix_demo | tee /tmp/helix_compute_only_default.json
```

Recommended comparison:

```bash
git diff main...fix/evaluator-p0-semantics
```

## 10. Current research interpretation

The conservative evaluator is still **not** validated as a predictor of real serving capacity or Pipeline ranking.

Current status:

- mechanism implementation: working MVP;
- default progress abstraction: compute-only;
- full-SLA-window lifetime: retained as an extreme-conservatism ablation;
- next decisive step: compare Pipeline rankings against an independent reference serving evaluator.

Do not connect this evaluator to deployment search until that ranking validation is performed.
