from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, RequestRuntime, SLA
from .evaluator import _network_bytes_by_link, evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .prefill_debt_budget_ablation import _budget_trial, _prefill_blocking_service_s
from .workload import build_helix_azure_conversation_workload


SEED = 19
OFFSET = 0
START = 0.010
STOP = 0.022
STEP = 0.0001
FIXED_OVERHEAD_S = 0.005
BUFFER_FACTOR = 1.05
EPS = 1e-9


def _frontier(trials: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in trials if row["feasible"]]
    first_unsafe = next((row for row in trials if not row["feasible"]), None)
    seen_unsafe = False
    monotonic = True
    for row in trials:
        if not row["feasible"]:
            seen_unsafe = True
        elif seen_unsafe:
            monotonic = False
            break
    return {
        "safe_intensity_lower_bound": safe[-1]["intensity"] if safe else None,
        "unsafe_intensity_upper_bound": first_unsafe["intensity"] if first_unsafe else None,
        "first_unsafe": first_unsafe,
        "monotonic_on_grid": monotonic,
    }


def _contribution(*, request, phase: Phase, pipeline, remaining_s: float) -> dict[str, float] | None:
    runtime = RequestRuntime(request, phase)
    demand = _network_bytes_by_link(runtime, pipeline)
    normalized = 0.0
    for link_id, data_bytes in demand.items():
        if data_bytes <= 0:
            continue
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        if capacity <= 0:
            return None
        normalized += data_bytes / capacity
    result = {link_id: 0.0 for link_id in pipeline.links}
    if normalized <= 0:
        return result
    for link_id, data_bytes in demand.items():
        if data_bytes <= 0:
            continue
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        weight = (data_bytes / capacity) / normalized
        result[link_id] = data_bytes / (weight * remaining_s)
    return result


def _prefill_compute_at_arrival(*, trace, request_by_id, pipeline, profiler) -> dict[str, float]:
    result: dict[str, float] = {}
    for snapshot in trace:
        if "Arrival" not in snapshot.event_types:
            continue
        for rid, phase in snapshot.request_phase.items():
            if phase != Phase.PREFILL.value or rid in result:
                continue
            request = request_by_id[rid]
            result[rid] = profiler.prefill_time(
                request,
                pipeline,
                snapshot.num_prefill,
                snapshot.num_decode,
            )
    return result


def _split_violation(*, snapshot, request_by_id, prefill_compute, pipeline, profiler, budget_sla):
    prefill_ids = [rid for rid, phase in snapshot.request_phase.items() if phase == Phase.PREFILL.value]
    decode_ids = [rid for rid, phase in snapshot.request_phase.items() if phase == Phase.DECODE.value]
    n_prefill = len(prefill_ids)
    n_decode = len(decode_ids)

    blocking_debt_s = sum(
        _prefill_blocking_service_s(
            request=request_by_id[rid],
            pipeline=pipeline,
            profiler=profiler,
            n_prefill=n_prefill,
            n_decode=n_decode,
        )
        for rid in prefill_ids
    )

    link_required = {link_id: 0.0 for link_id in pipeline.links}
    for rid in prefill_ids:
        compute_s = prefill_compute[rid]
        remaining_s = (
            budget_sla.ttft_s
            - compute_s
            - budget_sla.queue_overhead_s
            - budget_sla.fixed_overhead_s
        )
        if remaining_s <= EPS:
            return {
                "kind": "sla_time_prefill",
                "time_s": snapshot.time_s,
                "request_id": rid,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "compute_s": compute_s,
                "capacity_s": budget_sla.ttft_s,
            }
        contrib = _contribution(
            request=request_by_id[rid],
            phase=Phase.PREFILL,
            pipeline=pipeline,
            remaining_s=remaining_s,
        )
        if contrib is None:
            return {"kind": "network_zero_capacity", "time_s": snapshot.time_s, "request_id": rid}
        for link_id, value in contrib.items():
            link_required[link_id] += value

    for rid in decode_ids:
        request = request_by_id[rid]
        context = snapshot.request_context[rid]
        compute_s = profiler.decode_time_per_token(
            request,
            context,
            pipeline,
            n_prefill,
            n_decode,
        )
        remaining_s = (
            budget_sla.tpot_s
            - compute_s
            - blocking_debt_s
            - budget_sla.queue_overhead_s
            - budget_sla.fixed_overhead_s
        )
        if remaining_s <= EPS:
            return {
                "kind": "sla_time_decode",
                "time_s": snapshot.time_s,
                "request_id": rid,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "decode_compute_s": compute_s,
                "prefill_blocking_service_debt_s": blocking_debt_s,
                "required_compute_side_s": (
                    compute_s
                    + blocking_debt_s
                    + budget_sla.queue_overhead_s
                    + budget_sla.fixed_overhead_s
                ),
                "capacity_s": budget_sla.tpot_s,
                "event_types": list(snapshot.event_types),
                "decode_progress": snapshot.request_progress.get(rid),
                "context_tokens": context,
            }
        contrib = _contribution(
            request=request,
            phase=Phase.DECODE,
            pipeline=pipeline,
            remaining_s=remaining_s,
        )
        if contrib is None:
            return {"kind": "network_zero_capacity", "time_s": snapshot.time_s, "request_id": rid}
        for link_id, value in contrib.items():
            link_required[link_id] += value

    for link_id, required in link_required.items():
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        if required > capacity + EPS:
            return {
                "kind": "network",
                "time_s": snapshot.time_s,
                "link_id": link_id,
                "required_bytes_per_s": required,
                "capacity_bytes_per_s": capacity,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "prefill_blocking_service_debt_s": blocking_debt_s,
                "event_types": list(snapshot.event_types),
            }
    return None


def _split_trial(*, intensity, pipeline, workload, profiler, budget_sla):
    scaled = scale_workload(workload, intensity)
    request_by_id = {r.id: r for r in scaled}
    trajectory_sla = SLA(
        ttft_s=budget_sla.ttft_s,
        tpot_s=budget_sla.tpot_s,
        queue_overhead_s=0.0,
        fixed_overhead_s=0.0,
    )
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=trajectory_sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    if not run.feasible:
        return {
            "intensity": intensity,
            "feasible": False,
            "first_violation": {
                "kind": "trajectory_base_" + (run.first_violation.kind.value if run.first_violation else "unknown"),
                "time_s": run.first_violation.time_s if run.first_violation else None,
            },
        }
    prefill_compute = _prefill_compute_at_arrival(
        trace=run.trace,
        request_by_id=request_by_id,
        pipeline=pipeline,
        profiler=profiler,
    )
    for snapshot in run.trace:
        violation = _split_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            prefill_compute=prefill_compute,
            pipeline=pipeline,
            profiler=profiler,
            budget_sla=budget_sla,
        )
        if violation is not None:
            return {
                "intensity": intensity,
                "feasible": False,
                "first_violation": violation,
            }
    return {"intensity": intensity, "feasible": True, "first_violation": None}


def _helix_probe(*, intensity, pipeline, workload, sla, root):
    run = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=scale_workload(workload, intensity),
        sla=sla,
        helix_root=root,
    )
    metrics = tuple(run.query_metrics.values())
    return {
        "intensity": intensity,
        "feasible": run.feasible,
        "first_violation_kind": run.first_violation_kind,
        "first_violation_request_id": run.first_violation_request_id,
        "max_aligned_ttft_s": max(m.aligned_ttft_s for m in metrics),
        "max_tpot_s": max(m.max_tpot_s for m in metrics),
    }


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=SEED,
        interval_offset=OFFSET,
    )
    workload = sample.requests
    base_rate = (len(workload) - 1) / (workload[-1].arrival_time_s - workload[0].arrival_time_s)
    budget_sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=FIXED_OVERHEAD_S)
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    count = round((STOP - START) / STEP)
    intensities = [START + i * STEP for i in range(count + 1)]

    result = {
        "design": {
            "purpose": "test budget-only fixed overhead after E38 trajectory mismatch",
            "seed": SEED,
            "offset": OFFSET,
            "grid": [START, STOP, STEP],
            "progression": "profiled progress with fixed_overhead_s=0",
            "budget_accounting": "original TTFT/TPOT budgets retain fixed_overhead_s=0.005 and full Prefill blocking-service debt",
            "network_redline_changed": False,
            "production_evaluator_changed": False,
            "base_arrival_rate_rps": base_rate,
        },
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        baseline_trials = [
            _budget_trial(
                intensity=x,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=budget_sla,
            )
            for x in intensities
        ]
        split_trials = [
            _split_trial(
                intensity=x,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                budget_sla=budget_sla,
            )
            for x in intensities
        ]
        baseline = _frontier(baseline_trials)
        split = _frontier(split_trials)
        for frontier in (baseline, split):
            if frontier["safe_intensity_lower_bound"] is not None:
                frontier["safe_arrival_rate_lower_bound_rps"] = frontier["safe_intensity_lower_bound"] * base_rate
            if frontier["unsafe_intensity_upper_bound"] is not None:
                frontier["unsafe_arrival_rate_upper_bound_rps"] = frontier["unsafe_intensity_upper_bound"] * base_rate

        probes = {}
        safe = split["safe_intensity_lower_bound"]
        unsafe = split["unsafe_intensity_upper_bound"]
        if safe is not None and unsafe is not None:
            probes = {
                "at_split_safe": _helix_probe(
                    intensity=safe,
                    pipeline=pipeline,
                    workload=workload,
                    sla=budget_sla,
                    root=root,
                ),
                "at_split_first_unsafe": _helix_probe(
                    intensity=unsafe,
                    pipeline=pipeline,
                    workload=workload,
                    sla=budget_sla,
                    root=root,
                ),
                "at_5pct_above_split_unsafe": _helix_probe(
                    intensity=unsafe * BUFFER_FACTOR,
                    pipeline=pipeline,
                    workload=workload,
                    sla=budget_sla,
                    root=root,
                ),
            }
        result["pipelines"][pipeline.id] = {
            "e31_original_frontier": baseline,
            "split_fixed_overhead_frontier": split,
            "helix_probes": probes,
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
