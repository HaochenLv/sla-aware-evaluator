from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, RequestRuntime, SLA
from .evaluator import _network_bytes_by_link, evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .exact_singleton_prefill_guard_demo import _frontier, _trial as historical_guard_only_trial
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .workload import build_helix_azure_conversation_workload


START_INTENSITY = 0.010
STOP_INTENSITY = 0.020
STEP_INTENSITY = 0.0001
BUFFER_FACTOR = 1.05

# E22 midpoint inferred independently from isolated aligned-TTFT bandwidth
# thresholds on the same 8-stage HELIX configuration. This is intentionally an
# experiment-local input, not a production hard-coded runtime constant.
E22_BLOCKING_OVERHEAD_S_PER_TOKEN = 59.37720874470879e-6


def _decode_link_contribution(*, request, pipeline, remaining_s: float):
    runtime = RequestRuntime(request, Phase.DECODE)
    demand = _network_bytes_by_link(runtime, pipeline)
    normalized_cost = 0.0
    for link_id, data_bytes in demand.items():
        if data_bytes <= 0:
            continue
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        if capacity <= 0:
            return None
        normalized_cost += data_bytes / capacity
    if normalized_cost <= 0:
        return {link_id: 0.0 for link_id in pipeline.links}
    result = {link_id: 0.0 for link_id in pipeline.links}
    for link_id, data_bytes in demand.items():
        if data_bytes <= 0:
            continue
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        weight = (data_bytes / capacity) / normalized_cost
        result[link_id] = data_bytes / (weight * remaining_s)
    return result


def _prefill_blocking_service_s(*, request, pipeline, profiler, n_prefill, n_decode):
    compute_s = profiler.prefill_time(request, pipeline, n_prefill, n_decode)
    overhead_s = E22_BLOCKING_OVERHEAD_S_PER_TOKEN * request.input_tokens
    return compute_s + overhead_s


def _budget_consistent_violation(*, snapshot, request_by_id, pipeline, profiler, sla):
    prefill_ids = [
        rid for rid, phase in snapshot.request_phase.items() if phase == Phase.PREFILL.value
    ]
    decode_ids = [
        rid for rid, phase in snapshot.request_phase.items() if phase == Phase.DECODE.value
    ]
    if not decode_ids or not prefill_ids:
        return None

    n_prefill = len(prefill_ids)
    n_decode = len(decode_ids)
    prefill_debt_s = sum(
        _prefill_blocking_service_s(
            request=request_by_id[rid],
            pipeline=pipeline,
            profiler=profiler,
            n_prefill=n_prefill,
            n_decode=n_decode,
        )
        for rid in prefill_ids
    )

    link_required = dict(snapshot.link_required_bytes_per_s)
    tightest_decode = None
    for rid in decode_ids:
        request = request_by_id[rid]
        context = snapshot.request_context[rid]
        decode_s = profiler.decode_time_per_token(
            request, context, pipeline, n_prefill, n_decode
        )
        old_remaining_s = (
            sla.tpot_s - decode_s - sla.queue_overhead_s - sla.fixed_overhead_s
        )
        new_remaining_s = old_remaining_s - prefill_debt_s
        record = {
            "request_id": rid,
            "decode_compute_s": decode_s,
            "prefill_blocking_service_debt_s": prefill_debt_s,
            "old_remaining_network_budget_s": old_remaining_s,
            "new_remaining_network_budget_s": new_remaining_s,
        }
        if tightest_decode is None or new_remaining_s < tightest_decode["new_remaining_network_budget_s"]:
            tightest_decode = record
        if new_remaining_s <= 1e-9:
            return {
                "time_s": snapshot.time_s,
                "kind": "sla_time",
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                **record,
                "required_compute_side_s": decode_s
                + prefill_debt_s
                + sla.queue_overhead_s
                + sla.fixed_overhead_s,
                "capacity_s": sla.tpot_s,
            }

        old_contrib = _decode_link_contribution(
            request=request, pipeline=pipeline, remaining_s=old_remaining_s
        )
        new_contrib = _decode_link_contribution(
            request=request, pipeline=pipeline, remaining_s=new_remaining_s
        )
        if old_contrib is None or new_contrib is None:
            continue
        for link_id in link_required:
            link_required[link_id] += new_contrib[link_id] - old_contrib[link_id]

    for link_id, required in link_required.items():
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        if required > capacity + 1e-9:
            return {
                "time_s": snapshot.time_s,
                "kind": "network",
                "link_id": link_id,
                "required_bytes_per_s": required,
                "capacity_bytes_per_s": capacity,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "prefill_blocking_service_debt_s": prefill_debt_s,
                "tightest_decode": tightest_decode,
            }
    return None


def _budget_trial(*, intensity, pipeline, workload, profiler, sla):
    scaled = scale_workload(workload, intensity)
    request_by_id = {request.id: request for request in scaled}
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    if not run.feasible:
        return {
            "intensity": intensity,
            "feasible": False,
            "base_exact_feasible": False,
            "first_violation": {
                "kind": run.first_violation.kind.value if run.first_violation else None,
                "time_s": run.first_violation.time_s if run.first_violation else None,
            },
        }

    for snapshot in run.trace:
        violation = _budget_consistent_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            pipeline=pipeline,
            profiler=profiler,
            sla=sla,
        )
        if violation is not None:
            return {
                "intensity": intensity,
                "feasible": False,
                "base_exact_feasible": True,
                "first_violation": violation,
            }
    return {
        "intensity": intensity,
        "feasible": True,
        "base_exact_feasible": True,
        "first_violation": None,
    }


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
        "max_aligned_ttft_s": max(item.aligned_ttft_s for item in metrics),
        "max_tpot_s": max(item.max_tpot_s for item in metrics),
    }


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    count = round((STOP_INTENSITY - START_INTENSITY) / STEP_INTENSITY)
    intensities = [START_INTENSITY + i * STEP_INTENSITY for i in range(count + 1)]

    result = {
        "design": {
            "purpose": "test budget-consistent Prefill blocking-service debt inside Decode Delta while preserving the unchanged Conservative trajectory",
            "decode_rule": "exact HELIX singleton Decode profile alignment",
            "candidate_debt": "sum(profiler Prefill compute + E22 independently calibrated blocking overhead) for all active Prefills",
            "e22_blocking_overhead_s_per_token": E22_BLOCKING_OVERHEAD_S_PER_TOKEN,
            "budget_formula": "Delta_D=tau_D-T_decode-I_blocking-T_queue-T_fix",
            "network_redline_changed": False,
            "state_progression_changed": False,
            "production_evaluator_changed": False,
            "historical_guard_baseline_warning": "guard-only comparison uses the older compute-only E15 candidate and is provenance-only",
            "grid": [START_INTENSITY, STOP_INTENSITY, STEP_INTENSITY],
            "base_arrival_rate_rps": base_rate,
        },
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        historical_guard_trials = [
            historical_guard_only_trial(
                intensity=value,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for value in intensities
        ]
        budget_trials = [
            _budget_trial(
                intensity=value,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for value in intensities
        ]
        historical_guard_frontier = _frontier(historical_guard_trials)
        budget_frontier = _frontier(budget_trials)
        for frontier in (historical_guard_frontier, budget_frontier):
            if frontier["safe_intensity_lower_bound"] is not None:
                frontier["safe_arrival_rate_lower_bound_rps"] = (
                    frontier["safe_intensity_lower_bound"] * base_rate
                )
            if frontier["unsafe_intensity_upper_bound"] is not None:
                frontier["unsafe_arrival_rate_upper_bound_rps"] = (
                    frontier["unsafe_intensity_upper_bound"] * base_rate
                )

        safe = budget_frontier["safe_intensity_lower_bound"]
        unsafe = budget_frontier["unsafe_intensity_upper_bound"]
        helix_probes = {}
        if safe is not None and unsafe is not None:
            helix_probes = {
                "at_budget_safe": _helix_probe(
                    intensity=safe, pipeline=pipeline, workload=workload, sla=sla, root=root
                ),
                "at_budget_unsafe": _helix_probe(
                    intensity=unsafe, pipeline=pipeline, workload=workload, sla=sla, root=root
                ),
                "at_5pct_above_budget_unsafe": _helix_probe(
                    intensity=unsafe * BUFFER_FACTOR,
                    pipeline=pipeline,
                    workload=workload,
                    sla=sla,
                    root=root,
                ),
            }

        result["pipelines"][pipeline.id] = {
            "historical_compute_guard_only_frontier": historical_guard_frontier,
            "blocking_service_budget_frontier": budget_frontier,
            "helix_probes": helix_probes,
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
