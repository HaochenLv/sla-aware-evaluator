from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, SLA
from .evaluator import evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .workload import build_helix_azure_conversation_workload


START_INTENSITY = 0.010
STOP_INTENSITY = 0.020
STEP_INTENSITY = 0.0001
HELIX_PROBES = (0.0132, 0.0134, 0.0136, 0.0138)


def _guard_violation(*, snapshot, request_by_id, pipeline, profiler, sla):
    prefill_ids = [
        rid for rid, phase in snapshot.request_phase.items() if phase == Phase.PREFILL.value
    ]
    decode_ids = [
        rid for rid, phase in snapshot.request_phase.items() if phase == Phase.DECODE.value
    ]
    if not prefill_ids or not decode_ids:
        return None

    n_prefill = len(prefill_ids)
    n_decode = len(decode_ids)
    prefill_debt_s = sum(
        profiler.prefill_time(request_by_id[rid], pipeline, n_prefill, n_decode)
        for rid in prefill_ids
    )
    for rid in decode_ids:
        decode_s = profiler.decode_time_per_token(
            request_by_id[rid],
            snapshot.request_context[rid],
            pipeline,
            n_prefill,
            n_decode,
        )
        required_s = decode_s + prefill_debt_s + sla.fixed_overhead_s
        if required_s > sla.tpot_s + 1e-9:
            return {
                "time_s": snapshot.time_s,
                "request_id": rid,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "decode_compute_s": decode_s,
                "prefill_debt_s": prefill_debt_s,
                "fixed_overhead_s": sla.fixed_overhead_s,
                "required_s": required_s,
                "capacity_s": sla.tpot_s,
                "active_prefill_ids": prefill_ids,
            }
    return None


def _violation_dict(violation):
    if violation is None:
        return None
    return {
        "time_s": violation.time_s,
        "kind": violation.kind.value,
        "object_id": violation.object_id,
        "required": violation.required,
        "capacity": violation.capacity,
        "request_id": violation.request_id,
        "num_prefill": violation.num_prefill,
        "num_decode": violation.num_decode,
    }


def _trial(*, intensity, pipeline, workload, profiler, sla):
    scaled = scale_workload(workload, intensity)
    request_by_id = {request.id: request for request in scaled}
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    guard = None
    for snapshot in run.trace:
        guard = _guard_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            pipeline=pipeline,
            profiler=profiler,
            sla=sla,
        )
        if guard is not None:
            break
    return {
        "intensity": intensity,
        "feasible": run.feasible and guard is None,
        "base_exact_feasible": run.feasible,
        "base_first_violation": _violation_dict(run.first_violation),
        "guard_violation": guard,
    }


def _frontier(trials):
    safe = [trial for trial in trials if trial["feasible"]]
    unsafe = [trial for trial in trials if not trial["feasible"]]
    first_unsafe = unsafe[0] if unsafe else None
    seen_unsafe = False
    monotonic = True
    for trial in trials:
        if not trial["feasible"]:
            seen_unsafe = True
        elif seen_unsafe:
            monotonic = False
            break
    return {
        "safe_intensity_lower_bound": safe[-1]["intensity"] if safe else None,
        "unsafe_intensity_upper_bound": first_unsafe["intensity"] if first_unsafe else None,
        "monotonic_on_grid": monotonic,
        "first_unsafe": first_unsafe,
    }


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    workload_sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = workload_sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    count = round((STOP_INTENSITY - START_INTENSITY) / STEP_INTENSITY)
    intensities = [START_INTENSITY + i * STEP_INTENSITY for i in range(count + 1)]

    result = {
        "design": {
            "decode_rule": "exact HELIX singleton rule",
            "prefill_guard": "sum(full profiled compute of all active Prefills)",
            "grid": [START_INTENSITY, STOP_INTENSITY, STEP_INTENSITY],
            "base_arrival_rate_rps": base_rate,
        },
        "pipelines": {},
    }
    for pipeline in build_helix_pipelines():
        trials = [
            _trial(
                intensity=intensity,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for intensity in intensities
        ]
        frontier = _frontier(trials)
        frontier["safe_arrival_rate_lower_bound_rps"] = (
            frontier["safe_intensity_lower_bound"] * base_rate
            if frontier["safe_intensity_lower_bound"] is not None
            else None
        )
        frontier["unsafe_arrival_rate_upper_bound_rps"] = (
            frontier["unsafe_intensity_upper_bound"] * base_rate
            if frontier["unsafe_intensity_upper_bound"] is not None
            else None
        )
        probes = [
            _trial(
                intensity=value,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for value in HELIX_PROBES
        ]
        result["pipelines"][pipeline.id] = {
            "frontier": frontier,
            "helix_probe_results": probes,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
