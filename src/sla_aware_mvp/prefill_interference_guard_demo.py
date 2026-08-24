from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, SLA
from .evaluator import evaluate
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .workload import build_helix_azure_conversation_workload


START_INTENSITY = 0.004
STOP_INTENSITY = 0.030
STEP_INTENSITY = 0.0002


def _active_guard_violation(*, snapshot, request_by_id, pipeline, profiler, sla):
    prefill_ids = [
        request_id
        for request_id, phase in snapshot.request_phase.items()
        if phase == Phase.PREFILL.value
    ]
    decode_ids = [
        request_id
        for request_id, phase in snapshot.request_phase.items()
        if phase == Phase.DECODE.value
    ]
    if not prefill_ids or not decode_ids:
        return None

    n_prefill = len(prefill_ids)
    n_decode = len(decode_ids)
    prefill_debt_s = sum(
        profiler.prefill_time(
            request_by_id[request_id], pipeline, n_prefill, n_decode
        )
        for request_id in prefill_ids
    )
    for request_id in decode_ids:
        request = request_by_id[request_id]
        context = snapshot.request_context[request_id]
        base_decode_s = profiler.decode_time_per_token(
            request, context, pipeline, n_prefill, n_decode
        )
        required_s = base_decode_s + prefill_debt_s + sla.fixed_overhead_s
        if required_s > sla.tpot_s + 1e-9:
            return {
                "time_s": snapshot.time_s,
                "request_id": request_id,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "base_decode_s": base_decode_s,
                "prefill_interference_debt_s": prefill_debt_s,
                "fixed_overhead_s": sla.fixed_overhead_s,
                "required_s": required_s,
                "tpot_s": sla.tpot_s,
                "active_prefill_ids": prefill_ids,
            }
    return None


def _run_guarded(*, intensity, pipeline, workload, profiler, sla):
    scaled = scale_workload(workload, intensity)
    request_by_id = {request.id: request for request in scaled}
    config = EvaluatorConfig(record_trace=True)
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=config,
        profiler=profiler,
    )
    guard_violation = None
    for snapshot in run.trace:
        guard_violation = _active_guard_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            pipeline=pipeline,
            profiler=profiler,
            sla=sla,
        )
        if guard_violation is not None:
            break
    feasible = run.feasible and guard_violation is None
    return {
        "intensity": intensity,
        "feasible": feasible,
        "baseline_feasible": run.feasible,
        "baseline_violation": (
            run.first_violation.kind.value if run.first_violation is not None else None
        ),
        "guard_violation": guard_violation,
    }


def _scan_pipeline(*, pipeline, workload, profiler, sla):
    count = round((STOP_INTENSITY - START_INTENSITY) / STEP_INTENSITY)
    intensities = [START_INTENSITY + index * STEP_INTENSITY for index in range(count + 1)]
    trials = [
        _run_guarded(
            intensity=intensity,
            pipeline=pipeline,
            workload=workload,
            profiler=profiler,
            sla=sla,
        )
        for intensity in intensities
    ]
    safe = [trial for trial in trials if trial["feasible"]]
    unsafe = [trial for trial in trials if not trial["feasible"]]
    seen_unsafe = False
    monotonic = True
    for trial in trials:
        if not trial["feasible"]:
            seen_unsafe = True
        elif seen_unsafe:
            monotonic = False
            break
    first_unsafe = unsafe[0] if unsafe else None
    return {
        "safe_intensity_lower_bound": safe[-1]["intensity"] if safe else None,
        "unsafe_intensity_upper_bound": first_unsafe["intensity"] if first_unsafe else None,
        "monotonic_on_grid": monotonic,
        "first_unsafe": first_unsafe,
        "trials": trials,
    }


def main() -> None:
    root = artifact_root()
    profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
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
    result = {
        "design": {
            "guard": "base Decode profile + sum(full profiled compute of all active Prefills) + fixed overhead <= TPOT",
            "trajectory": "unchanged baseline Conservative trace; guard is observational only",
            "intensity_grid": [START_INTENSITY, STOP_INTENSITY, STEP_INTENSITY],
            "requests": len(workload),
            "base_arrival_rate_rps": base_rate,
        },
        "pipelines": {},
    }
    for pipeline in build_helix_pipelines():
        summary = _scan_pipeline(
            pipeline=pipeline,
            workload=workload,
            profiler=profiler,
            sla=sla,
        )
        summary["safe_arrival_rate_lower_bound_rps"] = (
            summary["safe_intensity_lower_bound"] * base_rate
            if summary["safe_intensity_lower_bound"] is not None
            else None
        )
        summary["unsafe_arrival_rate_upper_bound_rps"] = (
            summary["unsafe_intensity_upper_bound"] * base_rate
            if summary["unsafe_intensity_upper_bound"] is not None
            else None
        )
        result["pipelines"][pipeline.id] = summary
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
