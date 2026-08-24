from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, SLA
from .evaluator import evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .prefill_debt_budget_ablation import _budget_consistent_violation, _budget_trial
from .workload import build_helix_azure_conversation_workload


CASE_CONFIG = {
    "seed3": {"seed": 3, "offset": 0, "safe": 0.0152, "unsafe": 0.0153},
    "seed11": {"seed": 11, "offset": 0, "safe": 0.0155, "unsafe": 0.0156},
    "seed19": {"seed": 19, "offset": 0, "safe": 0.0152, "unsafe": 0.0153},
    "seed7_offset200": {"seed": 7, "offset": 200, "safe": 0.0087, "unsafe": 0.0088},
}

# Moderate local phase perturbation: less than one isolated Decode token interval
# (~112 ms in the pinned HELIX configuration), and coarser than E40's 10 ms grid.
JITTERS_S = tuple(round(-0.08 + 0.02 * i, 10) for i in range(9))


def _helix(*, pipeline, workload, sla, root):
    run = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=workload,
        sla=sla,
        helix_root=root,
    )
    metrics = tuple(run.query_metrics.values())
    return {
        "feasible": run.feasible,
        "first_violation_kind": run.first_violation_kind,
        "first_violation_request_id": run.first_violation_request_id,
        "max_aligned_ttft_s": max(m.aligned_ttft_s for m in metrics),
        "max_tpot_s": max(m.max_tpot_s for m in metrics),
    }


def _detect_critical_prefill(*, pipeline, workload, unsafe_intensity, profiler, sla):
    scaled = scale_workload(workload, unsafe_intensity)
    request_by_id = {request.id: request for request in scaled}
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    if not run.feasible:
        raise RuntimeError("base evaluator became infeasible before E31 debt audit")

    previous_prefills: set[str] = set()
    for snapshot in run.trace:
        current_prefills = {
            rid
            for rid, phase in snapshot.request_phase.items()
            if phase == Phase.PREFILL.value
        }
        violation = _budget_consistent_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            pipeline=pipeline,
            profiler=profiler,
            sla=sla,
        )
        if violation is None:
            previous_prefills = current_prefills
            continue

        new_prefills = sorted(current_prefills - previous_prefills)
        if new_prefills:
            critical_id = max(
                new_prefills,
                key=lambda rid: request_by_id[rid].arrival_time_s,
            )
            selection = "new_prefill_at_first_e31_violation"
        elif current_prefills:
            critical_id = max(
                current_prefills,
                key=lambda rid: request_by_id[rid].arrival_time_s,
            )
            selection = "latest_active_prefill_at_first_e31_violation"
        else:
            raise RuntimeError("E31 violation found without active Prefill")

        critical = request_by_id[critical_id]
        return {
            "request_id": critical_id,
            "input_tokens": critical.input_tokens,
            "output_tokens": critical.output_tokens,
            "arrival_time_s_at_unsafe": critical.arrival_time_s,
            "violation_time_s": snapshot.time_s,
            "event_types": list(snapshot.event_types),
            "selection": selection,
            "victim_decode_request_id": violation.get("request_id"),
            "num_prefill": violation.get("num_prefill"),
            "num_decode": violation.get("num_decode"),
        }
    raise RuntimeError("no E31 blocking-service violation found at configured unsafe intensity")


def _jitter_request(workload, request_id: str, jitter_s: float):
    shifted = []
    found = False
    for request in workload:
        if request.id == request_id:
            shifted.append(replace(request, arrival_time_s=request.arrival_time_s + jitter_s))
            found = True
        else:
            shifted.append(request)
    if not found:
        raise RuntimeError(f"critical request {request_id!r} not found")
    return tuple(shifted)


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)

    label = os.environ.get("VALIDATION_LABEL", "seed3")
    pipeline_id = os.environ.get("PIPELINE_ID", "helix-slow-link-placement")
    if label not in CASE_CONFIG:
        raise ValueError(f"unknown validation label {label!r}")
    config = CASE_CONFIG[label]

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=config["seed"],
        interval_offset=config["offset"],
    )
    workload = sample.requests
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    pipeline = next(p for p in build_helix_pipelines() if p.id == pipeline_id)

    critical = _detect_critical_prefill(
        pipeline=pipeline,
        workload=workload,
        unsafe_intensity=config["unsafe"],
        profiler=profiler,
        sla=sla,
    )

    safe_scaled = scale_workload(workload, config["safe"])
    original_critical = next(r for r in safe_scaled if r.id == critical["request_id"])

    rows = []
    for jitter_s in JITTERS_S:
        stressed = _jitter_request(safe_scaled, critical["request_id"], jitter_s)
        candidate = _budget_trial(
            intensity=1.0,
            pipeline=pipeline,
            workload=stressed,
            profiler=profiler,
            sla=sla,
        )
        helix = _helix(
            pipeline=pipeline,
            workload=stressed,
            sla=sla,
            root=root,
        )
        if candidate["feasible"] and not helix["feasible"]:
            classification = "candidate_optimism"
        elif not candidate["feasible"] and helix["feasible"]:
            classification = "candidate_conservative"
        elif candidate["feasible"] and helix["feasible"]:
            classification = "both_safe"
        else:
            classification = "both_unsafe"
        rows.append(
            {
                "jitter_s": jitter_s,
                "critical_arrival_s": original_critical.arrival_time_s + jitter_s,
                "candidate": {
                    "feasible": candidate["feasible"],
                    "first_violation": candidate["first_violation"],
                },
                "helix": helix,
                "classification": classification,
            }
        )

    optimism = [row for row in rows if row["classification"] == "candidate_optimism"]
    result = {
        "design": {
            "purpose": "moderate cross-workload phase-jitter falsification of unchanged E31 candidate",
            "workload_label": label,
            "seed": config["seed"],
            "interval_offset": config["offset"],
            "pipeline": pipeline.id,
            "candidate_safe_intensity": config["safe"],
            "candidate_first_unsafe_intensity": config["unsafe"],
            "jitter_grid_s": list(JITTERS_S),
            "jitter_window_note": "plus/minus 80 ms, intentionally below one isolated Decode-token interval in this pinned configuration",
            "candidate_changed": False,
            "production_evaluator_changed": False,
            "network_redline_changed": False,
        },
        "critical_prefill": critical,
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "candidate_optimism_cases": len(optimism),
            "optimism_jitters_s": [row["jitter_s"] for row in optimism],
            "helix_unsafe_cases": sum(not row["helix"]["feasible"] for row in rows),
            "max_helix_tpot_s": max(row["helix"]["max_tpot_s"] for row in rows),
            "classification_counts": {
                name: sum(row["classification"] == name for row in rows)
                for name in (
                    "both_safe",
                    "candidate_conservative",
                    "both_unsafe",
                    "candidate_optimism",
                )
            },
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
