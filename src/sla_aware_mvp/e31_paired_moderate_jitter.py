from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .e31_moderate_phase_stress import CASE_CONFIG, _detect_critical_prefill, _helix
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .prefill_debt_budget_ablation import _budget_trial
from .workload import build_helix_azure_conversation_workload

# Realistic second-order timing noise: each of two nearby arrivals moves by at most
# 40 ms, well below the ~112 ms isolated Decode-token interval in the pinned setup.
JITTERS_S = (-0.04, 0.0, 0.04)


def _paired_jitter(workload, first_id: str, first_jitter_s: float, second_id: str, second_jitter_s: float):
    shifted = []
    found = set()
    for request in workload:
        if request.id == first_id:
            shifted.append(replace(request, arrival_time_s=request.arrival_time_s + first_jitter_s))
            found.add(first_id)
        elif request.id == second_id:
            shifted.append(replace(request, arrival_time_s=request.arrival_time_s + second_jitter_s))
            found.add(second_id)
        else:
            shifted.append(request)
    if found != {first_id, second_id}:
        raise RuntimeError(f"paired jitter requests not found: expected {first_id!r}, {second_id!r}, got {sorted(found)!r}")
    return tuple(shifted)


def _classify(candidate, helix):
    if candidate["feasible"] and not helix["feasible"]:
        return "candidate_optimism"
    if not candidate["feasible"] and helix["feasible"]:
        return "candidate_conservative"
    if candidate["feasible"] and helix["feasible"]:
        return "both_safe"
    return "both_unsafe"


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
    critical_request = next(r for r in safe_scaled if r.id == critical["request_id"])
    neighbor = min(
        (r for r in safe_scaled if r.id != critical_request.id),
        key=lambda r: (abs(r.arrival_time_s - critical_request.arrival_time_s), r.arrival_time_s, r.id),
    )

    rows = []
    for critical_jitter_s in JITTERS_S:
        for neighbor_jitter_s in JITTERS_S:
            stressed = _paired_jitter(
                safe_scaled,
                critical_request.id,
                critical_jitter_s,
                neighbor.id,
                neighbor_jitter_s,
            )
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
            rows.append(
                {
                    "critical_jitter_s": critical_jitter_s,
                    "neighbor_jitter_s": neighbor_jitter_s,
                    "critical_arrival_s": critical_request.arrival_time_s + critical_jitter_s,
                    "neighbor_arrival_s": neighbor.arrival_time_s + neighbor_jitter_s,
                    "candidate": {
                        "feasible": candidate["feasible"],
                        "first_violation": candidate["first_violation"],
                    },
                    "helix": helix,
                    "classification": _classify(candidate, helix),
                }
            )

    optimism = [row for row in rows if row["classification"] == "candidate_optimism"]
    result = {
        "design": {
            "purpose": "paired moderate-arrival falsification of unchanged E31 candidate",
            "workload_label": label,
            "seed": config["seed"],
            "interval_offset": config["offset"],
            "pipeline": pipeline.id,
            "candidate_safe_intensity": config["safe"],
            "candidate_first_unsafe_intensity": config["unsafe"],
            "jitter_grid_s": list(JITTERS_S),
            "jitter_note": "two nearby arrivals independently perturbed by at most 40 ms",
            "candidate_changed": False,
            "production_evaluator_changed": False,
            "network_redline_changed": False,
        },
        "critical_prefill": critical,
        "neighbor_request": {
            "request_id": neighbor.id,
            "input_tokens": neighbor.input_tokens,
            "output_tokens": neighbor.output_tokens,
            "arrival_time_s": neighbor.arrival_time_s,
            "arrival_gap_from_critical_s": neighbor.arrival_time_s - critical_request.arrival_time_s,
        },
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "candidate_optimism_cases": len(optimism),
            "optimism_pairs_s": [
                [row["critical_jitter_s"], row["neighbor_jitter_s"]] for row in optimism
            ],
            "helix_unsafe_cases": sum(not row["helix"]["feasible"] for row in rows),
            "max_helix_tpot_s": max(row["helix"]["max_tpot_s"] for row in rows),
            "classification_counts": {
                name: sum(row["classification"] == name for row in rows)
                for name in ("both_safe", "candidate_conservative", "both_unsafe", "candidate_optimism")
            },
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
