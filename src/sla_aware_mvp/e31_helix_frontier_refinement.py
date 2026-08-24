from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .workload import build_helix_azure_conversation_workload

GRID_LOW = 0.01320
GRID_HIGH = 0.01390
GRID_STEP = 0.00005
REFINE_TOL = 0.000005


def _probe(*, intensity, pipeline, workload, sla, root):
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


def _refine(*, low_safe, high_unsafe, pipeline, workload, sla, root):
    probes = []
    low = low_safe
    high = high_unsafe
    while high - low > REFINE_TOL:
        mid = (low + high) / 2.0
        probe = _probe(
            intensity=mid,
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            root=root,
        )
        probes.append(probe)
        if probe["feasible"]:
            low = mid
        else:
            high = mid
    return {
        "safe_intensity_lower_bound": low,
        "unsafe_intensity_upper_bound": high,
        "width": high - low,
        "refinement_probes": probes,
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
        seed=7,
    )
    workload = sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)

    count = round((GRID_HIGH - GRID_LOW) / GRID_STEP)
    grid = [GRID_LOW + i * GRID_STEP for i in range(count + 1)]

    result = {
        "design": {
            "purpose": "refine the pinned HELIX frontier above the E31 candidate first-unsafe point",
            "candidate_safe_intensity": 0.0131,
            "candidate_first_unsafe_intensity": 0.0132,
            "grid": [GRID_LOW, GRID_HIGH, GRID_STEP],
            "refine_tolerance": REFINE_TOL,
            "base_arrival_rate_rps": base_rate,
            "evaluator_semantics_changed": False,
        },
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        grid_probes = [
            _probe(
                intensity=value,
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                root=root,
            )
            for value in grid
        ]
        monotonic = True
        seen_unsafe = False
        for probe in grid_probes:
            if not probe["feasible"]:
                seen_unsafe = True
            elif seen_unsafe:
                monotonic = False
                break

        first_unsafe_index = next(
            (i for i, probe in enumerate(grid_probes) if not probe["feasible"]),
            None,
        )
        refinement = None
        if first_unsafe_index is not None and first_unsafe_index > 0:
            low_safe = grid_probes[first_unsafe_index - 1]["intensity"]
            high_unsafe = grid_probes[first_unsafe_index]["intensity"]
            refinement = _refine(
                low_safe=low_safe,
                high_unsafe=high_unsafe,
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                root=root,
            )
            refinement["safe_arrival_rate_lower_bound_rps"] = (
                refinement["safe_intensity_lower_bound"] * base_rate
            )
            refinement["unsafe_arrival_rate_upper_bound_rps"] = (
                refinement["unsafe_intensity_upper_bound"] * base_rate
            )
            refinement["candidate_safe_gap_ratio"] = (
                refinement["safe_intensity_lower_bound"] / 0.0131 - 1.0
            )

        result["pipelines"][pipeline.id] = {
            "sampled_monotonic": monotonic,
            "grid_probes": grid_probes,
            "refinement": refinement,
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
