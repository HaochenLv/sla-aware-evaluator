from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .workload import build_helix_azure_conversation_workload


CASES = {
    "seed11_fast": {
        "seed": 11,
        "offset": 0,
        "pipeline_id": "helix-fast-link-placement",
        "candidate_unsafe": 0.0156,
    },
    "seed19_slow": {
        "seed": 19,
        "offset": 0,
        "pipeline_id": "helix-slow-link-placement",
        "candidate_unsafe": 0.0153,
    },
}
LADDER_FACTORS = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30)
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
    case_name = os.environ.get("REFINE_CASE", "seed11_fast")
    if case_name not in CASES:
        raise RuntimeError(f"unknown REFINE_CASE: {case_name}")
    case = CASES[case_name]

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=case["seed"],
        interval_offset=case["offset"],
    )
    workload = sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    pipeline = next(
        item for item in build_helix_pipelines() if item.id == case["pipeline_id"]
    )

    ladder = []
    candidate_unsafe = case["candidate_unsafe"]
    for factor in LADDER_FACTORS:
        ladder.append(
            _probe(
                intensity=candidate_unsafe * factor,
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                root=root,
            )
        )

    seen_unsafe = False
    sampled_monotonic = True
    for probe in ladder:
        if not probe["feasible"]:
            seen_unsafe = True
        elif seen_unsafe:
            sampled_monotonic = False

    first_unsafe_index = next(
        (i for i, probe in enumerate(ladder) if not probe["feasible"]),
        None,
    )
    refinement = None
    if first_unsafe_index is not None and first_unsafe_index > 0:
        refinement = _refine(
            low_safe=ladder[first_unsafe_index - 1]["intensity"],
            high_unsafe=ladder[first_unsafe_index]["intensity"],
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
        refinement["gap_above_candidate_unsafe_ratio"] = (
            refinement["safe_intensity_lower_bound"] / candidate_unsafe - 1.0
        )
    result = {
        "design": {
            "purpose": "quantify the largest observed E31 conservatism among E34 cases that remained HELIX-safe at +5%",
            "case": case_name,
            "seed": case["seed"],
            "offset": case["offset"],
            "pipeline_id": pipeline.id,
            "candidate_unsafe": candidate_unsafe,
            "ladder_factors": list(LADDER_FACTORS),
            "refine_tolerance": REFINE_TOL,
            "evaluator_semantics_changed": False,
        },
        "base_arrival_rate_rps": base_rate,
        "ladder_probes": ladder,
        "sampled_monotonic": sampled_monotonic,
        "refinement": refinement,
        "resolved_within_30pct": first_unsafe_index is not None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
