from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .prefill_debt_budget_ablation import _budget_trial
from .workload import build_helix_azure_conversation_workload

START = 0.006
STOP = 0.022
STEP = 0.0001
TPOT_VALUES_S = (0.130, 0.180)
WORKLOADS = {
    "seed7": {"seed": 7, "offset": 0},
    "seed19": {"seed": 19, "offset": 0},
}


def _grid():
    count = round((STOP - START) / STEP)
    return [round(START + i * STEP, 10) for i in range(count + 1)]


def _helix(*, intensity, pipeline, workload, sla, root):
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


def _prefix_edge(*, pipeline, workload, profiler, sla):
    last_safe = None
    first_unsafe = None
    for intensity in _grid():
        trial = _budget_trial(
            intensity=intensity,
            pipeline=pipeline,
            workload=workload,
            profiler=profiler,
            sla=sla,
        )
        if trial["feasible"]:
            if first_unsafe is None:
                last_safe = intensity
        else:
            first_unsafe = intensity
            break
    if last_safe is None or first_unsafe is None:
        raise RuntimeError(
            f"failed to find contiguous safe-prefix edge on [{START}, {STOP}] for TPOT={sla.tpot_s}"
        )
    return last_safe, first_unsafe


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)

    workload_label = os.environ.get("VALIDATION_LABEL", "seed7")
    pipeline_id = os.environ.get("PIPELINE_ID", "helix-slow-link-placement")
    if workload_label not in WORKLOADS:
        raise ValueError(f"unknown workload label {workload_label!r}")
    wc = WORKLOADS[workload_label]

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=wc["seed"],
        interval_offset=wc["offset"],
    )
    workload = sample.requests
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    pipeline = next(p for p in build_helix_pipelines() if p.id == pipeline_id)

    rows = []
    for tpot_s in TPOT_VALUES_S:
        sla = SLA(ttft_s=2.0, tpot_s=tpot_s, fixed_overhead_s=0.005)
        safe, unsafe = _prefix_edge(
            pipeline=pipeline,
            workload=workload,
            profiler=profiler,
            sla=sla,
        )
        helix_safe = _helix(
            intensity=safe,
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            root=root,
        )
        helix_unsafe = _helix(
            intensity=unsafe,
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            root=root,
        )
        if not helix_safe["feasible"]:
            classification = "candidate_optimism"
        elif helix_unsafe["feasible"]:
            classification = "candidate_conservative_at_prefix_edge"
        else:
            classification = "edge_overlap_within_grid_step"
        rows.append(
            {
                "tpot_s": tpot_s,
                "candidate_prefix_safe_intensity": safe,
                "candidate_first_unsafe_intensity": unsafe,
                "helix_at_candidate_safe": helix_safe,
                "helix_at_candidate_first_unsafe": helix_unsafe,
                "classification": classification,
            }
        )

    result = {
        "design": {
            "purpose": "orthogonal robustness test of unchanged E31 under moderate TPOT SLA changes",
            "workload_label": workload_label,
            "seed": wc["seed"],
            "interval_offset": wc["offset"],
            "pipeline": pipeline.id,
            "ttft_s": 2.0,
            "fixed_overhead_s": 0.005,
            "tpot_values_s": list(TPOT_VALUES_S),
            "candidate_scan": [START, STOP, STEP],
            "edge_definition": "largest contiguous safe prefix from scan start, then first unsafe point; no HELIX monotonicity assumed",
            "candidate_changed": False,
            "production_evaluator_changed": False,
            "network_redline_changed": False,
        },
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "candidate_optimism_cases": sum(r["classification"] == "candidate_optimism" for r in rows),
            "candidate_conservative_cases": sum(r["classification"] == "candidate_conservative_at_prefix_edge" for r in rows),
            "edge_overlap_cases": sum(r["classification"] == "edge_overlap_within_grid_step" for r in rows),
            "max_helix_tpot_s_at_candidate_safe": max(r["helix_at_candidate_safe"]["max_tpot_s"] for r in rows),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
