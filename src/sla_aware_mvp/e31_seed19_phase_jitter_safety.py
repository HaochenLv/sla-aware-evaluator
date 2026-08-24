from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .prefill_debt_budget_ablation import _budget_trial
from .workload import build_helix_azure_conversation_workload


SEED = 19
BASE_INTENSITY = 0.0152
INTERFERER_ID = "azure-00010"
JITTERS_S = tuple(round(-0.12 + 0.01 * i, 10) for i in range(25))


def _jitter_workload(workload, jitter_s: float):
    rows = []
    for request in workload:
        if request.id == INTERFERER_ID:
            rows.append(replace(request, arrival_time_s=request.arrival_time_s + jitter_s))
        else:
            rows.append(request)
    return tuple(rows)


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
        interval_offset=0,
    )
    scaled = scale_workload(sample.requests, BASE_INTENSITY)
    original_interferer = next(r for r in scaled if r.id == INTERFERER_ID)
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    pipeline = next(
        p for p in build_helix_pipelines() if p.id == "helix-slow-link-placement"
    )

    rows = []
    for jitter_s in JITTERS_S:
        workload = _jitter_workload(scaled, jitter_s)
        candidate = _budget_trial(
            intensity=1.0,
            pipeline=pipeline,
            workload=workload,
            profiler=profiler,
            sla=sla,
        )
        helix = _helix(
            pipeline=pipeline,
            workload=workload,
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
                "interferer_arrival_s": original_interferer.arrival_time_s + jitter_s,
                "candidate": {
                    "feasible": candidate["feasible"],
                    "first_violation": candidate["first_violation"],
                },
                "helix": helix,
                "classification": classification,
            }
        )

    optimism = [r for r in rows if r["classification"] == "candidate_optimism"]
    helix_unsafe = [r for r in rows if not r["helix"]["feasible"]]
    candidate_safe = [r for r in rows if r["candidate"]["feasible"]]
    result = {
        "design": {
            "purpose": "phase-jitter falsification of original E31 candidate at seed19 Slow candidate-safe intensity",
            "seed": SEED,
            "pipeline": pipeline.id,
            "base_intensity": BASE_INTENSITY,
            "jittered_request": INTERFERER_ID,
            "jitter_grid_s": list(JITTERS_S),
            "candidate_changed": False,
            "production_evaluator_changed": False,
            "network_redline_changed": False,
        },
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "candidate_safe_cases": len(candidate_safe),
            "helix_unsafe_cases": len(helix_unsafe),
            "candidate_optimism_cases": len(optimism),
            "optimism_jitters_s": [r["jitter_s"] for r in optimism],
            "max_helix_tpot_s": max(r["helix"]["max_tpot_s"] for r in rows),
            "classification_counts": {
                name: sum(r["classification"] == name for r in rows)
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
