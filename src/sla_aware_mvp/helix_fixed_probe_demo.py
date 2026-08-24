from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .workload import build_helix_azure_conversation_workload


def main() -> None:
    helix_root_env = os.environ.get("HELIX_ROOT")
    pipeline_id = os.environ.get("PROBE_PIPELINE")
    intensity_text = os.environ.get("PROBE_INTENSITY")
    if not helix_root_env or not pipeline_id or not intensity_text:
        raise RuntimeError("set HELIX_ROOT, PROBE_PIPELINE, and PROBE_INTENSITY")
    intensity = float(intensity_text)
    helix_root = Path(helix_root_env)

    sample = build_helix_azure_conversation_workload(
        helix_root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    pipelines = {pipeline.id: pipeline for pipeline in build_helix_pipelines()}
    if pipeline_id not in pipelines:
        raise ValueError(f"unknown pipeline {pipeline_id!r}")
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    run = evaluate_helix_fixed_reference(
        pipeline=pipelines[pipeline_id],
        workload=scale_workload(workload, intensity),
        sla=sla,
        helix_root=helix_root,
    )
    max_aligned_ttft = max(
        (metric.aligned_ttft_s for metric in run.query_metrics.values()), default=None
    )
    max_tpot = max(
        (metric.max_tpot_s for metric in run.query_metrics.values()), default=None
    )
    print(
        json.dumps(
            {
                "pipeline_id": pipeline_id,
                "intensity": intensity,
                "arrival_rate_rps": intensity * base_rate,
                "feasible": run.feasible,
                "finished_requests": run.finished_requests,
                "total_requests": run.total_requests,
                "final_time_s": run.final_time_s,
                "max_aligned_ttft_s": max_aligned_ttft,
                "max_tpot_s": max_tpot,
                "first_violation_kind": run.first_violation_kind,
                "first_violation_request_id": run.first_violation_request_id,
                "first_violation_observed_s": run.first_violation_observed_s,
                "first_violation_limit_s": run.first_violation_limit_s,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
