from __future__ import annotations

import json
import os

from .domain import RequestSpec, SLA
from .helix_demo import build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference


def main() -> None:
    helix_root = os.environ.get("HELIX_ROOT")
    if not helix_root:
        raise RuntimeError("set HELIX_ROOT to the pinned Helix-ASPLOS25 checkout")

    workload = (
        RequestSpec("smoke-0", 0.0, 256, 8),
        RequestSpec("smoke-1", 0.35, 384, 8),
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    result = {}
    for pipeline in build_helix_pipelines():
        run = evaluate_helix_fixed_reference(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            helix_root=helix_root,
        )
        result[pipeline.id] = {
            "feasible": run.feasible,
            "finished_requests": run.finished_requests,
            "final_time_s": run.final_time_s,
            "first_violation_kind": run.first_violation_kind,
            "first_violation_request_id": run.first_violation_request_id,
            "max_aligned_ttft_s": max(
                metric.aligned_ttft_s for metric in run.query_metrics.values()
            ),
            "max_true_first_token_ttft_s": max(
                metric.true_first_token_ttft_s
                for metric in run.query_metrics.values()
            ),
            "max_tpot_s": max(
                metric.max_tpot_s for metric in run.query_metrics.values()
            ),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
