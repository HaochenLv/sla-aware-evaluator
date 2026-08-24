from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import find_capacity
from .domain import EvaluatorConfig, SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .workload import build_helix_azure_conversation_workload


def main() -> None:
    helix_root_env = os.environ.get("HELIX_ROOT")
    if not helix_root_env:
        raise RuntimeError("set HELIX_ROOT")
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
    profiler = HelixA100Llama2Profiler.from_artifact(helix_root, commit=HELIX_COMMIT)
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    result = {
        "workload_requests": len(workload),
        "workload_output_tokens": sum(request.output_tokens for request in workload),
        "base_arrival_rate_rps": base_rate,
        "pipelines": {},
    }
    for pipeline in build_helix_pipelines():
        capacity = find_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            config=EvaluatorConfig(decode_block_size=16),
            profiler=profiler,
            initial_intensity=0.25,
            tolerance=0.02,
            max_intensity=16.0,
            verification_grid_points=9,
        )
        result["pipelines"][pipeline.id] = {
            "safe_intensity": capacity.safe_intensity,
            "unsafe_intensity": capacity.unsafe_intensity,
            "safe_arrival_rate_rps": capacity.safe_intensity * base_rate,
            "unsafe_arrival_rate_rps": (
                capacity.unsafe_intensity * base_rate
                if capacity.unsafe_intensity is not None
                else None
            ),
            "right_censored": capacity.right_censored,
            "trials": len(capacity.trials),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
