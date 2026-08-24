from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import find_capacity
from .domain import EvaluatorConfig, SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_capacity import find_helix_fixed_capacity
from .workload import build_helix_azure_conversation_workload


def _base_rate(workload) -> float:
    return (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )


def main() -> None:
    helix_root_env = os.environ.get("HELIX_ROOT")
    if not helix_root_env:
        raise RuntimeError("set HELIX_ROOT to the pinned Helix-ASPLOS25 checkout")
    helix_root = Path(helix_root_env)

    sample = build_helix_azure_conversation_workload(
        helix_root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = sample.requests
    base_rate = _base_rate(workload)
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    profiler = HelixA100Llama2Profiler.from_artifact(helix_root, commit=HELIX_COMMIT)

    result = {
        "experiment": "30s fixed-pipeline Conservative vs HELIX Reference capacity",
        "helix_commit": HELIX_COMMIT,
        "workload_requests": len(workload),
        "workload_output_tokens": sum(request.output_tokens for request in workload),
        "base_arrival_rate_rps": base_rate,
        "sla": {"ttft_s": sla.ttft_s, "tpot_s": sla.tpot_s},
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        conservative = find_capacity(
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
        reference = find_helix_fixed_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            helix_root=helix_root,
            initial_intensity=0.25,
            tolerance=0.05,
            max_intensity=4.0,
        )
        result["pipelines"][pipeline.id] = {
            "conservative": {
                "safe_intensity": conservative.safe_intensity,
                "unsafe_intensity": conservative.unsafe_intensity,
                "safe_arrival_rate_rps": conservative.safe_intensity * base_rate,
                "unsafe_arrival_rate_rps": (
                    conservative.unsafe_intensity * base_rate
                    if conservative.unsafe_intensity is not None
                    else None
                ),
                "trials": len(conservative.trials),
            },
            "helix_reference": {
                "safe_intensity": reference.safe_intensity,
                "unsafe_intensity": reference.unsafe_intensity,
                "safe_arrival_rate_rps": reference.safe_intensity * base_rate,
                "unsafe_arrival_rate_rps": (
                    reference.unsafe_intensity * base_rate
                    if reference.unsafe_intensity is not None
                    else None
                ),
                "safe_max_aligned_ttft_s": max(
                    metric.aligned_ttft_s
                    for metric in reference.safe_run.query_metrics.values()
                ),
                "safe_max_true_first_token_ttft_s": max(
                    metric.true_first_token_ttft_s
                    for metric in reference.safe_run.query_metrics.values()
                ),
                "safe_max_tpot_s": max(
                    metric.max_tpot_s
                    for metric in reference.safe_run.query_metrics.values()
                ),
                "first_unsafe_kind": (
                    reference.unsafe_run.first_violation_kind
                    if reference.unsafe_run is not None
                    else None
                ),
                "trials": len(reference.trials),
            },
            "safe_capacity_ratio_conservative_over_reference": (
                conservative.safe_intensity / reference.safe_intensity
                if reference.safe_intensity > 0
                else None
            ),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
