from __future__ import annotations

import json
import time

from .capacity import find_capacity
from .domain import EvaluatorConfig, SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .reference import ReferenceConfig, find_reference_capacity
from .reference_v1 import find_reference_capacity_v1
from .reference_v1_diagnostics import diagnose_reference_v1_tpot
from .workload import build_helix_azure_conversation_workload


def _rate(intensity: float | None, base_rate: float) -> float | None:
    return intensity * base_rate if intensity is not None else None


def _capacity_summary(capacity, base_rate: float, elapsed_s: float) -> dict:
    unsafe = capacity.representative_unsafe_run
    violation = unsafe.first_violation if unsafe is not None else None
    return {
        "safe_intensity": capacity.safe_intensity,
        "unsafe_intensity": capacity.unsafe_intensity,
        "safe_arrival_rate_rps": _rate(capacity.safe_intensity, base_rate),
        "unsafe_arrival_rate_rps": _rate(capacity.unsafe_intensity, base_rate),
        "right_censored": capacity.right_censored,
        "first_unsafe_kind": (
            violation.kind.value if hasattr(getattr(violation, "kind", None), "value")
            else violation.kind if violation is not None else None
        ),
        "capacity_trials": len(capacity.trials),
        "elapsed_wall_s": elapsed_s,
    }


def main() -> None:
    root = artifact_root()
    profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    workload_sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = workload_sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    conservative_config = EvaluatorConfig(decode_block_size=16)
    reference_config = ReferenceConfig()
    search_max_intensity = 16.0

    output = {
        "experiment": "E1 Reference v0 vs v1 Decode batching",
        "controlled_change": (
            "Reference Decode batching only: v0 opportunistic per-node microbatch -> "
            "v1 deterministic cross-stage Decode cohort"
        ),
        "unchanged": [
            "Conservative Evaluator",
            "HELIX LLaMA-2-70B A100 profile",
            "30 s Azure-derived workload",
            "SLA",
            "Pipeline placement/network",
            "Prefill FIFO",
            "explicit FCFS D/B links",
            "memory model",
        ],
        "workload_requests": len(workload),
        "total_output_tokens": sum(request.output_tokens for request in workload),
        "base_arrival_rate_rps": base_rate,
        "sla": {"ttft_s": 2.0, "tpot_s": 0.150, "fixed_overhead_s": 0.005},
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        started = time.perf_counter()
        conservative = find_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            config=conservative_config,
            profiler=profiler,
            tolerance=0.02,
            max_intensity=search_max_intensity,
            verification_grid_points=9,
        )
        conservative_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        v0 = find_reference_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=profiler,
            config=reference_config,
            tolerance=0.02,
            max_intensity=search_max_intensity,
            verification_grid_points=9,
        )
        v0_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        v1 = find_reference_capacity_v1(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=profiler,
            config=reference_config,
            tolerance=0.02,
            max_intensity=search_max_intensity,
            verification_grid_points=9,
        )
        v1_elapsed = time.perf_counter() - started

        diagnostic = None
        if v1.unsafe_intensity is not None:
            diag, _ = diagnose_reference_v1_tpot(
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                profiler=profiler,
                intensity=v1.unsafe_intensity,
                config=reference_config,
            )
            diagnostic = {
                "request_id": diag.request_id,
                "observed_tpot_s": diag.observed_tpot_s,
                "failing_output_token_index": diag.failing_output_token_index,
                "decode_batch_sizes": list(diag.decode_batch_sizes),
                "gpu_queue_wait_s": diag.gpu_queue_wait_s,
                "gpu_service_s": diag.gpu_service_elapsed_s,
                "link_queue_wait_s": diag.link_queue_wait_s,
                "link_service_s": diag.explicit_link_service_s,
                "configured_overhead_s": diag.configured_overhead_s,
                "unattributed_s": diag.unattributed_s,
                "counterfactual_feasible": diag.counterfactual_feasible,
                "counterfactual_first_violation_kind": (
                    diag.counterfactual_first_violation_kind
                ),
                "counterfactual_gpu_queue_wait_s": (
                    diag.counterfactual_gpu_queue_wait_s
                ),
                "counterfactual_link_queue_wait_s": (
                    diag.counterfactual_link_queue_wait_s
                ),
            }

        output["pipelines"][pipeline.id] = {
            "conservative": _capacity_summary(
                conservative, base_rate, conservative_elapsed
            ),
            "reference_v0": _capacity_summary(v0, base_rate, v0_elapsed),
            "reference_v1": _capacity_summary(v1, base_rate, v1_elapsed),
            "reference_v1_first_unsafe_diagnostic": diagnostic,
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
