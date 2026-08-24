from __future__ import annotations

import json
import time

from .domain import SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .reference import find_reference_capacity
from .reference_diagnostics import diagnose_reference_tpot
from .reference_round import find_reference_capacity_round
from .reference_round_diagnostics import diagnose_reference_round_tpot
from .workload import build_helix_azure_conversation_workload


def _run_summary(capacity, base_rate: float) -> dict:
    safe = capacity.representative_safe_run
    unsafe = capacity.representative_unsafe_run
    violation = unsafe.first_violation if unsafe is not None else None
    return {
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
        "safe_max_ttft_s": max(safe.ttft_s_by_request.values(), default=0.0),
        "safe_max_tpot_s": max(safe.max_tpot_s_by_request.values(), default=0.0),
        "safe_completed_requests": safe.completed_requests,
        "safe_completed_tokens": safe.completed_tokens,
        "safe_peak_prefill": safe.peak_prefill,
        "safe_peak_decode": safe.peak_decode,
        "unsafe_completed_requests": unsafe.completed_requests if unsafe else None,
        "unsafe_completed_tokens": unsafe.completed_tokens if unsafe else None,
        "unsafe_peak_prefill": unsafe.peak_prefill if unsafe else None,
        "unsafe_peak_decode": unsafe.peak_decode if unsafe else None,
        "unsafe_kind": violation.kind if violation else None,
        "unsafe_request_id": violation.request_id if violation else None,
        "unsafe_observed_s": violation.observed if violation else None,
        "unsafe_time_s": violation.time_s if violation else None,
    }


def _diagnostic_summary(diagnostic) -> dict:
    return {
        "request_id": diagnostic.request_id,
        "token_index": diagnostic.failing_output_token_index,
        "context_tokens": diagnostic.failing_context_tokens,
        "observed_tpot_s": diagnostic.observed_tpot_s,
        "gpu_queue_wait_s": diagnostic.gpu_queue_wait_s,
        "gpu_service_s": diagnostic.gpu_service_elapsed_s,
        "link_queue_wait_s": diagnostic.link_queue_wait_s,
        "link_service_s": diagnostic.explicit_link_service_s,
        "overhead_s": diagnostic.configured_overhead_s,
        "unattributed_s": diagnostic.unattributed_s,
        "decode_batch_sizes": diagnostic.decode_batch_sizes,
        "counterfactual_feasible": diagnostic.counterfactual_feasible,
        "counterfactual_first_violation_kind": (
            diagnostic.counterfactual_first_violation_kind
        ),
        "counterfactual_first_violation_observed_s": (
            diagnostic.counterfactual_first_violation_observed_s
        ),
        "counterfactual_gpu_queue_wait_s": (
            diagnostic.counterfactual_gpu_queue_wait_s
        ),
        "counterfactual_link_queue_wait_s": (
            diagnostic.counterfactual_link_queue_wait_s
        ),
    }


def main() -> None:
    root = artifact_root()
    profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
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

    output = {
        "experiment": "Reference v0 vs deterministic Decode-round",
        "helix_commit": HELIX_COMMIT,
        "workload": {
            "duration_s": 30,
            "requests": len(workload),
            "total_output_tokens": sum(request.output_tokens for request in workload),
            "base_arrival_rate_rps": base_rate,
            "source": sample.provenance.source,
        },
        "sla": {
            "ttft_s": sla.ttft_s,
            "tpot_s": sla.tpot_s,
            "fixed_overhead_s": sla.fixed_overhead_s,
        },
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        pipeline_result = {}
        for name, finder, diagnose in (
            ("reference_v0", find_reference_capacity, diagnose_reference_tpot),
            (
                "reference_decode_round",
                find_reference_capacity_round,
                diagnose_reference_round_tpot,
            ),
        ):
            started = time.perf_counter()
            capacity = finder(
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                profiler=profiler,
                tolerance=0.02,
                max_intensity=16.0,
                verification_grid_points=9,
            )
            elapsed = time.perf_counter() - started
            summary = _run_summary(capacity, base_rate)
            summary["elapsed_wall_s"] = elapsed
            if (
                capacity.unsafe_intensity is not None
                and summary["unsafe_kind"] == "tpot"
            ):
                diagnostic, _ = diagnose(
                    pipeline=pipeline,
                    workload=workload,
                    sla=sla,
                    profiler=profiler,
                    intensity=capacity.unsafe_intensity,
                )
                summary["diagnostic"] = _diagnostic_summary(diagnostic)
            else:
                summary["diagnostic"] = None
            pipeline_result[name] = summary
        output["pipelines"][pipeline.id] = pipeline_result

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
