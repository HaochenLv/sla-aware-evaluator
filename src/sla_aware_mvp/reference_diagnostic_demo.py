from __future__ import annotations

import json

from .domain import SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .reference import ReferenceConfig, find_reference_capacity
from .reference_diagnostics import diagnose_reference_tpot
from .workload import build_helix_azure_conversation_workload


def _pairs(items, value_name: str) -> list[dict]:
    return [
        {"object_id": object_id, value_name: value}
        for object_id, value in items
    ]


def _diagnostic_to_dict(diagnostic) -> dict:
    return {
        "pipeline_id": diagnostic.pipeline_id,
        "intensity": diagnostic.intensity,
        "request_id": diagnostic.request_id,
        "violation_kind": diagnostic.violation_kind,
        "observed_tpot_s": diagnostic.observed_tpot_s,
        "tpot_limit_s": diagnostic.tpot_limit_s,
        "failing_context_tokens": diagnostic.failing_context_tokens,
        "failing_output_token_index": diagnostic.failing_output_token_index,
        "gpu_profile_service_s": diagnostic.gpu_profile_service_s,
        "gpu_service_elapsed_s": diagnostic.gpu_service_elapsed_s,
        "gpu_queue_wait_s": diagnostic.gpu_queue_wait_s,
        "explicit_link_service_s": diagnostic.explicit_link_service_s,
        "link_queue_wait_s": diagnostic.link_queue_wait_s,
        "configured_overhead_s": diagnostic.configured_overhead_s,
        "accounted_service_s": diagnostic.accounted_service_s,
        "fully_attributed_s": diagnostic.fully_attributed_s,
        "residual_wait_s": diagnostic.residual_wait_s,
        "residual_wait_fraction": diagnostic.residual_wait_fraction,
        "unattributed_s": diagnostic.unattributed_s,
        "unattributed_fraction": diagnostic.unattributed_fraction,
        "decode_batch_sizes": diagnostic.decode_batch_sizes,
        "stage_profile_service_s": [
            {"stage_id": stage_id, "service_s": service_s}
            for stage_id, service_s in diagnostic.stage_service_s
        ],
        "stage_gpu_queue_wait_s": [
            {"stage_id": stage_id, "queue_wait_s": wait_s}
            for stage_id, wait_s in diagnostic.stage_gpu_queue_wait_s
        ],
        "stage_gpu_service_elapsed_s": [
            {"stage_id": stage_id, "service_elapsed_s": service_s}
            for stage_id, service_s in diagnostic.stage_gpu_service_elapsed_s
        ],
        "link_queue_wait_s_by_link": _pairs(
            diagnostic.link_queue_wait_s_by_link, "queue_wait_s"
        ),
        "link_service_elapsed_s_by_link": _pairs(
            diagnostic.link_service_elapsed_s_by_link, "service_elapsed_s"
        ),
        "counterfactual_network_multiplier": (
            diagnostic.counterfactual_network_multiplier
        ),
        "counterfactual_feasible": diagnostic.counterfactual_feasible,
        "counterfactual_first_violation_kind": (
            diagnostic.counterfactual_first_violation_kind
        ),
        "counterfactual_first_violation_request_id": (
            diagnostic.counterfactual_first_violation_request_id
        ),
        "counterfactual_first_violation_observed_s": (
            diagnostic.counterfactual_first_violation_observed_s
        ),
        "counterfactual_max_tpot_s": diagnostic.counterfactual_max_tpot_s,
        "counterfactual_gpu_queue_wait_s": (
            diagnostic.counterfactual_gpu_queue_wait_s
        ),
        "counterfactual_gpu_service_elapsed_s": (
            diagnostic.counterfactual_gpu_service_elapsed_s
        ),
        "counterfactual_link_queue_wait_s": (
            diagnostic.counterfactual_link_queue_wait_s
        ),
        "counterfactual_link_service_elapsed_s": (
            diagnostic.counterfactual_link_service_elapsed_s
        ),
        "counterfactual_unattributed_s": diagnostic.counterfactual_unattributed_s,
        "counterfactual_unattributed_fraction": (
            diagnostic.counterfactual_unattributed_fraction
        ),
        "counterfactual_decode_batch_sizes": (
            diagnostic.counterfactual_decode_batch_sizes
        ),
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
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    config = ReferenceConfig()

    result = {
        "experiment_role": (
            "attribute the first Reference-v0 TPOT failure to exact simulated GPU "
            "queue, GPU service, link queue, link service, and configured overhead; "
            "this experiment does not change scheduling or claim ranking validity"
        ),
        "helix_commit": HELIX_COMMIT,
        "workload": {
            "duration_s": 30,
            "requests": len(workload),
            "total_output_tokens": sum(request.output_tokens for request in workload),
        },
        "sla": {
            "ttft_s": sla.ttft_s,
            "tpot_s": sla.tpot_s,
            "fixed_overhead_s": sla.fixed_overhead_s,
        },
        "diagnostic_semantics": {
            "exact_decomposition": (
                "observed TPOT = GPU queue wait + GPU service elapsed + link queue "
                "wait + link service elapsed + configured overhead + unattributed"
            ),
            "trace_policy": (
                "Reference evaluate_reference emits observational enqueue/start/complete "
                "timestamps; queue order, batching, service duration, and feasibility "
                "semantics are unchanged"
            ),
            "residual_wait": (
                "residual_wait is now GPU queue wait + link queue wait; unattributed is "
                "reported separately and should be approximately zero if tracing closes"
            ),
            "counterfactual": (
                "rerun the identical scaled workload with every physical-link capacity "
                "multiplied by 1e6 and attribute its first TPOT failure with the same trace"
            ),
        },
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        capacity = find_reference_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=profiler,
            config=config,
            tolerance=0.02,
            max_intensity=16.0,
            verification_grid_points=9,
        )
        unsafe_intensity = capacity.unsafe_intensity
        if unsafe_intensity is None:
            result["pipelines"][pipeline.id] = {
                "reference_right_censored": True,
                "safe_intensity_lower_bound": capacity.safe_intensity,
                "unsafe_intensity_upper_bound": None,
                "diagnostic": None,
            }
            continue

        diagnostic, original_run = diagnose_reference_tpot(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=profiler,
            intensity=unsafe_intensity,
            config=config,
            network_capacity_multiplier=1_000_000.0,
        )
        result["pipelines"][pipeline.id] = {
            "reference_right_censored": False,
            "safe_intensity_lower_bound": capacity.safe_intensity,
            "unsafe_intensity_upper_bound": unsafe_intensity,
            "original_run": {
                "feasible": original_run.feasible,
                "processed_events": original_run.processed_events,
                "completed_requests": original_run.completed_requests,
                "completed_tokens": original_run.completed_tokens,
                "peak_prefill": original_run.peak_prefill,
                "peak_decode": original_run.peak_decode,
            },
            "diagnostic": _diagnostic_to_dict(diagnostic),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
