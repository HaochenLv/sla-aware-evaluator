from __future__ import annotations

import json

from .domain import SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .reference import ReferenceConfig, find_reference_capacity
from .reference_diagnostics import diagnose_reference_tpot
from .workload import build_helix_azure_conversation_workload


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
        "explicit_link_service_s": diagnostic.explicit_link_service_s,
        "configured_overhead_s": diagnostic.configured_overhead_s,
        "accounted_service_s": diagnostic.accounted_service_s,
        "residual_wait_s": diagnostic.residual_wait_s,
        "residual_wait_fraction": diagnostic.residual_wait_fraction,
        "decode_batch_sizes": diagnostic.decode_batch_sizes,
        "stage_service_s": [
            {"stage_id": stage_id, "service_s": service_s}
            for stage_id, service_s in diagnostic.stage_service_s
        ],
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
            "diagnose why Reference v0 TPOT jumps when Decode overlap appears; "
            "this experiment does not change the scheduler or claim ranking validity"
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
            "decomposition": (
                "observed TPOT = profiled GPU service + explicit D/B link service "
                "+ configured overhead + residual waiting/synchronization"
            ),
            "residual_warning": (
                "residual waiting is not labeled GPU wait; it can include GPU queue, "
                "link queue, and batching/synchronization effects"
            ),
            "counterfactual": (
                "rerun the identical scaled workload with every physical-link capacity "
                "multiplied by 1e6; persistence of TPOT failure indicates network is "
                "not the dominant source under Reference v0"
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
