from __future__ import annotations

import json
import time

from .capacity import find_capacity
from .domain import EvaluatorConfig, SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .reference import ReferenceConfig, find_reference_capacity
from .workload import build_helix_azure_conversation_workload


def _reference_run_summary(run, pipeline) -> dict:
    return {
        "feasible": run.feasible,
        "final_time_s": run.final_time_s,
        "processed_events": run.processed_events,
        "completed_requests": run.completed_requests,
        "completed_tokens": run.completed_tokens,
        "peak_prefill": run.peak_prefill,
        "peak_decode": run.peak_decode,
        "max_ttft_s": max(run.ttft_s_by_request.values(), default=0.0),
        "max_tpot_s": max(run.max_tpot_s_by_request.values(), default=0.0),
        "peak_memory_bytes": dict(run.peak_memory_bytes),
        "peak_memory_utilization": {
            node_id: (
                used / pipeline.nodes[node_id].memory_capacity_bytes
                if pipeline.nodes[node_id].memory_capacity_bytes > 0
                else None
            )
            for node_id, used in run.peak_memory_bytes.items()
        },
        "first_violations": [
            {
                "time_s": violation.time_s,
                "kind": violation.kind,
                "object_id": violation.object_id,
                "observed": violation.observed,
                "limit": violation.limit,
                "request_id": violation.request_id,
            }
            for violation in run.first_violations
        ],
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

    result = {
        "experiment_role": (
            "reference-v0 smoke test only; two pipelines are insufficient for "
            "ranking-correlation validation"
        ),
        "helix_commit": HELIX_COMMIT,
        "profile": {
            "source": profiler.provenance.source,
            "model": profiler.provenance.source_model,
            "gpu": profiler.provenance.source_gpu,
            "quality": profiler.provenance.quality_label,
            "observed_dimensions": sorted(profiler.provenance.observed_dimensions),
        },
        "workload": {
            "source": workload_sample.provenance.source,
            "kind": workload_sample.provenance.kind,
            "duration_s": 30,
            "requests": len(workload),
            "base_arrival_rate_rps": base_rate,
        },
        "sla": {
            "ttft_s": sla.ttft_s,
            "tpot_s": sla.tpot_s,
            "fixed_overhead_s": sla.fixed_overhead_s,
        },
        "conservative_semantics": {
            "progress_policy": "compute_only",
            "network": "SLA-derived bandwidth reservation/red-line",
        },
        "reference_semantics": {
            "gpu": "per-node FIFO queued service; contiguous same-stage Decode microbatch",
            "network": "per-physical-link FCFS explicit D/B transfers",
            "ttft": "end-of-Prefill latency plus configured overhead",
            "tpot": "every end-to-end Decode-token interval",
            "memory": "static weights plus active-request KV context",
            "important_limit": "reference v0 is an independent simulator, not a reproduction of HELIX/vLLM runtime scheduling",
        },
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        conservative_started = time.perf_counter()
        conservative = find_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            config=conservative_config,
            profiler=profiler,
            tolerance=0.02,
            max_intensity=16.0,
            verification_grid_points=9,
        )
        conservative_elapsed = time.perf_counter() - conservative_started

        reference_started = time.perf_counter()
        reference = find_reference_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=profiler,
            config=reference_config,
            tolerance=0.02,
            max_intensity=16.0,
            verification_grid_points=9,
        )
        reference_elapsed = time.perf_counter() - reference_started

        conservative_violation = conservative.representative_unsafe_run.first_violation
        reference_violation = reference.representative_unsafe_run.first_violation
        result["pipelines"][pipeline.id] = {
            "conservative": {
                "safe_intensity_lower_bound": conservative.safe_intensity,
                "unsafe_intensity_upper_bound": conservative.unsafe_intensity,
                "safe_arrival_rate_lower_bound_rps": conservative.safe_intensity * base_rate,
                "unsafe_arrival_rate_upper_bound_rps": conservative.unsafe_intensity * base_rate,
                "first_unsafe_kind": (
                    conservative_violation.kind.value if conservative_violation else None
                ),
                "capacity_trials": len(conservative.trials),
                "elapsed_wall_s": conservative_elapsed,
            },
            "reference": {
                "safe_intensity_lower_bound": reference.safe_intensity,
                "unsafe_intensity_upper_bound": reference.unsafe_intensity,
                "safe_arrival_rate_lower_bound_rps": reference.safe_intensity * base_rate,
                "unsafe_arrival_rate_upper_bound_rps": reference.unsafe_intensity * base_rate,
                "first_unsafe_kind": reference_violation.kind if reference_violation else None,
                "capacity_trials": len(reference.trials),
                "elapsed_wall_s": reference_elapsed,
                "safe_run": _reference_run_summary(
                    reference.representative_safe_run, pipeline
                ),
                "unsafe_run": _reference_run_summary(
                    reference.representative_unsafe_run, pipeline
                ),
            },
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
