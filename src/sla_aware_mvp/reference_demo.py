from __future__ import annotations

import json
import time

from .capacity import find_capacity
from .domain import EvaluatorConfig, SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .reference import ReferenceConfig, find_reference_capacity
from .workload import build_helix_azure_conversation_workload


def _reference_run_summary(run, pipeline) -> dict | None:
    if run is None:
        return None
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


def _rate(intensity: float | None, base_rate: float) -> float | None:
    return intensity * base_rate if intensity is not None else None


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

    result = {
        "experiment_role": (
            "reference-v0 execution smoke test; capacity may be right-censored, "
            "and two pipelines are insufficient for ranking-correlation validation"
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
            "total_output_tokens": sum(request.output_tokens for request in workload),
            "base_arrival_rate_rps": base_rate,
        },
        "sla": {
            "ttft_s": sla.ttft_s,
            "tpot_s": sla.tpot_s,
            "fixed_overhead_s": sla.fixed_overhead_s,
        },
        "capacity_search": {
            "max_intensity": search_max_intensity,
            "right_censoring_semantics": (
                "if max_intensity is still safe, report C >= max_intensity with "
                "unsafe bound/run = null and continue the smoke experiment"
            ),
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
            max_intensity=search_max_intensity,
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
            max_intensity=search_max_intensity,
            verification_grid_points=9,
        )
        reference_elapsed = time.perf_counter() - reference_started

        conservative_unsafe = conservative.representative_unsafe_run
        conservative_violation = (
            conservative_unsafe.first_violation if conservative_unsafe is not None else None
        )
        reference_unsafe = reference.representative_unsafe_run
        reference_violation = (
            reference_unsafe.first_violation if reference_unsafe is not None else None
        )

        result["pipelines"][pipeline.id] = {
            "conservative": {
                "safe_intensity_lower_bound": conservative.safe_intensity,
                "unsafe_intensity_upper_bound": conservative.unsafe_intensity,
                "safe_arrival_rate_lower_bound_rps": _rate(
                    conservative.safe_intensity, base_rate
                ),
                "unsafe_arrival_rate_upper_bound_rps": _rate(
                    conservative.unsafe_intensity, base_rate
                ),
                "right_censored": conservative.right_censored,
                "search_max_intensity": conservative.search_max_intensity,
                "first_unsafe_kind": (
                    conservative_violation.kind.value if conservative_violation else None
                ),
                "capacity_trials": len(conservative.trials),
                "monotonicity_verified_on_samples": (
                    conservative.monotonicity_verified_on_samples
                ),
                "verification_probe_intensities": (
                    conservative.verification_probe_intensities
                ),
                "elapsed_wall_s": conservative_elapsed,
            },
            "reference": {
                "safe_intensity_lower_bound": reference.safe_intensity,
                "unsafe_intensity_upper_bound": reference.unsafe_intensity,
                "safe_arrival_rate_lower_bound_rps": _rate(
                    reference.safe_intensity, base_rate
                ),
                "unsafe_arrival_rate_upper_bound_rps": _rate(
                    reference.unsafe_intensity, base_rate
                ),
                "right_censored": reference.right_censored,
                "search_max_intensity": reference.search_max_intensity,
                "first_unsafe_kind": (
                    reference_violation.kind if reference_violation else None
                ),
                "capacity_trials": len(reference.trials),
                "monotonicity_verified_on_samples": (
                    reference.monotonicity_verified_on_samples
                ),
                "verification_probe_intensities": (
                    reference.verification_probe_intensities
                ),
                "elapsed_wall_s": reference_elapsed,
                "safe_run": _reference_run_summary(
                    reference.representative_safe_run, pipeline
                ),
                "unsafe_run": _reference_run_summary(reference_unsafe, pipeline),
            },
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
