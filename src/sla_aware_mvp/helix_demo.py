from __future__ import annotations

import json
from pathlib import Path

from .capacity import find_capacity
from .domain import (
    Boundary,
    EvaluatorConfig,
    GPUNode,
    Link,
    ModelSpec,
    Pipeline,
    SLA,
    Stage,
)
from .helix import HelixA100Llama2Profiler
from .workload import build_helix_azure_conversation_workload


HELIX_COMMIT = "8639497a4aaf1eb3b7594614cb0bbd376c1342b3"


def artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data/raw/helix/Helix-ASPLOS25"


def build_helix_model() -> ModelSpec:
    return ModelSpec(
        name="LLaMA-2-70B",
        num_layers=80,
        total_params=70e9,
        weight_bytes_per_param=2,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        kv_element_bytes=2,
        activation_element_bytes=2,
        flops_per_token_per_layer=24 * 8192**2,
    )


def build_helix_pipelines() -> tuple[Pipeline, Pipeline]:
    model = build_helix_model()
    nodes = {
        f"n{index}": GPUNode(
            id=f"n{index}",
            compute_tflops=0.0,
            memory_capacity_bytes=int(40e9),
            hardware_type="A100-40GB",
        )
        for index in range(8)
    }
    fast_order = tuple(f"n{index}" for index in range(8))
    slow_order = ("n0", "n2", "n4", "n6", "n1", "n3", "n5", "n7")
    links: dict[str, Link] = {}
    for index in range(7):
        links[f"fast-{index}"] = Link(
            f"fast-{index}", fast_order[index], fast_order[index + 1], 10e9 / 8
        )
        links[f"slow-{index}"] = Link(
            f"slow-{index}", slow_order[index], slow_order[index + 1], 2.5e9 / 8
        )

    def make_pipeline(identifier: str, order: tuple[str, ...], prefix: str) -> Pipeline:
        stages = tuple(
            Stage(f"s{index}", index * 10, (index + 1) * 10, node_id)
            for index, node_id in enumerate(order)
        )
        boundaries = tuple(
            Boundary(f"s{index}", f"s{index + 1}", (f"{prefix}-{index}",))
            for index in range(7)
        )
        return Pipeline(identifier, model, nodes, links, stages, boundaries)

    return (
        make_pipeline("helix-slow-link-placement", slow_order, "slow"),
        make_pipeline("helix-fast-link-placement", fast_order, "fast"),
    )


def _run_summary(run, pipeline: Pipeline) -> dict:
    if run is None:
        return {}
    return {
        "feasible": run.feasible,
        "final_time_s": run.final_time_s,
        "processed_events": run.processed_events,
        "peak_prefill": run.peak_prefill,
        "peak_decode": run.peak_decode,
        "min_link_headroom_bytes_per_s": run.min_link_headroom_bytes_per_s,
        "peak_memory_bytes": run.peak_memory_bytes,
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
                "kind": violation.kind.value,
                "object_id": violation.object_id,
                "required": violation.required,
                "capacity": violation.capacity,
                "request_id": violation.request_id,
                "num_prefill": violation.num_prefill,
                "num_decode": violation.num_decode,
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
        duration_s=120,
        target_request_rate=1.5,
        seed=7,
    )
    workload = workload_sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    config = EvaluatorConfig(decode_block_size=16)
    progress_policy = (
        "full_sla_window_ablation"
        if config.conservative_network_lifetime
        else "compute_only"
    )
    result = {
        "helix_commit": HELIX_COMMIT,
        "experiment_quality": "mixed: M profile, generated Azure-derived workload, S topology/SLA",
        "evaluator_semantics": {
            "progress_policy": progress_policy,
            "conservative_network_lifetime": config.conservative_network_lifetime,
            "capacity_search": "sampled monotonicity verification + local refinement",
        },
        "profile": {
            "source": profiler.provenance.source,
            "model": profiler.provenance.source_model,
            "gpu": profiler.provenance.source_gpu,
            "quality": profiler.provenance.quality_label,
            "context_observed": "context_len" in profiler.provenance.observed_dimensions,
        },
        "workload": {
            "source": workload_sample.provenance.source,
            "kind": workload_sample.provenance.kind,
            "requests": len(workload),
            "base_arrival_rate_rps": round(base_rate, 4),
        },
        "sla": {"ttft_s": sla.ttft_s, "tpot_s": sla.tpot_s},
        "pipelines": {},
    }
    for pipeline in build_helix_pipelines():
        capacity = find_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            config=config,
            profiler=profiler,
            tolerance=0.01,
        )
        violation = capacity.representative_unsafe_run.first_violation
        result["pipelines"][pipeline.id] = {
            "safe_intensity_lower_bound": capacity.safe_intensity,
            "unsafe_intensity_upper_bound": capacity.unsafe_intensity,
            "safe_arrival_rate_lower_bound_rps": capacity.safe_intensity * base_rate,
            "unsafe_arrival_rate_upper_bound_rps": capacity.unsafe_intensity * base_rate,
            "first_unsafe_bottleneck": violation.kind.value if violation else None,
            "bottleneck_object": violation.object_id if violation else None,
            "capacity_trials": len(capacity.trials),
            "monotonicity_verified_on_samples": capacity.monotonicity_verified_on_samples,
            "verification_probe_intensities": capacity.verification_probe_intensities,
            "safe_run": _run_summary(capacity.representative_safe_run, pipeline),
            "unsafe_run": _run_summary(capacity.representative_unsafe_run, pipeline),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
