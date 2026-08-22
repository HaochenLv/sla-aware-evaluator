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
    return (
        Path(__file__).resolve().parents[2]
        / "data/raw/helix/Helix-ASPLOS25"
    )


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
    result = {
        "helix_commit": HELIX_COMMIT,
        "experiment_quality": "mixed: M profile, generated Azure-derived workload, S topology/SLA",
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
            "safe_intensity_lower_bound": round(capacity.safe_intensity, 4),
            "unsafe_intensity_upper_bound": round(capacity.unsafe_intensity, 4),
            "safe_arrival_rate_lower_bound_rps": round(
                capacity.safe_intensity * base_rate, 4
            ),
            "first_unsafe_bottleneck": violation.kind.value if violation else None,
            "bottleneck_object": violation.object_id if violation else None,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
