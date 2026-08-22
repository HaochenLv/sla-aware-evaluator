from __future__ import annotations

import json

from .capacity import find_capacity
from .domain import (
    Boundary,
    EvaluatorConfig,
    GPUNode,
    Link,
    ModelSpec,
    Pipeline,
    RequestSpec,
    SLA,
    Stage,
)


GIB = 1024**3


def build_model() -> ModelSpec:
    # Qwen2.5-7B structural fields come from raw_data/model.md. The profile is
    # explicitly analytical/estimated, not a measured Qwen GPU profile.
    return ModelSpec(
        name="Qwen2.5-7B",
        num_layers=28,
        total_params=7e9,
        weight_bytes_per_param=2,
        hidden_size=3584,
        num_attention_heads=28,
        num_kv_heads=4,
        kv_element_bytes=2,
        flops_per_token_per_layer=24 * 3584**2,
    )


def build_infrastructure() -> tuple[dict[str, GPUNode], dict[str, Link]]:
    nodes = {
        "edge-a": GPUNode("edge-a", 35.0, 24 * GIB, 1 * GIB, 1 * GIB),
        "accelerator-b": GPUNode("accelerator-b", 100.0, 40 * GIB, 1 * GIB, 1 * GIB),
        "edge-c": GPUNode("edge-c", 50.0, 24 * GIB, 1 * GIB, 1 * GIB),
    }
    links = {
        "a-b": Link("a-b", "edge-a", "accelerator-b", 0.50e9 / 8),
        "b-c": Link("b-c", "accelerator-b", "edge-c", 0.50e9 / 8),
        "a-c": Link("a-c", "edge-a", "edge-c", 10e9 / 8),
    }
    return nodes, links


def build_pipelines() -> tuple[Pipeline, Pipeline]:
    model = build_model()
    nodes, links = build_infrastructure()
    compute_first = Pipeline(
        id="compute-first-slow-network",
        model=model,
        nodes=nodes,
        links=links,
        stages=(
            Stage("s0", 0, 7, "edge-a"),
            Stage("s1", 7, 21, "accelerator-b"),
            Stage("s2", 21, 28, "edge-c"),
        ),
        boundaries=(
            Boundary("s0", "s1", ("a-b",)),
            Boundary("s1", "s2", ("b-c",)),
        ),
    )
    network_aware = Pipeline(
        id="network-aware-fast-link",
        model=model,
        nodes=nodes,
        links=links,
        stages=(
            Stage("s0", 0, 14, "edge-a"),
            Stage("s1", 14, 28, "edge-c"),
        ),
        boundaries=(Boundary("s0", "s1", ("a-c",)),),
    )
    return compute_first, network_aware


def build_workload() -> tuple[RequestSpec, ...]:
    lengths = (
        (512, 80),
        (1024, 128),
        (768, 96),
        (384, 64),
        (1280, 160),
        (640, 96),
        (896, 128),
        (512, 80),
        (1152, 144),
        (448, 72),
        (960, 128),
        (704, 96),
    )
    arrival = 0.0
    gaps = (0.72, 0.55, 0.83, 0.48, 0.69, 0.92, 0.51, 0.77, 0.44, 0.88, 0.63)
    result = []
    for index, (input_tokens, output_tokens) in enumerate(lengths):
        if index:
            arrival += gaps[index - 1]
        result.append(
            RequestSpec(f"r{index:02d}", arrival, input_tokens, output_tokens)
        )
    return tuple(result)


def main() -> None:
    config = EvaluatorConfig(decode_block_size=16)
    sla = SLA(ttft_s=1.0, tpot_s=0.070, fixed_overhead_s=0.005)
    workload = build_workload()
    base_arrival_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    output = {
        "data_quality": "E (analytical/estimated MVP profile)",
        "sla": {"ttft_s": sla.ttft_s, "tpot_s": sla.tpot_s},
        "base_workload_arrival_rate_rps": round(base_arrival_rate, 4),
        "pipelines": {},
    }
    for pipeline in build_pipelines():
        capacity = find_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            config=config,
            tolerance=0.01,
        )
        violation = capacity.representative_unsafe_run.first_violation
        output["pipelines"][pipeline.id] = {
            "safe_intensity_lower_bound": round(capacity.safe_intensity, 4),
            "unsafe_intensity_upper_bound": round(capacity.unsafe_intensity, 4),
            "safe_arrival_rate_lower_bound_rps": round(
                capacity.safe_intensity * base_arrival_rate, 4
            ),
            "first_unsafe_bottleneck": violation.kind.value if violation else None,
            "bottleneck_object": violation.object_id if violation else None,
            "capacity_trials": len(capacity.trials),
        }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
