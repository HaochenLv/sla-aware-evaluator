from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import Boundary, EvaluatorConfig, GPUNode, Link, Pipeline, RequestSpec, SLA, Stage
from .evaluator import evaluate
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_model
from .helix_fixed_reference import evaluate_helix_fixed_reference


PATTERNS: dict[str, tuple[float, ...]] = {
    "mild": (0.7, 1.0, 1.4, 0.8, 1.8, 1.1, 2.2),
    "one_bottleneck": (0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
}
PROMPTS = (1024, 1709)
HELIX_TOKEN_BYTES = 2
HELIX_CPU_CONCAT_BYTES_PER_S = 4e9
HELIX_CPU_TRANSFER_BYTES_PER_S = 5e9
BISECTION_STEPS = 8


def build_scaled_heterogeneous_pipeline(pattern_name: str, scale_gbps: float) -> Pipeline:
    ratios = PATTERNS[pattern_name]
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
    stages = tuple(
        Stage(f"s{index}", index * 10, (index + 1) * 10, f"n{index}")
        for index in range(8)
    )
    links = {
        f"link-{index}": Link(
            id=f"link-{index}",
            source=f"n{index}",
            target=f"n{index + 1}",
            capacity_bytes_per_s=scale_gbps * ratios[index] * 1e9 / 8.0,
        )
        for index in range(7)
    }
    boundaries = tuple(
        Boundary(f"s{index}", f"s{index + 1}", (f"link-{index}",))
        for index in range(7)
    )
    pipeline = Pipeline(
        id=f"hetero-{pattern_name}-scale-{scale_gbps:.9f}",
        model=model,
        nodes=nodes,
        links=links,
        stages=stages,
        boundaries=boundaries,
    )
    pipeline.validate()
    return pipeline


def _serial_coefficient_s_gbps(data_bytes: float, ratios: tuple[float, ...]) -> float:
    """Return c such that total serial link time is c / scale_gbps seconds."""
    return sum(data_bytes / (ratio * 1e9 / 8.0) for ratio in ratios)


def _current_evaluator_threshold(
    *, prompt_tokens: int, pattern_name: str, profiler: HelixA100Llama2Profiler, sla: SLA
) -> dict:
    pipeline = build_scaled_heterogeneous_pipeline(pattern_name, 10.0)
    request = RequestSpec("eval", 0.0, prompt_tokens, 1)
    compute_s = profiler.prefill_time(request, pipeline, 1, 0)
    remaining_s = sla.ttft_s - compute_s - sla.queue_overhead_s - sla.fixed_overhead_s
    payload_bytes = pipeline.model.activation_bytes_per_token * prompt_tokens
    coefficient = _serial_coefficient_s_gbps(payload_bytes, PATTERNS[pattern_name])
    threshold = coefficient / remaining_s
    return {
        "prefill_compute_s": compute_s,
        "remaining_budget_s": remaining_s,
        "payload_bytes": payload_bytes,
        "serial_coefficient_s_gbps": coefficient,
        "minimum_scale_gbps": threshold,
    }


def _source_aware_threshold(
    *, prompt_tokens: int, pattern_name: str, profiler: HelixA100Llama2Profiler, sla: SLA
) -> dict:
    """Diagnostic prediction using pinned HELIX runtime overheads, not a model revision."""
    pipeline = build_scaled_heterogeneous_pipeline(pattern_name, 10.0)
    request = RequestSpec("source", 0.0, prompt_tokens, 1)
    compute_s = profiler.prefill_time(request, pipeline, 1, 0)
    activation_bytes = pipeline.model.activation_bytes_per_token * prompt_tokens
    cpu_overhead_per_node_s = (
        activation_bytes / HELIX_CPU_CONCAT_BYTES_PER_S
        + activation_bytes / HELIX_CPU_TRANSFER_BYTES_PER_S
    )
    cpu_overhead_s = len(pipeline.stages) * cpu_overhead_per_node_s
    remaining_s = sla.ttft_s - compute_s - cpu_overhead_s - sla.fixed_overhead_s
    network_payload_bytes = (
        pipeline.model.activation_bytes_per_token + HELIX_TOKEN_BYTES
    ) * prompt_tokens
    coefficient = _serial_coefficient_s_gbps(network_payload_bytes, PATTERNS[pattern_name])
    threshold = coefficient / remaining_s
    return {
        "prefill_compute_s": compute_s,
        "helix_cpu_buffer_overhead_s": cpu_overhead_s,
        "remaining_budget_s": remaining_s,
        "network_payload_bytes": network_payload_bytes,
        "serial_coefficient_s_gbps": coefficient,
        "minimum_scale_gbps": threshold,
    }


def _evaluator_probe(
    *, prompt_tokens: int, pattern_name: str, scale_gbps: float,
    profiler: HelixA100Llama2Profiler, sla: SLA,
) -> dict:
    request = RequestSpec("eval-probe", 0.0, prompt_tokens, 1)
    result = evaluate(
        pipeline=build_scaled_heterogeneous_pipeline(pattern_name, scale_gbps),
        workload=(request,),
        sla=sla,
        config=EvaluatorConfig(),
        profiler=profiler,
    )
    return {
        "scale_gbps": scale_gbps,
        "feasible": result.feasible,
        "first_violation_kind": result.first_violation.kind.value if result.first_violation else None,
    }


def _helix_probe(
    *, root: Path, prompt_tokens: int, pattern_name: str, scale_gbps: float, sla: SLA
) -> dict:
    request = RequestSpec("helix-probe", 0.0, prompt_tokens, 1)
    run = evaluate_helix_fixed_reference(
        pipeline=build_scaled_heterogeneous_pipeline(pattern_name, scale_gbps),
        workload=(request,),
        sla=sla,
        helix_root=root,
    )
    metric = run.query_metrics[request.id]
    return {
        "scale_gbps": scale_gbps,
        "aligned_ttft_s": metric.aligned_ttft_s,
        "aligned_ttft_safe": metric.aligned_ttft_s <= sla.ttft_s,
        "true_first_token_ttft_s": metric.true_first_token_ttft_s,
        "max_tpot_s": metric.max_tpot_s,
        "reference_feasible": run.feasible,
        "first_violation_kind": run.first_violation_kind,
    }


def _helix_threshold_bracket(
    *, root: Path, prompt_tokens: int, pattern_name: str, center_gbps: float, sla: SLA
) -> tuple[dict, dict, list[dict]]:
    cache: dict[float, dict] = {}

    def probe(scale_gbps: float) -> dict:
        key = round(scale_gbps, 12)
        if key not in cache:
            cache[key] = _helix_probe(
                root=root,
                prompt_tokens=prompt_tokens,
                pattern_name=pattern_name,
                scale_gbps=scale_gbps,
                sla=sla,
            )
        return cache[key]

    low = center_gbps * 0.8
    high = center_gbps * 1.2
    low_result = probe(low)
    high_result = probe(high)
    if low_result["aligned_ttft_safe"]:
        raise RuntimeError("source-aware lower bracket is already safe")
    if not high_result["aligned_ttft_safe"]:
        raise RuntimeError("source-aware upper bracket is still unsafe")

    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2.0
        result = probe(mid)
        if result["aligned_ttft_safe"]:
            high = mid
            high_result = result
        else:
            low = mid
            low_result = result
    return low_result, high_result, [cache[key] for key in sorted(cache)]


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    prompt_tokens = int(os.environ["PROMPT_TOKENS"])
    pattern_name = os.environ["HETERO_PATTERN"]
    if prompt_tokens not in PROMPTS:
        raise ValueError(f"PROMPT_TOKENS must be one of {PROMPTS}")
    if pattern_name not in PATTERNS:
        raise ValueError(f"HETERO_PATTERN must be one of {tuple(PATTERNS)}")

    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    current = _current_evaluator_threshold(
        prompt_tokens=prompt_tokens,
        pattern_name=pattern_name,
        profiler=profiler,
        sla=sla,
    )
    source = _source_aware_threshold(
        prompt_tokens=prompt_tokens,
        pattern_name=pattern_name,
        profiler=profiler,
        sla=sla,
    )
    current_checks = [
        _evaluator_probe(
            prompt_tokens=prompt_tokens,
            pattern_name=pattern_name,
            scale_gbps=current["minimum_scale_gbps"] * factor,
            profiler=profiler,
            sla=sla,
        )
        for factor in (0.995, 1.005)
    ]
    unsafe, safe, trials = _helix_threshold_bracket(
        root=root,
        prompt_tokens=prompt_tokens,
        pattern_name=pattern_name,
        center_gbps=source["minimum_scale_gbps"],
        sla=sla,
    )
    midpoint = (unsafe["scale_gbps"] + safe["scale_gbps"]) / 2.0

    result = {
        "design": {
            "purpose": "validate the evaluator normalized-cost network red line on an isolated heterogeneous-link path",
            "controlled_variables": "same LLaMA-2-70B model, A100 8x10-layer pipeline, one request, output_tokens=1, SLA, measured profile, pinned HELIX runtime",
            "changed_variable": "seven fixed link-capacity ratios multiplied by one common scale",
            "evaluator_semantics_changed": False,
            "helix_commit": HELIX_COMMIT,
            "prompt_tokens": prompt_tokens,
            "pattern": pattern_name,
            "link_capacity_ratios": list(PATTERNS[pattern_name]),
            "bisection_steps": BISECTION_STEPS,
            "ttft_semantics": "aligned Prefill completion; true first-token TTFT remains diagnostic",
        },
        "current_evaluator": {
            **current,
            "sanity_checks": current_checks,
        },
        "source_aware_same_redline": source,
        "helix_reference": {
            "unsafe_lower_scale_gbps": unsafe["scale_gbps"],
            "safe_upper_scale_gbps": safe["scale_gbps"],
            "threshold_midpoint_scale_gbps": midpoint,
            "unsafe_aligned_ttft_s": unsafe["aligned_ttft_s"],
            "safe_aligned_ttft_s": safe["aligned_ttft_s"],
            "safe_true_first_token_ttft_s": safe["true_first_token_ttft_s"],
            "safe_max_tpot_s": safe["max_tpot_s"],
        },
        "comparison": {
            "source_prediction_inside_helix_bracket": (
                unsafe["scale_gbps"] < source["minimum_scale_gbps"] <= safe["scale_gbps"]
            ),
            "current_prediction_relative_error_to_helix_midpoint": (
                current["minimum_scale_gbps"] - midpoint
            ) / midpoint,
            "source_prediction_relative_error_to_helix_midpoint": (
                source["minimum_scale_gbps"] - midpoint
            ) / midpoint,
        },
        "trials": trials,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
