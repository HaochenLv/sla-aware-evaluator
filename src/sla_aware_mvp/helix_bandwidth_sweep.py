from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import Boundary, GPUNode, Link, Pipeline, SLA, Stage
from .helix_bandwidth_ordering_validation import _capacity_summary
from .helix_demo import HELIX_COMMIT, build_helix_model
from .workload import build_helix_azure_conversation_workload


BANDWIDTHS_GBPS = (1.25, 2.5, 5.0, 10.0, 20.0)


def build_bandwidth_pipeline(gbps: float) -> Pipeline:
    """Build the fixed 8x10-layer A100 pipeline with one controlled link bandwidth."""
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
            capacity_bytes_per_s=gbps * 1e9 / 8.0,
        )
        for index in range(7)
    }
    boundaries = tuple(
        Boundary(f"s{index}", f"s{index + 1}", (f"link-{index}",))
        for index in range(7)
    )
    pipeline = Pipeline(f"bandwidth-{gbps:g}gbps", model, nodes, links, stages, boundaries)
    pipeline.validate()
    return pipeline


def _adjacent_relation(lower: dict, higher: dict) -> str:
    if lower.get("status") != "ok" or higher.get("status") != "ok":
        return "unresolved_search_failure"
    lower_safe = lower["safe_intensity_lower_bound"]
    lower_unsafe = lower["unsafe_intensity_upper_bound"]
    higher_safe = higher["safe_intensity_lower_bound"]
    higher_unsafe = higher["unsafe_intensity_upper_bound"]
    if lower_unsafe is not None and lower_unsafe < higher_safe:
        return "resolved_increase"
    if higher_unsafe is not None and higher_unsafe < lower_safe:
        return "resolved_decrease"
    return "overlapping_brackets"


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    offset = int(os.environ.get("VALIDATION_OFFSET", "0"))

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
        interval_offset=offset,
    )
    workload = sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)

    summaries: dict[str, dict] = {}
    ordered: list[tuple[float, dict]] = []
    for gbps in BANDWIDTHS_GBPS:
        pipeline = build_bandwidth_pipeline(gbps)
        summary = _capacity_summary(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            root=root,
            base_rate=base_rate,
        )
        summaries[pipeline.id] = summary
        ordered.append((gbps, summary))

    adjacent = []
    for (lower_bw, lower), (higher_bw, higher) in zip(ordered, ordered[1:]):
        adjacent.append(
            {
                "lower_gbps": lower_bw,
                "higher_gbps": higher_bw,
                "relation": _adjacent_relation(lower, higher),
                "lower_safe_intensity": lower.get("safe_intensity_lower_bound"),
                "higher_safe_intensity": higher.get("safe_intensity_lower_bound"),
            }
        )

    ok_summaries = [summary for _, summary in ordered if summary.get("status") == "ok"]
    safe_nondecreasing = len(ok_summaries) == len(ordered) and all(
        current[1]["safe_intensity_lower_bound"]
        <= following[1]["safe_intensity_lower_bound"]
        for current, following in zip(ordered, ordered[1:])
    )
    resolved_decrease = any(item["relation"] == "resolved_decrease" for item in adjacent)

    input_tokens = [request.input_tokens for request in workload]
    output_tokens = [request.output_tokens for request in workload]
    result = {
        "design": {
            "purpose": "measure HELIX fixed-pipeline capacity sensitivity to inter-stage bandwidth",
            "controlled_variables": "same model, A100 nodes, 8x10-layer placement, node order, workload content, SLA, and HELIX runtime",
            "changed_variable": "all seven inter-stage links swept together across 1.25/2.5/5/10/20 Gbps",
            "reference_ttft_semantics": "aligned Prefill completion retained; true first-token TTFT is diagnostic only",
            "helix_commit": HELIX_COMMIT,
            "interval_offset": offset,
            "seed": 7,
            "bandwidths_gbps": list(BANDWIDTHS_GBPS),
            "sla": {
                "ttft_s": sla.ttft_s,
                "tpot_s": sla.tpot_s,
                "fixed_overhead_s": sla.fixed_overhead_s,
            },
        },
        "workload": {
            "request_count": len(workload),
            "base_arrival_rate_rps": base_rate,
            "mean_input_tokens": sum(input_tokens) / len(input_tokens),
            "mean_output_tokens": sum(output_tokens) / len(output_tokens),
            "max_input_tokens": max(input_tokens),
            "max_output_tokens": max(output_tokens),
        },
        "pipelines": summaries,
        "adjacent_bandwidth_relations": adjacent,
        "safe_lower_bound_nondecreasing": safe_nondecreasing,
        "any_resolved_capacity_decrease": resolved_decrease,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
