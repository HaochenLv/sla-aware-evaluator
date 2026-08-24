from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import Boundary, GPUNode, Link, Pipeline, SLA, Stage
from .helix_demo import HELIX_COMMIT, build_helix_model
from .helix_fixed_capacity import find_helix_fixed_capacity
from .workload import build_helix_azure_conversation_workload


SLOW_GBPS = 2.5
FAST_GBPS = 10.0
INITIAL_INTENSITY = 0.015
TOLERANCE = 0.0001
MAX_INTENSITY = 0.05


def build_bandwidth_control_pipelines() -> tuple[Pipeline, Pipeline]:
    """Build two pipelines that differ only in inter-stage link bandwidth."""
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

    def make_pipeline(identifier: str, gbps: float) -> Pipeline:
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
        pipeline = Pipeline(identifier, model, nodes, links, stages, boundaries)
        pipeline.validate()
        return pipeline

    return (
        make_pipeline("bandwidth-2.5gbps", SLOW_GBPS),
        make_pipeline("bandwidth-10gbps", FAST_GBPS),
    )


def _max_metric(run, name: str) -> float | None:
    if run is None or not run.query_metrics:
        return None
    return max(getattr(metric, name) for metric in run.query_metrics.values())


def _capacity_summary(*, pipeline, workload, sla, root, base_rate):
    try:
        result = find_helix_fixed_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            helix_root=root,
            initial_intensity=INITIAL_INTENSITY,
            tolerance=TOLERANCE,
            max_intensity=MAX_INTENSITY,
        )
    except RuntimeError as exc:
        return {
            "status": "search_failed",
            "error": str(exc),
        }

    return {
        "status": "ok",
        "safe_intensity_lower_bound": result.safe_intensity,
        "unsafe_intensity_upper_bound": result.unsafe_intensity,
        "safe_arrival_rate_lower_bound_rps": result.safe_intensity * base_rate,
        "unsafe_arrival_rate_upper_bound_rps": (
            result.unsafe_intensity * base_rate
            if result.unsafe_intensity is not None
            else None
        ),
        "right_censored": result.right_censored,
        "safe_max_aligned_ttft_s": _max_metric(result.safe_run, "aligned_ttft_s"),
        "safe_max_true_ttft_s": _max_metric(result.safe_run, "true_first_token_ttft_s"),
        "safe_max_tpot_s": _max_metric(result.safe_run, "max_tpot_s"),
        "unsafe_violation_kind": (
            result.unsafe_run.first_violation_kind if result.unsafe_run is not None else None
        ),
        "unsafe_max_tpot_s": _max_metric(result.unsafe_run, "max_tpot_s"),
        "trials": [
            {
                "intensity": trial.intensity,
                "feasible": trial.feasible,
                "violation_kind": trial.violation_kind,
            }
            for trial in sorted(result.trials, key=lambda item: item.intensity)
        ],
    }


def _ordering(slow: dict, fast: dict) -> str:
    if slow.get("status") != "ok" or fast.get("status") != "ok":
        return "unresolved_search_failure"
    slow_safe = slow["safe_intensity_lower_bound"]
    slow_unsafe = slow["unsafe_intensity_upper_bound"]
    fast_safe = fast["safe_intensity_lower_bound"]
    fast_unsafe = fast["unsafe_intensity_upper_bound"]
    if slow_unsafe is not None and slow_unsafe < fast_safe:
        return "fast>slow"
    if fast_unsafe is not None and fast_unsafe < slow_safe:
        return "slow>fast"
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
    slow_pipeline, fast_pipeline = build_bandwidth_control_pipelines()

    slow = _capacity_summary(
        pipeline=slow_pipeline,
        workload=workload,
        sla=sla,
        root=root,
        base_rate=base_rate,
    )
    fast = _capacity_summary(
        pipeline=fast_pipeline,
        workload=workload,
        sla=sla,
        root=root,
        base_rate=base_rate,
    )

    output_tokens = [request.output_tokens for request in workload]
    input_tokens = [request.input_tokens for request in workload]
    result = {
        "design": {
            "purpose": "measure HELIX Slow/Fast capacity-order stability when only link bandwidth changes",
            "controlled_variables": "same model, A100 nodes, 8x10-layer placement, node order, workload content, SLA, and HELIX runtime",
            "changed_variable": "all seven inter-stage links: 2.5 Gbps versus 10 Gbps",
            "reference_ttft_semantics": "aligned Prefill completion, retained to isolate bandwidth timing; true first-token TTFT is diagnostic only",
            "helix_commit": HELIX_COMMIT,
            "capacity_search": {
                "initial_intensity": INITIAL_INTENSITY,
                "tolerance": TOLERANCE,
                "max_intensity": MAX_INTENSITY,
            },
            "interval_offset": offset,
            "seed": 7,
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
        "pipelines": {
            slow_pipeline.id: slow,
            fast_pipeline.id: fast,
        },
        "pairwise_ordering": _ordering(slow, fast),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
