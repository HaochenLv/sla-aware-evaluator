from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import EvaluatorConfig, RequestSpec, SLA
from .evaluator import evaluate
from .helix import HelixA100Llama2Profiler
from .helix_bandwidth_sweep import build_bandwidth_pipeline
from .helix_demo import HELIX_COMMIT
from .helix_fixed_reference import evaluate_helix_fixed_reference


PROMPT_TOKENS = (512, 1024, 1536, 1821)
LOWER_GBPS = 0.5
UPPER_GBPS = 4.0
BISECTION_STEPS = 9


def evaluator_threshold_gbps(
    *, prompt_tokens: int, profiler: HelixA100Llama2Profiler, sla: SLA
) -> tuple[float, float, float]:
    """Closed-form threshold implied by the existing equal-link network red line.

    For one isolated Prefill request on this fixed 8-stage pipeline, every one of
    the seven distinct inter-stage links carries the same activation payload D.
    The evaluator's normalized-cost allocation therefore gives each link 1/7 of
    the remaining TTFT network budget, so per-link feasibility is

        7 D / remaining_time <= link_capacity.

    This is only an algebraic readout of the current evaluator equation; it does
    not change evaluator semantics.
    """
    pipeline = build_bandwidth_pipeline(10.0)
    request = RequestSpec(
        id=f"eval-{prompt_tokens}",
        arrival_time_s=0.0,
        input_tokens=prompt_tokens,
        output_tokens=1,
    )
    compute_s = profiler.prefill_time(request, pipeline, 1, 0)
    remaining_s = sla.ttft_s - compute_s - sla.queue_overhead_s - sla.fixed_overhead_s
    if remaining_s <= 0:
        return float("inf"), compute_s, remaining_s
    data_bytes = pipeline.model.activation_bytes_per_token * prompt_tokens
    link_count = len(pipeline.boundaries)
    required_bytes_per_s = link_count * data_bytes / remaining_s
    threshold_gbps = required_bytes_per_s * 8.0 / 1e9
    return threshold_gbps, compute_s, remaining_s


def run_evaluator_check(
    *,
    prompt_tokens: int,
    bandwidth_gbps: float,
    profiler: HelixA100Llama2Profiler,
    sla: SLA,
) -> dict:
    pipeline = build_bandwidth_pipeline(bandwidth_gbps)
    request = RequestSpec(
        id=f"eval-check-{prompt_tokens}",
        arrival_time_s=0.0,
        input_tokens=prompt_tokens,
        output_tokens=1,
    )
    result = evaluate(
        pipeline=pipeline,
        workload=(request,),
        sla=sla,
        config=EvaluatorConfig(),
        profiler=profiler,
    )
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "feasible": result.feasible,
        "first_violation_kind": (
            result.first_violation.kind.value if result.first_violation else None
        ),
    }


def helix_probe(*, root: Path, prompt_tokens: int, bandwidth_gbps: float, sla: SLA) -> dict:
    pipeline = build_bandwidth_pipeline(bandwidth_gbps)
    request = RequestSpec(
        id=f"helix-{prompt_tokens}",
        arrival_time_s=0.0,
        input_tokens=prompt_tokens,
        output_tokens=1,
    )
    run = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=(request,),
        sla=sla,
        helix_root=root,
    )
    metric = run.query_metrics[request.id]
    aligned_safe = metric.aligned_ttft_s <= sla.ttft_s
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "reference_feasible": run.feasible,
        "aligned_ttft_safe": aligned_safe,
        "aligned_ttft_s": metric.aligned_ttft_s,
        "true_first_token_ttft_s": metric.true_first_token_ttft_s,
        "max_tpot_s": metric.max_tpot_s,
        "first_violation_kind": run.first_violation_kind,
    }


def helix_threshold_bracket(
    *, root: Path, prompt_tokens: int, sla: SLA
) -> tuple[dict, dict, list[dict]]:
    cache: dict[float, dict] = {}

    def probe(bandwidth_gbps: float) -> dict:
        key = round(bandwidth_gbps, 12)
        if key not in cache:
            cache[key] = helix_probe(
                root=root,
                prompt_tokens=prompt_tokens,
                bandwidth_gbps=bandwidth_gbps,
                sla=sla,
            )
        return cache[key]

    low = LOWER_GBPS
    high = UPPER_GBPS
    low_result = probe(low)
    high_result = probe(high)
    if low_result["aligned_ttft_safe"]:
        raise RuntimeError(f"lower bracket already safe for prompt={prompt_tokens}")
    if not high_result["aligned_ttft_safe"]:
        raise RuntimeError(f"upper bracket still unsafe for prompt={prompt_tokens}")

    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2.0
        result = probe(mid)
        if result["aligned_ttft_safe"]:
            high = mid
            high_result = result
        else:
            low = mid
            low_result = result

    trials = [cache[key] for key in sorted(cache)]
    return low_result, high_result, trials


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    prompt_tokens = int(os.environ["PROMPT_TOKENS"])
    if prompt_tokens not in PROMPT_TOKENS:
        raise ValueError(f"PROMPT_TOKENS must be one of {PROMPT_TOKENS}")

    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    predicted_gbps, prefill_compute_s, remaining_s = evaluator_threshold_gbps(
        prompt_tokens=prompt_tokens,
        profiler=profiler,
        sla=sla,
    )
    evaluator_checks = [
        run_evaluator_check(
            prompt_tokens=prompt_tokens,
            bandwidth_gbps=predicted_gbps * factor,
            profiler=profiler,
            sla=sla,
        )
        for factor in (0.995, 1.005)
    ]
    unsafe, safe, trials = helix_threshold_bracket(
        root=root,
        prompt_tokens=prompt_tokens,
        sla=sla,
    )
    midpoint = (unsafe["bandwidth_gbps"] + safe["bandwidth_gbps"]) / 2.0
    bracket_contains_prediction = (
        unsafe["bandwidth_gbps"] < predicted_gbps <= safe["bandwidth_gbps"]
    )
    relative_error_to_midpoint = (predicted_gbps - midpoint) / midpoint

    result = {
        "design": {
            "purpose": "validate the existing evaluator network red-line threshold against isolated HELIX execution",
            "controlled_variables": "same LLaMA-2-70B model, A100 8x10-layer pipeline, HELIX measured profile, one isolated request, output_tokens=1, TTFT/TPOT SLA, pinned HELIX runtime",
            "changed_variable": "prompt length; HELIX bandwidth is bisected only to locate each isolated aligned-TTFT threshold",
            "evaluator_semantics_changed": False,
            "reference_ttft_semantics": "aligned Prefill completion, matching the current Conservative evaluator; true first-token TTFT is diagnostic only",
            "helix_commit": HELIX_COMMIT,
            "bisection_steps": BISECTION_STEPS,
            "initial_bracket_gbps": [LOWER_GBPS, UPPER_GBPS],
        },
        "prompt_tokens": prompt_tokens,
        "evaluator_prediction": {
            "prefill_compute_s": prefill_compute_s,
            "remaining_network_budget_s": remaining_s,
            "activation_bytes_per_token": build_bandwidth_pipeline(10.0).model.activation_bytes_per_token,
            "inter_stage_link_count": 7,
            "minimum_bandwidth_gbps": predicted_gbps,
            "sanity_checks": evaluator_checks,
        },
        "helix_reference": {
            "unsafe_lower_gbps": unsafe["bandwidth_gbps"],
            "safe_upper_gbps": safe["bandwidth_gbps"],
            "threshold_midpoint_gbps": midpoint,
            "unsafe_aligned_ttft_s": unsafe["aligned_ttft_s"],
            "safe_aligned_ttft_s": safe["aligned_ttft_s"],
            "safe_true_first_token_ttft_s": safe["true_first_token_ttft_s"],
            "safe_max_tpot_s": safe["max_tpot_s"],
        },
        "comparison": {
            "helix_bracket_contains_evaluator_prediction": bracket_contains_prediction,
            "relative_error_to_helix_midpoint": relative_error_to_midpoint,
            "absolute_error_to_helix_midpoint_gbps": predicted_gbps - midpoint,
        },
        "trials": trials,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
