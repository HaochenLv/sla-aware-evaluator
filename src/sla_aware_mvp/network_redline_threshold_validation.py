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


HELIX_SEARCH_LOW_GBPS = 0.5
HELIX_SEARCH_HIGH_GBPS = 4.0
HELIX_SEARCH_TOLERANCE_GBPS = 0.002


def _request(prompt_tokens: int) -> RequestSpec:
    return RequestSpec(
        id=f"isolated-prompt-{prompt_tokens}",
        arrival_time_s=0.0,
        input_tokens=prompt_tokens,
        output_tokens=1,
    )


def _evaluator_redline_threshold_gbps(*, request, pipeline, profiler, sla) -> dict:
    compute_s = profiler.prefill_time(request, pipeline, 1, 0)
    remaining_s = (
        sla.ttft_s - compute_s - sla.queue_overhead_s - sla.fixed_overhead_s
    )
    if remaining_s <= 0:
        return {
            "status": "compute_time_infeasible",
            "prefill_compute_s": compute_s,
            "remaining_network_budget_s": remaining_s,
            "threshold_gbps": None,
        }

    # For this controlled pipeline every stage boundary carries the same Prompt
    # activation volume over one link, and all seven links are swept together.
    # The current Evaluator allocates the residual SLA window in proportion to
    # D_e / B_e. Equal D_e and equal B_e therefore give 1/7 of the residual
    # window to each boundary, so the exact red-line threshold is 7D/Delta.
    data_bytes_per_boundary = (
        pipeline.model.activation_bytes_per_token * request.input_tokens
    )
    boundary_count = len(pipeline.boundaries)
    required_bytes_per_s = (
        boundary_count * data_bytes_per_boundary / remaining_s
    )
    threshold_gbps = required_bytes_per_s * 8.0 / 1e9
    return {
        "status": "ok",
        "prefill_compute_s": compute_s,
        "remaining_network_budget_s": remaining_s,
        "activation_bytes_per_boundary": data_bytes_per_boundary,
        "boundary_count": boundary_count,
        "required_bytes_per_s_per_link": required_bytes_per_s,
        "threshold_gbps": threshold_gbps,
    }


def _evaluator_probe(*, request, bandwidth_gbps, profiler, sla) -> dict:
    pipeline = build_bandwidth_pipeline(bandwidth_gbps)
    run = evaluate(
        pipeline=pipeline,
        workload=(request,),
        sla=sla,
        config=EvaluatorConfig(),
        profiler=profiler,
    )
    violation = run.first_violation
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "feasible": run.feasible,
        "violation_kind": violation.kind.value if violation is not None else None,
        "violation_object": violation.object_id if violation is not None else None,
    }


def _helix_probe(*, request, bandwidth_gbps, sla, root) -> dict:
    pipeline = build_bandwidth_pipeline(bandwidth_gbps)
    run = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=(request,),
        sla=sla,
        helix_root=root,
    )
    metric = run.query_metrics[request.id]
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "feasible": run.feasible,
        "violation_kind": run.first_violation_kind,
        "aligned_ttft_s": metric.aligned_ttft_s,
        "true_first_token_ttft_s": metric.true_first_token_ttft_s,
        "max_tpot_s": metric.max_tpot_s,
    }


def _helix_threshold(*, request, sla, root) -> dict:
    low = HELIX_SEARCH_LOW_GBPS
    high = HELIX_SEARCH_HIGH_GBPS
    low_probe = _helix_probe(
        request=request, bandwidth_gbps=low, sla=sla, root=root
    )
    high_probe = _helix_probe(
        request=request, bandwidth_gbps=high, sla=sla, root=root
    )
    if low_probe["feasible"]:
        raise RuntimeError("HELIX lower search bound is already feasible")
    if not high_probe["feasible"]:
        raise RuntimeError("HELIX upper search bound is still infeasible")

    probes = [low_probe, high_probe]
    while high - low > HELIX_SEARCH_TOLERANCE_GBPS:
        mid = (low + high) / 2.0
        probe = _helix_probe(
            request=request, bandwidth_gbps=mid, sla=sla, root=root
        )
        probes.append(probe)
        if probe["feasible"]:
            high = mid
            high_probe = probe
        else:
            low = mid
            low_probe = probe

    return {
        "unsafe_lower_bound_gbps": low,
        "safe_upper_bound_gbps": high,
        "threshold_midpoint_gbps": (low + high) / 2.0,
        "unsafe_probe": low_probe,
        "safe_probe": high_probe,
        "probe_count": len(probes),
    }


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    prompt_tokens = int(os.environ["PROMPT_TOKENS"])

    request = _request(prompt_tokens)
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    nominal_pipeline = build_bandwidth_pipeline(10.0)

    redline = _evaluator_redline_threshold_gbps(
        request=request,
        pipeline=nominal_pipeline,
        profiler=profiler,
        sla=sla,
    )
    if redline["status"] != "ok":
        raise RuntimeError("Evaluator compute budget is already infeasible")
    predicted = redline["threshold_gbps"]
    evaluator_checks = {
        "below_predicted": _evaluator_probe(
            request=request,
            bandwidth_gbps=predicted * 0.995,
            profiler=profiler,
            sla=sla,
        ),
        "above_predicted": _evaluator_probe(
            request=request,
            bandwidth_gbps=predicted * 1.005,
            profiler=profiler,
            sla=sla,
        ),
    }

    helix = _helix_threshold(request=request, sla=sla, root=root)
    helix_mid = helix["threshold_midpoint_gbps"]
    relative_error = (predicted - helix_mid) / helix_mid

    result = {
        "design": {
            "purpose": "directly validate the existing Evaluator Prompt network red-line threshold against isolated HELIX execution",
            "controlled_variables": "same LLaMA-2-70B model, 8x10-layer A100 pipeline, one isolated request, one Decode token, SLA, and pinned HELIX runtime/profile",
            "changed_variable": "prompt length; for each prompt all seven inter-stage links are swept together",
            "no_queueing_or_overlap": True,
            "reference_ttft_semantics": "aligned Prefill completion, matching current Evaluator semantics; true first-token TTFT remains diagnostic",
            "helix_commit": HELIX_COMMIT,
            "helix_search_tolerance_gbps": HELIX_SEARCH_TOLERANCE_GBPS,
        },
        "request": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": request.output_tokens,
        },
        "evaluator_redline": redline,
        "evaluator_boundary_checks": evaluator_checks,
        "helix_aligned_ttft_threshold": helix,
        "comparison": {
            "evaluator_predicted_threshold_gbps": predicted,
            "helix_threshold_midpoint_gbps": helix_mid,
            "signed_relative_error": relative_error,
            "absolute_relative_error": abs(relative_error),
            "evaluator_is_more_conservative": predicted >= helix["safe_upper_bound_gbps"],
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
