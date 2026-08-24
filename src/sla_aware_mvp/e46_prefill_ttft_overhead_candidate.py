from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, RequestRuntime, SLA
from .evaluator import _network_bytes_by_link, evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .prefill_debt_budget_ablation import (
    E22_BLOCKING_OVERHEAD_S_PER_TOKEN,
    _budget_trial,
    _decode_link_contribution,
)
from .workload import build_helix_azure_conversation_workload

INTENSITY = 0.0109
SIZE_SCALE = 1.2
TARGET_REQUEST_ID = "azure-00008"


def _scale_sizes(workload, factor: float):
    return tuple(
        replace(
            request,
            input_tokens=max(1, int(round(request.input_tokens * factor))),
            output_tokens=max(1, int(round(request.output_tokens * factor))),
        )
        for request in workload
    )


class PrefillTTFTOverheadProfiler:
    """Research-only E46 candidate.

    Charge the independently calibrated E22-style Prefill intrinsic service
    overhead to Prefill's own TTFT compute-side budget. Decode semantics are
    unchanged. The coefficient remains experiment-local provenance, not a
    universal HELIX constant.
    """

    def __init__(self, base):
        self.base = base
        self.provenance = base.provenance

    def prefill_time(self, request, pipeline, n_prefill, n_decode):
        return self.base.prefill_time(request, pipeline, n_prefill, n_decode) + (
            E22_BLOCKING_OVERHEAD_S_PER_TOKEN * request.input_tokens
        )

    def decode_time_per_token(
        self, request, context_tokens, pipeline, n_prefill, n_decode
    ):
        return self.base.decode_time_per_token(
            request, context_tokens, pipeline, n_prefill, n_decode
        )


def _candidate_budget_violation(*, snapshot, request_by_id, pipeline, profiler, sla):
    """E31 Decode debt with Prefill intrinsic overhead charged exactly once."""
    prefill_ids = [
        rid for rid, phase in snapshot.request_phase.items() if phase == Phase.PREFILL.value
    ]
    decode_ids = [
        rid for rid, phase in snapshot.request_phase.items() if phase == Phase.DECODE.value
    ]
    if not prefill_ids or not decode_ids:
        return None

    n_prefill = len(prefill_ids)
    n_decode = len(decode_ids)
    # profiler.prefill_time already contains the E22-style intrinsic overhead.
    prefill_debt_s = sum(
        profiler.prefill_time(
            request_by_id[rid], pipeline, n_prefill, n_decode
        )
        for rid in prefill_ids
    )

    link_required = dict(snapshot.link_required_bytes_per_s)
    for rid in decode_ids:
        request = request_by_id[rid]
        context = snapshot.request_context[rid]
        decode_s = profiler.decode_time_per_token(
            request, context, pipeline, n_prefill, n_decode
        )
        old_remaining_s = (
            sla.tpot_s - decode_s - sla.queue_overhead_s - sla.fixed_overhead_s
        )
        new_remaining_s = old_remaining_s - prefill_debt_s
        if new_remaining_s <= 1e-9:
            return {
                "time_s": snapshot.time_s,
                "kind": "sla_time",
                "request_id": rid,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "decode_compute_s": decode_s,
                "prefill_blocking_service_debt_s": prefill_debt_s,
                "required_compute_side_s": decode_s
                + prefill_debt_s
                + sla.queue_overhead_s
                + sla.fixed_overhead_s,
                "capacity_s": sla.tpot_s,
            }
        old_contrib = _decode_link_contribution(
            request=request, pipeline=pipeline, remaining_s=old_remaining_s
        )
        new_contrib = _decode_link_contribution(
            request=request, pipeline=pipeline, remaining_s=new_remaining_s
        )
        if old_contrib is None or new_contrib is None:
            continue
        for link_id in link_required:
            link_required[link_id] += new_contrib[link_id] - old_contrib[link_id]

    for link_id, required in link_required.items():
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        if required > capacity + 1e-9:
            return {
                "time_s": snapshot.time_s,
                "kind": "network",
                "link_id": link_id,
                "required_bytes_per_s": required,
                "capacity_bytes_per_s": capacity,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "prefill_blocking_service_debt_s": prefill_debt_s,
            }
    return None


def _candidate_trial(*, intensity, pipeline, workload, profiler, sla):
    scaled = scale_workload(workload, intensity)
    request_by_id = {request.id: request for request in scaled}
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    if not run.feasible:
        violation = run.first_violation
        return {
            "intensity": intensity,
            "feasible": False,
            "source": "base_evaluator_with_prefill_ttft_overhead",
            "first_violation": {
                "kind": violation.kind.value if violation else None,
                "time_s": violation.time_s if violation else None,
                "object_id": violation.object_id if violation else None,
                "request_id": violation.request_id if violation else None,
                "required": violation.required if violation else None,
                "capacity": violation.capacity if violation else None,
                "num_prefill": violation.num_prefill if violation else None,
                "num_decode": violation.num_decode if violation else None,
            },
        }
    for snapshot in run.trace:
        violation = _candidate_budget_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            pipeline=pipeline,
            profiler=profiler,
            sla=sla,
        )
        if violation is not None:
            return {
                "intensity": intensity,
                "feasible": False,
                "source": "e31_decode_blocking_debt",
                "first_violation": violation,
            }
    return {
        "intensity": intensity,
        "feasible": True,
        "source": None,
        "first_violation": None,
    }


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
        interval_offset=0,
    )
    sized_workload = _scale_sizes(sample.requests, SIZE_SCALE)
    scaled_workload = scale_workload(sized_workload, INTENSITY)
    target = next(request for request in scaled_workload if request.id == TARGET_REQUEST_ID)

    pipeline = next(
        item for item in build_helix_pipelines()
        if item.id == "helix-slow-link-placement"
    )
    base_profile = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    base_profiler = ExactHelixDecodeRuntimeProfiler(base_profile)
    candidate_profiler = PrefillTTFTOverheadProfiler(base_profiler)
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)

    baseline = _budget_trial(
        intensity=INTENSITY,
        pipeline=pipeline,
        workload=sized_workload,
        profiler=base_profiler,
        sla=sla,
    )
    candidate = _candidate_trial(
        intensity=INTENSITY,
        pipeline=pipeline,
        workload=sized_workload,
        profiler=candidate_profiler,
        sla=sla,
    )
    helix = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=scaled_workload,
        sla=sla,
        helix_root=root,
    )

    base_compute = base_profiler.prefill_time(target, pipeline, 1, 0)
    overhead = E22_BLOCKING_OVERHEAD_S_PER_TOKEN * target.input_tokens
    demand = _network_bytes_by_link(RequestRuntime(target, Phase.PREFILL), pipeline)
    ideal_network = sum(
        data_bytes / pipeline.links[link_id].capacity_bytes_per_s
        for link_id, data_bytes in demand.items()
        if data_bytes > 0
    )
    remaining_network = (
        sla.ttft_s
        - base_compute
        - overhead
        - sla.queue_overhead_s
        - sla.fixed_overhead_s
    )
    metrics = tuple(helix.query_metrics.values())

    result = {
        "design": {
            "purpose": "first falsification test of E22-style Prefill intrinsic overhead in Prefill's own TTFT budget",
            "candidate_changed": True,
            "production_evaluator_changed": False,
            "network_redline_changed": False,
            "decode_blocking_debt_double_counted": False,
            "coefficient_s_per_token": E22_BLOCKING_OVERHEAD_S_PER_TOKEN,
            "intensity": INTENSITY,
            "size_scale": SIZE_SCALE,
        },
        "target": {
            "request_id": TARGET_REQUEST_ID,
            "input_tokens": target.input_tokens,
            "output_tokens": target.output_tokens,
            "base_profile_compute_s": base_compute,
            "prefill_intrinsic_overhead_s": overhead,
            "ideal_internal_network_s": ideal_network,
            "remaining_network_budget_after_overhead_s": remaining_network,
            "ideal_network_minus_remaining_s": ideal_network - remaining_network,
        },
        "unchanged_e31": baseline,
        "ttft_overhead_candidate": candidate,
        "helix": {
            "feasible": helix.feasible,
            "first_violation_kind": helix.first_violation_kind,
            "first_violation_request_id": helix.first_violation_request_id,
            "first_violation_observed_s": helix.first_violation_observed_s,
            "first_violation_limit_s": helix.first_violation_limit_s,
            "max_aligned_ttft_s": max(item.aligned_ttft_s for item in metrics),
            "max_tpot_s": max(item.max_tpot_s for item in metrics),
        },
        "falsification": {
            "dangerous_optimism": bool(candidate["feasible"] and not helix.feasible),
            "counterexample_closed": bool((not candidate["feasible"]) and (not helix.feasible)),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
