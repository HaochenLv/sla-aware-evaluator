from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .exact_singleton_prefill_guard_demo import _frontier
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .prefill_debt_budget_ablation import _budget_trial
from .workload import build_helix_azure_conversation_workload

CANDIDATE_START = 0.008
CANDIDATE_STOP = 0.025
CANDIDATE_STEP = 0.0001
BUFFER_FACTOR = 1.05


def _workload_stats(workload):
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    return {
        "request_count": len(workload),
        "base_arrival_rate_rps": base_rate,
        "input_mean": sum(r.input_tokens for r in workload) / len(workload),
        "output_mean": sum(r.output_tokens for r in workload) / len(workload),
        "max_input": max(r.input_tokens for r in workload),
        "max_output": max(r.output_tokens for r in workload),
    }


def _helix_probe(*, intensity, pipeline, workload, sla, root):
    run = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=scale_workload(workload, intensity),
        sla=sla,
        helix_root=root,
    )
    metrics = tuple(run.query_metrics.values())
    return {
        "intensity": intensity,
        "feasible": run.feasible,
        "first_violation_kind": run.first_violation_kind,
        "max_aligned_ttft_s": max(item.aligned_ttft_s for item in metrics),
        "max_tpot_s": max(item.max_tpot_s for item in metrics),
    }


def _classification(*, at_safe, at_unsafe):
    if not at_safe["feasible"]:
        return "observed_candidate_optimism"
    if at_unsafe["feasible"]:
        return "candidate_conservative_at_frontier"
    return "frontiers_overlap_within_candidate_grid_step"


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    seed = int(os.environ.get("VALIDATION_SEED", "3"))
    interval_offset = int(os.environ.get("VALIDATION_OFFSET", "0"))

    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=seed,
        interval_offset=interval_offset,
    )
    workload = sample.requests
    stats = _workload_stats(workload)
    base_rate = stats["base_arrival_rate_rps"]
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)

    count = round((CANDIDATE_STOP - CANDIDATE_START) / CANDIDATE_STEP)
    intensities = [CANDIDATE_START + i * CANDIDATE_STEP for i in range(count + 1)]

    result = {
        "design": {
            "purpose": "validate E31 blocking-service-debt budget candidate across held-out workload variants",
            "seed": seed,
            "interval_offset": interval_offset,
            "candidate_grid": [CANDIDATE_START, CANDIDATE_STOP, CANDIDATE_STEP],
            "budget_formula": "Delta_D=tau_D-T_decode-I_blocking-T_queue-T_fix",
            "candidate_debt": "sum(profiled Prefill compute + E22 blocking overhead) over active Prefills",
            "state_progression_changed": False,
            "network_redline_changed": False,
            "production_evaluator_changed": False,
        },
        "workload": stats,
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        trials = [
            _budget_trial(
                intensity=value,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for value in intensities
        ]
        frontier = _frontier(trials)
        safe = frontier["safe_intensity_lower_bound"]
        unsafe = frontier["unsafe_intensity_upper_bound"]
        if safe is None or unsafe is None:
            raise RuntimeError("E31 frontier not bracketed by validation grid")
        frontier["safe_arrival_rate_lower_bound_rps"] = safe * base_rate
        frontier["unsafe_arrival_rate_upper_bound_rps"] = unsafe * base_rate

        probes = {
            "at_candidate_safe": _helix_probe(
                intensity=safe,
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                root=root,
            ),
            "at_candidate_unsafe": _helix_probe(
                intensity=unsafe,
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                root=root,
            ),
            "at_5pct_above_candidate_unsafe": _helix_probe(
                intensity=unsafe * BUFFER_FACTOR,
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                root=root,
            ),
        }
        result["pipelines"][pipeline.id] = {
            "candidate_frontier": frontier,
            "helix_probes": probes,
            "classification": _classification(
                at_safe=probes["at_candidate_safe"],
                at_unsafe=probes["at_candidate_unsafe"],
            ),
            "buffer_still_helix_safe": probes["at_5pct_above_candidate_unsafe"]["feasible"],
        }

    result["observed_candidate_optimism"] = any(
        item["classification"] == "observed_candidate_optimism"
        for item in result["pipelines"].values()
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
