from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import SLA
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .exact_singleton_prefill_guard_demo import _frontier, _trial
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_capacity import find_helix_fixed_capacity
from .workload import build_helix_azure_conversation_workload


CANDIDATE_START = 0.008
CANDIDATE_STOP = 0.025
CANDIDATE_STEP = 0.0001
HELIX_TOLERANCE = 0.0001
HELIX_INITIAL_INTENSITY = 0.013
HELIX_MAX_INTENSITY = 0.10


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def _workload_stats(workload):
    inputs = [request.input_tokens for request in workload]
    outputs = [request.output_tokens for request in workload]
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    return {
        "request_count": len(workload),
        "base_arrival_rate_rps": base_rate,
        "input_tokens": {
            "mean": sum(inputs) / len(inputs),
            "p50": _percentile(inputs, 0.50),
            "p90": _percentile(inputs, 0.90),
            "max": max(inputs),
        },
        "output_tokens": {
            "mean": sum(outputs) / len(outputs),
            "p50": _percentile(outputs, 0.50),
            "p90": _percentile(outputs, 0.90),
            "max": max(outputs),
        },
    }


def _relation(candidate_frontier, helix_result):
    cand_safe = candidate_frontier["safe_intensity_lower_bound"]
    cand_unsafe = candidate_frontier["unsafe_intensity_upper_bound"]
    helix_safe = helix_result.safe_intensity
    helix_unsafe = helix_result.unsafe_intensity

    if cand_unsafe is not None and cand_unsafe <= helix_safe + 1e-12:
        return "candidate_frontier_left_of_known_helix_safe"
    if helix_unsafe is not None and cand_safe is not None and cand_safe >= helix_unsafe - 1e-12:
        return "observed_candidate_optimism"
    return "frontier_brackets_overlap_or_are_inconclusive"


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
    candidate_intensities = [
        CANDIDATE_START + index * CANDIDATE_STEP for index in range(count + 1)
    ]

    result = {
        "design": {
            "purpose": "validate exact singleton Decode + full active-Prefill debt across workload variants",
            "candidate_grid": [CANDIDATE_START, CANDIDATE_STOP, CANDIDATE_STEP],
            "helix_capacity_search": {
                "initial_intensity": HELIX_INITIAL_INTENSITY,
                "tolerance": HELIX_TOLERANCE,
                "max_intensity": HELIX_MAX_INTENSITY,
            },
            "seed": seed,
            "interval_offset": interval_offset,
            "sla": {"ttft_s": sla.ttft_s, "tpot_s": sla.tpot_s, "fixed_overhead_s": sla.fixed_overhead_s},
        },
        "workload": stats,
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        candidate_trials = [
            _trial(
                intensity=intensity,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for intensity in candidate_intensities
        ]
        candidate_frontier = _frontier(candidate_trials)
        candidate_frontier["safe_arrival_rate_lower_bound_rps"] = (
            candidate_frontier["safe_intensity_lower_bound"] * base_rate
            if candidate_frontier["safe_intensity_lower_bound"] is not None
            else None
        )
        candidate_frontier["unsafe_arrival_rate_upper_bound_rps"] = (
            candidate_frontier["unsafe_intensity_upper_bound"] * base_rate
            if candidate_frontier["unsafe_intensity_upper_bound"] is not None
            else None
        )

        helix = find_helix_fixed_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            helix_root=root,
            initial_intensity=HELIX_INITIAL_INTENSITY,
            tolerance=HELIX_TOLERANCE,
            max_intensity=HELIX_MAX_INTENSITY,
        )
        helix_summary = {
            "safe_intensity_lower_bound": helix.safe_intensity,
            "unsafe_intensity_upper_bound": helix.unsafe_intensity,
            "safe_arrival_rate_lower_bound_rps": helix.safe_intensity * base_rate,
            "unsafe_arrival_rate_upper_bound_rps": (
                helix.unsafe_intensity * base_rate if helix.unsafe_intensity is not None else None
            ),
            "right_censored": helix.right_censored,
            "trials": [
                {
                    "intensity": trial.intensity,
                    "feasible": trial.feasible,
                    "violation_kind": trial.violation_kind,
                }
                for trial in helix.trials
            ],
        }

        cand_safe = candidate_frontier["safe_intensity_lower_bound"]
        gap = helix.safe_intensity - cand_safe if cand_safe is not None else None
        result["pipelines"][pipeline.id] = {
            "candidate": candidate_frontier,
            "helix": helix_summary,
            "comparison": {
                "relation": _relation(candidate_frontier, helix),
                "safe_intensity_gap_helix_minus_candidate": gap,
                "candidate_to_helix_safe_ratio": (
                    cand_safe / helix.safe_intensity
                    if cand_safe is not None and helix.safe_intensity > 0
                    else None
                ),
            },
        }

    pipeline_items = list(result["pipelines"].items())
    if len(pipeline_items) == 2:
        (_, first), (_, second) = pipeline_items
        result["pairwise"] = {
            "candidate_safe_order": (
                "first>second"
                if first["candidate"]["safe_intensity_lower_bound"] > second["candidate"]["safe_intensity_lower_bound"]
                else "first<second"
                if first["candidate"]["safe_intensity_lower_bound"] < second["candidate"]["safe_intensity_lower_bound"]
                else "tie"
            ),
            "helix_safe_order": (
                "first>second"
                if first["helix"]["safe_intensity_lower_bound"] > second["helix"]["safe_intensity_lower_bound"]
                else "first<second"
                if first["helix"]["safe_intensity_lower_bound"] < second["helix"]["safe_intensity_lower_bound"]
                else "tie"
            ),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
