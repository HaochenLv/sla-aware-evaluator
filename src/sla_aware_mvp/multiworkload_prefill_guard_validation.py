from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import SLA
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .exact_singleton_prefill_guard_demo import _frontier, _trial
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .workload import build_helix_azure_conversation_workload


CANDIDATE_START = 0.008
CANDIDATE_STOP = 0.025
CANDIDATE_STEP = 0.0001
BUFFER_FACTOR = 1.05


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


def _helix_probe(*, intensity, pipeline, workload, sla, root):
    run = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=scale_workload(workload, intensity),
        sla=sla,
        helix_root=root,
    )
    return {
        "intensity": intensity,
        "feasible": run.feasible,
        "violation_kind": run.first_violation_kind,
        "max_aligned_ttft_s": max(item.aligned_ttft_s for item in run.query_metrics),
        "max_true_ttft_s": max(item.true_first_token_ttft_s for item in run.query_metrics),
        "max_tpot_s": max(
            (max(item.decode_tpot_s) if item.decode_tpot_s else 0.0)
            for item in run.query_metrics
        ),
    }


def _classification(*, candidate_safe_probe, candidate_unsafe_probe):
    if not candidate_safe_probe["feasible"]:
        return "observed_candidate_optimism"
    if candidate_unsafe_probe["feasible"]:
        return "candidate_conservative_at_its_frontier"
    return "candidate_and_helix_frontiers_overlap_within_one_candidate_grid_step"


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
            "helix_probe_policy": "probe candidate safe edge, candidate unsafe edge, and 5% above unsafe edge",
            "seed": seed,
            "interval_offset": interval_offset,
            "sla": {
                "ttft_s": sla.ttft_s,
                "tpot_s": sla.tpot_s,
                "fixed_overhead_s": sla.fixed_overhead_s,
            },
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
        safe_intensity = candidate_frontier["safe_intensity_lower_bound"]
        unsafe_intensity = candidate_frontier["unsafe_intensity_upper_bound"]
        if safe_intensity is None or unsafe_intensity is None:
            raise RuntimeError("candidate frontier not bracketed by validation grid")

        candidate_frontier["safe_arrival_rate_lower_bound_rps"] = safe_intensity * base_rate
        candidate_frontier["unsafe_arrival_rate_upper_bound_rps"] = unsafe_intensity * base_rate

        buffer_intensity = unsafe_intensity * BUFFER_FACTOR
        helix_safe_edge = _helix_probe(
            intensity=safe_intensity,
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            root=root,
        )
        helix_unsafe_edge = _helix_probe(
            intensity=unsafe_intensity,
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            root=root,
        )
        helix_buffer = _helix_probe(
            intensity=buffer_intensity,
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            root=root,
        )

        result["pipelines"][pipeline.id] = {
            "candidate": candidate_frontier,
            "helix_probes": {
                "at_candidate_safe": helix_safe_edge,
                "at_candidate_unsafe": helix_unsafe_edge,
                "at_5pct_above_candidate_unsafe": helix_buffer,
            },
            "comparison": {
                "classification": _classification(
                    candidate_safe_probe=helix_safe_edge,
                    candidate_unsafe_probe=helix_unsafe_edge,
                ),
                "candidate_unsafe_to_buffer_span": buffer_intensity - unsafe_intensity,
                "buffer_is_still_helix_safe": helix_buffer["feasible"],
            },
        }

    pipeline_items = list(result["pipelines"].items())
    if len(pipeline_items) == 2:
        (_, first), (_, second) = pipeline_items
        first_safe = first["candidate"]["safe_intensity_lower_bound"]
        second_safe = second["candidate"]["safe_intensity_lower_bound"]
        result["pairwise"] = {
            "candidate_safe_order": (
                "first>second" if first_safe > second_safe else "first<second" if first_safe < second_safe else "tie"
            ),
            "helix_at_candidate_unsafe": {
                "first_feasible": first["helix_probes"]["at_candidate_unsafe"]["feasible"],
                "second_feasible": second["helix_probes"]["at_candidate_unsafe"]["feasible"],
            },
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
