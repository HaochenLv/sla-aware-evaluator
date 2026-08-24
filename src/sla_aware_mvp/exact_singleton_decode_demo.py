from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, SLA
from .evaluator import evaluate
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .workload import build_helix_azure_conversation_workload


START_INTENSITY = 0.010
STOP_INTENSITY = 0.030
STEP_INTENSITY = 0.0002
HELIX_FRONTIER_PROBES = (0.0132, 0.0134, 0.0136, 0.0138)


class ExactHelixDecodeRuntimeProfiler:
    """Match the pinned HELIX runtime's singleton-Decode special case exactly.

    The public Decode profile is indexed by active Decode tokens. HELIX first
    interpolates that profile and then doubles decode_time only when the batch
    has exactly one Decode token. For n_decode >= 2 the profile value is used
    unchanged.
    """

    def __init__(self, base: HelixA100Llama2Profiler) -> None:
        self.base = base
        self.provenance = base.provenance

    def memory_usage(self, *args, **kwargs):
        return self.base.memory_usage(*args, **kwargs)

    def prefill_time(self, *args, **kwargs):
        return self.base.prefill_time(*args, **kwargs)

    def decode_time_per_token(
        self, request, context_tokens, pipeline, n_prefill, n_decode
    ):
        raw = self.base.decode_time_per_token(
            request, context_tokens, pipeline, n_prefill, n_decode
        )
        return 2.0 * raw if n_decode == 1 else raw


def _violation_dict(violation):
    if violation is None:
        return None
    return {
        "time_s": violation.time_s,
        "kind": violation.kind.value,
        "object_id": violation.object_id,
        "required": violation.required,
        "capacity": violation.capacity,
        "request_id": violation.request_id,
        "num_prefill": violation.num_prefill,
        "num_decode": violation.num_decode,
    }


def _trial(*, intensity, pipeline, workload, profiler, sla):
    run = evaluate(
        pipeline=pipeline,
        workload=scale_workload(workload, intensity),
        sla=sla,
        config=EvaluatorConfig(),
        profiler=profiler,
    )
    return {
        "intensity": intensity,
        "feasible": run.feasible,
        "first_violation": _violation_dict(run.first_violation),
    }


def _frontier(trials):
    safe = [trial for trial in trials if trial["feasible"]]
    unsafe = [trial for trial in trials if not trial["feasible"]]
    first_unsafe = unsafe[0] if unsafe else None
    seen_unsafe = False
    monotonic = True
    for trial in trials:
        if not trial["feasible"]:
            seen_unsafe = True
        elif seen_unsafe:
            monotonic = False
            break
    return {
        "safe_intensity_lower_bound": safe[-1]["intensity"] if safe else None,
        "unsafe_intensity_upper_bound": first_unsafe["intensity"] if first_unsafe else None,
        "monotonic_on_grid": monotonic,
        "first_unsafe": first_unsafe,
    }


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    workload_sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = workload_sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    count = round((STOP_INTENSITY - START_INTENSITY) / STEP_INTENSITY)
    intensities = [START_INTENSITY + index * STEP_INTENSITY for index in range(count + 1)]

    result = {
        "design": {
            "decode_rule": "2x raw profile iff n_decode == 1; raw profile otherwise",
            "grid": [START_INTENSITY, STOP_INTENSITY, STEP_INTENSITY],
            "base_arrival_rate_rps": base_rate,
            "helix_frontier_probes": list(HELIX_FRONTIER_PROBES),
        },
        "pipelines": {},
    }
    for pipeline in build_helix_pipelines():
        trials = [
            _trial(
                intensity=intensity,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for intensity in intensities
        ]
        frontier = _frontier(trials)
        frontier["safe_arrival_rate_lower_bound_rps"] = (
            frontier["safe_intensity_lower_bound"] * base_rate
            if frontier["safe_intensity_lower_bound"] is not None
            else None
        )
        frontier["unsafe_arrival_rate_upper_bound_rps"] = (
            frontier["unsafe_intensity_upper_bound"] * base_rate
            if frontier["unsafe_intensity_upper_bound"] is not None
            else None
        )
        probe_results = [
            _trial(
                intensity=intensity,
                pipeline=pipeline,
                workload=workload,
                profiler=profiler,
                sla=sla,
            )
            for intensity in HELIX_FRONTIER_PROBES
        ]
        result["pipelines"][pipeline.id] = {
            "frontier": frontier,
            "helix_frontier_probe_results": probe_results,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
