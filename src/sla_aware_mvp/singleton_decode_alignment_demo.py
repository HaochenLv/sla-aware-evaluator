from __future__ import annotations

import json
import os
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, SLA
from .evaluator import evaluate
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .workload import build_helix_azure_conversation_workload


START_INTENSITY = 0.004
STOP_INTENSITY = 0.030
STEP_INTENSITY = 0.0002


class SingletonDecodeAlignedProfiler:
    """Conservative wrapper matching HELIX's singleton Decode runtime rule.

    HELIX doubles the interpolated Decode time when one Decode token is in the
    execution batch. The current Conservative evaluator models each request's
    per-token Decode service independently, so treating every profiled Decode as
    a singleton is both runtime-aligned for the isolated case and conservative
    relative to batching multiple Decode tokens.
    """

    def __init__(self, base: HelixA100Llama2Profiler) -> None:
        self.base = base
        self.provenance = base.provenance

    def memory_usage(self, *args, **kwargs):
        return self.base.memory_usage(*args, **kwargs)

    def prefill_time(self, *args, **kwargs):
        return self.base.prefill_time(*args, **kwargs)

    def decode_time_per_token(self, *args, **kwargs):
        return 2.0 * self.base.decode_time_per_token(*args, **kwargs)


def _guard_violation(*, snapshot, request_by_id, pipeline, profiler, sla):
    prefill_ids = [rid for rid, phase in snapshot.request_phase.items() if phase == Phase.PREFILL.value]
    decode_ids = [rid for rid, phase in snapshot.request_phase.items() if phase == Phase.DECODE.value]
    if not prefill_ids or not decode_ids:
        return None
    n_prefill = len(prefill_ids)
    n_decode = len(decode_ids)
    debt_s = sum(
        profiler.prefill_time(request_by_id[rid], pipeline, n_prefill, n_decode)
        for rid in prefill_ids
    )
    for rid in decode_ids:
        base_decode_s = profiler.decode_time_per_token(
            request_by_id[rid], snapshot.request_context[rid], pipeline, n_prefill, n_decode
        )
        required_s = base_decode_s + debt_s + sla.fixed_overhead_s
        if required_s > sla.tpot_s + 1e-9:
            return {
                "time_s": snapshot.time_s,
                "request_id": rid,
                "num_prefill": n_prefill,
                "num_decode": n_decode,
                "base_decode_s": base_decode_s,
                "prefill_debt_s": debt_s,
                "required_s": required_s,
                "tpot_s": sla.tpot_s,
                "active_prefill_ids": prefill_ids,
            }
    return None


def _trial(*, intensity, pipeline, workload, profiler, sla):
    scaled = scale_workload(workload, intensity)
    request_by_id = {request.id: request for request in scaled}
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    guard = None
    for snapshot in run.trace:
        guard = _guard_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            pipeline=pipeline,
            profiler=profiler,
            sla=sla,
        )
        if guard is not None:
            break
    return {
        "intensity": intensity,
        "aligned_evaluator_feasible": run.feasible,
        "aligned_guarded_feasible": run.feasible and guard is None,
        "evaluator_violation": run.first_violation.kind.value if run.first_violation else None,
        "guard_violation": guard,
    }


def _frontier(trials, key):
    safe = [trial for trial in trials if trial[key]]
    unsafe = [trial for trial in trials if not trial[key]]
    first_unsafe = unsafe[0] if unsafe else None
    seen_unsafe = False
    monotonic = True
    for trial in trials:
        if not trial[key]:
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
    raw_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = SingletonDecodeAlignedProfiler(raw_profiler)
    workload_sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = workload_sample.requests
    base_rate = (len(workload) - 1) / (workload[-1].arrival_time_s - workload[0].arrival_time_s)
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    count = round((STOP_INTENSITY - START_INTENSITY) / STEP_INTENSITY)
    intensities = [START_INTENSITY + i * STEP_INTENSITY for i in range(count + 1)]

    result = {
        "design": {
            "decode_rule": "2x raw HELIX Decode profile for every per-request token",
            "reason": "pinned HELIX runtime doubles decode_time when decode_phase_tokens == 1",
            "guard": "runtime-aligned base Decode + sum(active Prefill full profiled compute) + fixed <= TPOT",
            "grid": [START_INTENSITY, STOP_INTENSITY, STEP_INTENSITY],
            "base_arrival_rate_rps": base_rate,
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
        evaluator_frontier = _frontier(trials, "aligned_evaluator_feasible")
        guarded_frontier = _frontier(trials, "aligned_guarded_feasible")
        for frontier in (evaluator_frontier, guarded_frontier):
            frontier["safe_arrival_rate_lower_bound_rps"] = (
                frontier["safe_intensity_lower_bound"] * base_rate
                if frontier["safe_intensity_lower_bound"] is not None else None
            )
            frontier["unsafe_arrival_rate_upper_bound_rps"] = (
                frontier["unsafe_intensity_upper_bound"] * base_rate
                if frontier["unsafe_intensity_upper_bound"] is not None else None
            )
        result["pipelines"][pipeline.id] = {
            "aligned_decode_only": evaluator_frontier,
            "aligned_decode_plus_prefill_guard": guarded_frontier,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
