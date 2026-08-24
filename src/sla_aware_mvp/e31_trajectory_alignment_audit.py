from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, SLA
from .evaluator import evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import _build_helix_simulator, _issue_fixed_query, _load_helix_runtime
from .workload import build_helix_azure_conversation_workload


SEED = 19
OFFSET = 0
TARGET_ID = "azure-00009"
INTERFERER_ID = "azure-00010"
INTENSITIES = (0.0153, 0.0183779296875, 0.01838091796875)
MILESTONES = (16, 64, 128, 256, 320, 352, 384, 400)
FIXED_OVERHEAD_S = 0.005
EPS = 1e-9


def _first_ge(progress_rows: list[tuple[float, float]], milestone: int) -> float | None:
    for time_s, progress in progress_rows:
        if progress >= milestone - EPS:
            return time_s
    return None


def _evaluator_timeline(*, pipeline, workload, profiler, intensity: float, fixed_overhead_s: float) -> dict[str, Any]:
    scaled = scale_workload(workload, intensity)
    by_id = {r.id: r for r in scaled}
    target = by_id[TARGET_ID]
    interferer = by_id[INTERFERER_ID]
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=fixed_overhead_s)
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    if not run.feasible:
        raise RuntimeError(
            f"base Evaluator trajectory unexpectedly infeasible at {intensity} fixed={fixed_overhead_s}: {run.first_violation}"
        )

    prefill_to_decode = None
    progress_rows: list[tuple[float, float]] = []
    at_interferer_arrival = None
    finish_time = None
    previous = None
    for snapshot in run.trace:
        phase = snapshot.request_phase.get(TARGET_ID)
        if (
            prefill_to_decode is None
            and phase == Phase.DECODE.value
            and "PrefillToDecode" in snapshot.event_types
        ):
            prefill_to_decode = snapshot.time_s
        if phase == Phase.DECODE.value:
            progress_rows.append((snapshot.time_s, snapshot.request_progress[TARGET_ID]))
        if abs(snapshot.time_s - interferer.arrival_time_s) <= 1e-8 and "Arrival" in snapshot.event_types:
            at_interferer_arrival = {
                "time_s": snapshot.time_s,
                "active": TARGET_ID in snapshot.request_phase,
                "phase": snapshot.request_phase.get(TARGET_ID),
                "progress": snapshot.request_progress.get(TARGET_ID),
                "context": snapshot.request_context.get(TARGET_ID),
                "num_prefill": snapshot.num_prefill,
                "num_decode": snapshot.num_decode,
            }
        if previous is not None:
            if (
                TARGET_ID in previous.request_phase
                and TARGET_ID not in snapshot.request_phase
                and "Finish" in snapshot.event_types
            ):
                finish_time = snapshot.time_s
        previous = snapshot

    if prefill_to_decode is None or finish_time is None or at_interferer_arrival is None:
        raise RuntimeError("missing Evaluator target timeline marker")

    return {
        "fixed_overhead_s_in_progression": fixed_overhead_s,
        "target_arrival_s": target.arrival_time_s,
        "interferer_arrival_s": interferer.arrival_time_s,
        "prefill_finish_s": prefill_to_decode,
        "prefill_latency_s": prefill_to_decode - target.arrival_time_s,
        "finish_s": finish_time,
        "target_lifetime_s": finish_time - target.arrival_time_s,
        "decode_span_s": finish_time - prefill_to_decode,
        "progress_at_interferer_arrival": at_interferer_arrival,
        "milestone_time_s": {
            str(m): _first_ge(progress_rows, m) for m in MILESTONES
        },
        "theoretical_fixed_accumulation_full_decode_s": target.output_tokens * fixed_overhead_s,
        "processed_events": run.processed_events,
    }


def _simulate_to_completion(simulator: Any) -> int:
    events = 0
    while simulator.query_manager.queries_on_the_fly:
        ok, _ = simulator.simulate_next_event()
        events += 1
        if not ok:
            raise RuntimeError("HELIX event queue drained with unfinished queries")
        if events > 5_000_000:
            raise RuntimeError("E38 HELIX event limit exceeded")
    return events


def _helix_timeline(*, root: Path, pipeline, workload, intensity: float) -> dict[str, Any]:
    scaled = scale_workload(workload, intensity)
    by_id = {r.id: r for r in scaled}
    target_spec = by_id[TARGET_ID]
    interferer_spec = by_id[INTERFERER_ID]
    runtime = _load_helix_runtime(root)
    simulator, mini_pipeline, _ = _build_helix_simulator(pipeline=pipeline, runtime=runtime)
    base_time = simulator.current_time
    id_to_uid: dict[str, int] = {}
    for request in scaled:
        uid = _issue_fixed_query(
            simulator=simulator,
            runtime=runtime,
            creation_time=base_time + request.arrival_time_s,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            mini_pipeline=mini_pipeline,
        )
        id_to_uid[request.id] = uid
    events = _simulate_to_completion(simulator)
    query = simulator.query_manager.finished_queries[id_to_uid[TARGET_ID]][1]
    history = list(query.inference_history)
    prefill = history[0]
    decodes = history[1:]
    if len(decodes) != target_spec.output_tokens:
        raise RuntimeError("unexpected HELIX Decode iteration count")

    arrival10_abs = base_time + interferer_spec.arrival_time_s
    completed_at_arrival = sum(item.end_time <= arrival10_abs + EPS for item in decodes)
    started_at_arrival = sum(item.start_time <= arrival10_abs + EPS for item in decodes)
    durations = [item.end_time - item.start_time for item in decodes]
    finish_rel = decodes[-1].end_time - base_time
    prefill_finish_rel = prefill.end_time - base_time
    return {
        "target_arrival_s": target_spec.arrival_time_s,
        "interferer_arrival_s": interferer_spec.arrival_time_s,
        "prefill_finish_s": prefill_finish_rel,
        "prefill_latency_s": prefill_finish_rel - target_spec.arrival_time_s,
        "first_decode_start_s": decodes[0].start_time - base_time,
        "first_decode_end_s": decodes[0].end_time - base_time,
        "finish_s": finish_rel,
        "target_lifetime_s": finish_rel - target_spec.arrival_time_s,
        "decode_span_s": finish_rel - prefill_finish_rel,
        "completed_at_interferer_arrival": completed_at_arrival,
        "started_at_interferer_arrival": started_at_arrival,
        "active_decode_at_interferer_arrival": completed_at_arrival < target_spec.output_tokens,
        "milestone_completion_s": {
            str(m): decodes[m - 1].end_time - base_time for m in MILESTONES
        },
        "mean_raw_decode_iteration_s": sum(durations) / len(durations),
        "max_raw_decode_iteration_s": max(durations),
        "min_raw_decode_iteration_s": min(durations),
        "fixed_overhead_only_used_for_sla_metric_s": FIXED_OVERHEAD_S,
        "processed_events": events,
    }


def _comparison(current: dict[str, Any], no_fixed: dict[str, Any], helix: dict[str, Any]) -> dict[str, Any]:
    current_progress = current["progress_at_interferer_arrival"]["progress"]
    no_fixed_progress = no_fixed["progress_at_interferer_arrival"]["progress"]
    return {
        "prefill_finish_gap_current_minus_helix_s": current["prefill_finish_s"] - helix["prefill_finish_s"],
        "prefill_finish_gap_no_fixed_minus_helix_s": no_fixed["prefill_finish_s"] - helix["prefill_finish_s"],
        "finish_gap_current_minus_helix_s": current["finish_s"] - helix["finish_s"],
        "finish_gap_no_fixed_minus_helix_s": no_fixed["finish_s"] - helix["finish_s"],
        "decode_span_gap_current_minus_helix_s": current["decode_span_s"] - helix["decode_span_s"],
        "decode_span_gap_no_fixed_minus_helix_s": no_fixed["decode_span_s"] - helix["decode_span_s"],
        "evaluator_current_progress_at_interferer_arrival": current_progress,
        "evaluator_no_fixed_progress_at_interferer_arrival": no_fixed_progress,
        "helix_completed_tokens_at_interferer_arrival": helix["completed_at_interferer_arrival"],
        "helix_active_at_interferer_arrival": helix["active_decode_at_interferer_arrival"],
        "current_evaluator_active_at_interferer_arrival": current["progress_at_interferer_arrival"]["active"],
        "no_fixed_evaluator_active_at_interferer_arrival": no_fixed["progress_at_interferer_arrival"]["active"],
        "full_decode_fixed_accumulation_s": current["theoretical_fixed_accumulation_full_decode_s"],
        "finish_gap_reduction_when_fixed_removed_s": (
            current["finish_s"] - helix["finish_s"]
        ) - (
            no_fixed["finish_s"] - helix["finish_s"]
        ),
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
        seed=SEED,
        interval_offset=OFFSET,
    )
    workload = sample.requests
    pipeline = next(
        p for p in build_helix_pipelines() if p.id == "helix-slow-link-placement"
    )
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)

    rows = []
    for intensity in INTENSITIES:
        current = _evaluator_timeline(
            pipeline=pipeline,
            workload=workload,
            profiler=profiler,
            intensity=intensity,
            fixed_overhead_s=FIXED_OVERHEAD_S,
        )
        no_fixed = _evaluator_timeline(
            pipeline=pipeline,
            workload=workload,
            profiler=profiler,
            intensity=intensity,
            fixed_overhead_s=0.0,
        )
        helix = _helix_timeline(
            root=root,
            pipeline=pipeline,
            workload=workload,
            intensity=intensity,
        )
        rows.append(
            {
                "intensity": intensity,
                "evaluator_current": current,
                "evaluator_progress_only_counterfactual": no_fixed,
                "helix_runtime": helix,
                "comparison": _comparison(current, no_fixed, helix),
            }
        )

    result = {
        "design": {
            "purpose": "separate seed19 Conservative-vs-HELIX trajectory mismatch from Prefill exposure timing",
            "victim_request": TARGET_ID,
            "interferer_request": INTERFERER_ID,
            "pipeline": pipeline.id,
            "intensities": list(INTENSITIES),
            "fixed_overhead_s": FIXED_OVERHEAD_S,
            "current_evaluator_progression": "profiled compute + queue overhead + fixed overhead",
            "counterfactual_progression": "same Evaluator with fixed overhead set to zero; observational trajectory audit only",
            "helix_runtime": "raw HELIX execution; fixed overhead is added only to extracted SLA metric, not runtime progression",
            "production_evaluator_changed": False,
            "network_redline_changed": False,
        },
        "rows": rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
