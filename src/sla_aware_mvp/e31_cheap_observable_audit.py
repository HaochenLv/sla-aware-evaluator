from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .capacity import scale_workload
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import _build_helix_simulator, _issue_fixed_query, _load_helix_runtime
from .seed19_tightness_diagnostic import QueueTracer
from .workload import build_helix_azure_conversation_workload


BLOCK_SIZE = 16
TARGET_INPUT = 256
TARGET_OUTPUT = 64
INTERFERER_OUTPUT = 1
CONTROLLED = (
    ("helix-fast-link-placement", 256, 0.01),
    ("helix-fast-link-placement", 256, 0.08),
    ("helix-fast-link-placement", 256, 0.12),
    ("helix-slow-link-placement", 1024, 0.01),
    ("helix-slow-link-placement", 1024, 0.08),
    ("helix-slow-link-placement", 1024, 0.12),
)
SEED19_POINTS = (0.0153, 0.0183779296875, 0.01838091796875)
EPS = 1e-9


def _simulate(simulator: Any, limit: int = 5_000_000) -> int:
    events = 0
    while simulator.query_manager.queries_on_the_fly:
        ok, _ = simulator.simulate_next_event()
        events += 1
        if not ok:
            raise RuntimeError("HELIX event queue drained with unfinished queries")
        if events > limit:
            raise RuntimeError("E37 event limit exceeded")
    return events


def _prefill_service(tracer: QueueTracer, query_uid: int) -> float:
    total = 0.0
    for batch in tracer.batch_intervals:
        matched = any(
            uid == query_uid and phase == "Initialization"
            for uid, phase in zip(batch.base_query_uids, batch.phases)
        )
        if matched:
            total += batch.duration_s
    return total


def _request_observables(*, query: Any, arrival_s: float, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    decodes = list(query.inference_history[1:])
    completed = sum(item.end_time <= arrival_s + EPS for item in decodes)
    started = sum(item.start_time <= arrival_s + EPS for item in decodes)
    in_flight = any(item.start_time < arrival_s < item.end_time for item in decodes)
    return {
        "event_type": "Arrival",
        "prefill_age_s": 0.0,
        "decode_completed_tokens": completed,
        "decode_started_tokens": started,
        "decode_block_index": completed // BLOCK_SIZE,
        "decode_context_tokens": input_tokens + completed,
        "decode_remaining_tokens": max(output_tokens - completed, 0),
        "decode_remaining_fraction": max(output_tokens - completed, 0) / output_tokens,
        "decode_iteration_in_flight_at_arrival": in_flight,
    }


def _max_target_wait(*, query: Any, tracer: QueueTracer, names: dict[int, str]) -> dict[str, Any]:
    best = None
    for item in query.inference_history[1:]:
        summary = tracer.summarize_iteration(item.request_uid, names)
        raw = item.end_time - item.start_time
        row = {
            "raw_tpot_s": raw,
            "queue_wait_s": summary["queue_wait_s"],
            "layer_service_s": summary["layer_service_s"],
            "blocker_phase_s": summary["blocker_phase_s"],
            "top_blocker_queries": summary["top_blocker_queries"],
        }
        if best is None or row["queue_wait_s"] > best["queue_wait_s"] + EPS or (
            abs(row["queue_wait_s"] - best["queue_wait_s"]) <= EPS and raw > best["raw_tpot_s"]
        ):
            best = row
    if best is None:
        raise RuntimeError("target has no Decode iteration")
    return best


def _new_runtime(root: Path, pipeline):
    runtime = _load_helix_runtime(root)
    simulator, mini_pipeline, _ = _build_helix_simulator(pipeline=pipeline, runtime=runtime)
    tracer = QueueTracer(simulator)
    tracer.install()
    return runtime, simulator, mini_pipeline, tracer


def _isolated_prefill_finish(root: Path, pipeline) -> float:
    runtime, simulator, mini_pipeline, _ = _new_runtime(root, pipeline)
    base = simulator.current_time
    uid = _issue_fixed_query(
        simulator=simulator,
        runtime=runtime,
        creation_time=base,
        input_tokens=TARGET_INPUT,
        output_tokens=TARGET_OUTPUT,
        mini_pipeline=mini_pipeline,
    )
    _simulate(simulator, 1_000_000)
    query = simulator.query_manager.finished_queries[uid][1]
    return query.inference_history[0].end_time - base


def _controlled_case(root: Path, pipeline, prefill_tokens: int, offset_s: float, baseline_prefill_finish_s: float) -> dict[str, Any]:
    runtime, simulator, mini_pipeline, tracer = _new_runtime(root, pipeline)
    base = simulator.current_time
    target_uid = _issue_fixed_query(
        simulator=simulator,
        runtime=runtime,
        creation_time=base,
        input_tokens=TARGET_INPUT,
        output_tokens=TARGET_OUTPUT,
        mini_pipeline=mini_pipeline,
    )
    arrival = base + baseline_prefill_finish_s + offset_s
    interferer_uid = _issue_fixed_query(
        simulator=simulator,
        runtime=runtime,
        creation_time=arrival,
        input_tokens=prefill_tokens,
        output_tokens=INTERFERER_OUTPUT,
        mini_pipeline=mini_pipeline,
    )
    _simulate(simulator, 1_000_000)
    target = simulator.query_manager.finished_queries[target_uid][1]
    names = {target_uid: "target", interferer_uid: "interferer"}
    wait = _max_target_wait(query=target, tracer=tracer, names=names)
    service = _prefill_service(tracer, interferer_uid)
    obs = _request_observables(
        query=target,
        arrival_s=arrival,
        input_tokens=TARGET_INPUT,
        output_tokens=TARGET_OUTPUT,
    )
    return {
        "source": "E10-style controlled rerun",
        "pipeline_id": pipeline.id,
        "interferer_prompt_tokens": prefill_tokens,
        "arrival_offset_s": offset_s,
        "full_prefill_service_s": service,
        "observed_queue_wait_s": wait["queue_wait_s"],
        "exposure_fraction": wait["queue_wait_s"] / service if service > 0 else None,
        "observables": obs,
        "target_max_wait": wait,
    }


def _seed19_case(root: Path, pipeline, workload, intensity: float) -> dict[str, Any]:
    scaled = scale_workload(workload, intensity)
    by_id = {r.id: r for r in scaled}
    target_spec = by_id["azure-00009"]
    interferer_spec = by_id["azure-00010"]
    runtime, simulator, mini_pipeline, tracer = _new_runtime(root, pipeline)
    base = simulator.current_time
    names: dict[int, str] = {}
    target_uid = None
    interferer_uid = None
    for request in scaled:
        uid = _issue_fixed_query(
            simulator=simulator,
            runtime=runtime,
            creation_time=base + request.arrival_time_s,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            mini_pipeline=mini_pipeline,
        )
        names[uid] = request.id
        if request.id == target_spec.id:
            target_uid = uid
        if request.id == interferer_spec.id:
            interferer_uid = uid
    if target_uid is None or interferer_uid is None:
        raise RuntimeError("seed19 target/interferer missing")
    _simulate(simulator)
    target = simulator.query_manager.finished_queries[target_uid][1]
    arrival = base + interferer_spec.arrival_time_s
    wait = _max_target_wait(query=target, tracer=tracer, names=names)
    service = _prefill_service(tracer, interferer_uid)
    obs = _request_observables(
        query=target,
        arrival_s=arrival,
        input_tokens=target_spec.input_tokens,
        output_tokens=target_spec.output_tokens,
    )
    return {
        "source": "seed19 Slow",
        "pipeline_id": pipeline.id,
        "intensity": intensity,
        "interferer_prompt_tokens": interferer_spec.input_tokens,
        "full_prefill_service_s": service,
        "observed_queue_wait_s": wait["queue_wait_s"],
        "exposure_fraction": wait["queue_wait_s"] / service if service > 0 else None,
        "observables": obs,
        "target_max_wait": wait,
    }


def _same_visible_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    o = row["observables"]
    return (
        row["pipeline_id"],
        row["interferer_prompt_tokens"],
        o["event_type"],
        round(o["prefill_age_s"], 12),
        o["decode_completed_tokens"],
        o["decode_block_index"],
        o["decode_context_tokens"],
        o["decode_remaining_tokens"],
    )


def _collision_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_same_visible_signature(row), []).append(row)
    collisions = []
    for signature, items in groups.items():
        if len(items) < 2:
            continue
        fractions = [item["exposure_fraction"] for item in items if item["exposure_fraction"] is not None]
        if len(fractions) < 2:
            continue
        spread = max(fractions) - min(fractions)
        collisions.append(
            {
                "signature": list(signature),
                "cases": [
                    {
                        "source": item["source"],
                        "arrival_offset_s": item.get("arrival_offset_s"),
                        "intensity": item.get("intensity"),
                        "exposure_fraction": item["exposure_fraction"],
                        "decode_iteration_in_flight_at_arrival": item["observables"]["decode_iteration_in_flight_at_arrival"],
                    }
                    for item in items
                ],
                "exposure_spread": spread,
            }
        )
    collisions.sort(key=lambda item: item["exposure_spread"], reverse=True)
    return {
        "identical_visible_state_collision_groups": len(collisions),
        "maximum_exposure_spread_under_identical_visible_state": collisions[0]["exposure_spread"] if collisions else 0.0,
        "worst_collision": collisions[0] if collisions else None,
    }


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    pipelines = {p.id: p for p in build_helix_pipelines()}
    baselines = {
        pid: _isolated_prefill_finish(root, pipelines[pid])
        for pid in {case[0] for case in CONTROLLED}
    }
    controlled_rows = [
        _controlled_case(root, pipelines[pid], prompt, offset, baselines[pid])
        for pid, prompt, offset in CONTROLLED
    ]

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=19,
        interval_offset=0,
    )
    slow = pipelines["helix-slow-link-placement"]
    seed19_rows = [_seed19_case(root, slow, sample.requests, x) for x in SEED19_POINTS]
    all_rows = controlled_rows + seed19_rows
    result = {
        "design": {
            "purpose": "test whether cheap request/event-level observables already available to the scheduler-free Evaluator can explain Prefill exposure variation",
            "observables_tested": [
                "event type",
                "Prefill age at overlap onset",
                "completed Decode tokens",
                "16-token Decode block index",
                "Decode context/remaining tokens",
            ],
            "evaluator_semantics_changed": False,
            "scheduler_state_added": False,
            "network_redline_changed": False,
        },
        "controlled_cases": controlled_rows,
        "seed19_cases": seed19_rows,
        "collision_audit": _collision_audit(all_rows),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
