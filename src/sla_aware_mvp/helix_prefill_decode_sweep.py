from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .helix_demo import build_helix_pipelines
from .helix_fixed_reference import (
    _build_helix_simulator,
    _issue_fixed_query,
    _load_helix_runtime,
)
from .helix_queue_diagnostic import HelixQueueTracer


TARGET_INPUT_TOKENS = 256
TARGET_OUTPUT_TOKENS = 64
INTERFERER_OUTPUT_TOKENS = 1
PREFILL_LENGTHS = (64, 256, 512, 1024)
DECODE_RELATIVE_OFFSETS_S = (0.01, 0.04, 0.08, 0.12)


def _simulate_to_completion(simulator: Any) -> int:
    events = 0
    while simulator.query_manager.queries_on_the_fly:
        succeeded, _ = simulator.simulate_next_event()
        events += 1
        if not succeeded:
            raise RuntimeError("HELIX event queue drained with unfinished queries")
        if events > 1_000_000:
            raise RuntimeError("HELIX two-query sweep event limit exceeded")
    return events


def _build_runtime_case(*, helix_root: Path, pipeline_id: str):
    pipelines = {pipeline.id: pipeline for pipeline in build_helix_pipelines()}
    pipeline = pipelines[pipeline_id]
    runtime = _load_helix_runtime(helix_root)
    simulator, mini_pipeline, _ = _build_helix_simulator(
        pipeline=pipeline,
        runtime=runtime,
    )
    tracer = HelixQueueTracer(simulator)
    tracer.install()
    return runtime, simulator, mini_pipeline, tracer


def _query(simulator: Any, query_uid: int):
    return simulator.query_manager.finished_queries[query_uid][1]


def _max_decode_item(query: Any) -> tuple[int, Any]:
    items = list(enumerate(query.inference_history[1:], start=1))
    if not items:
        raise RuntimeError("query has no decode iteration")
    return max(items, key=lambda pair: pair[1].end_time - pair[1].start_time)


def _phase_name(phase: Any) -> str:
    return getattr(phase, "name", str(phase))


def _baseline(*, helix_root: Path, pipeline_id: str) -> dict[str, Any]:
    runtime, simulator, mini_pipeline, tracer = _build_runtime_case(
        helix_root=helix_root,
        pipeline_id=pipeline_id,
    )
    base_time = simulator.current_time
    target_uid = _issue_fixed_query(
        simulator=simulator,
        runtime=runtime,
        creation_time=base_time,
        input_tokens=TARGET_INPUT_TOKENS,
        output_tokens=TARGET_OUTPUT_TOKENS,
        mini_pipeline=mini_pipeline,
    )
    events = _simulate_to_completion(simulator)
    target = _query(simulator, target_uid)
    prefill = target.inference_history[0]
    decode_index, decode = _max_decode_item(target)
    summary = tracer.summarize_request(
        request_uid=decode.request_uid,
        query_name_by_uid={target_uid: "target"},
    )
    return {
        "events": events,
        "prefill_finish_rel_s": prefill.end_time - base_time,
        "max_decode_index": decode_index,
        "max_raw_tpot_s": decode.end_time - decode.start_time,
        "decode_layer_service_s": summary["layer_service_s"],
        "decode_queue_wait_s": summary["queue_wait_s"],
    }


def _interferer_prefill_service(tracer: HelixQueueTracer, interferer_uid: int) -> dict[str, float]:
    pure = 0.0
    mixed = 0.0
    total_batch_time_containing_prefill = 0.0
    for batch in tracer.batch_intervals:
        matched = False
        for uid, phase in zip(batch.base_query_uids, batch.phases):
            if uid == interferer_uid and phase == "Initialization":
                matched = True
                break
        if not matched:
            continue
        total_batch_time_containing_prefill += batch.duration_s
        if set(batch.phases) == {"Initialization"}:
            pure += batch.duration_s
        else:
            mixed += batch.duration_s
    return {
        "pure_prefill_batch_service_s": pure,
        "mixed_batch_service_s": mixed,
        "all_batches_containing_prefill_s": total_batch_time_containing_prefill,
    }


def _run_case(
    *,
    helix_root: Path,
    pipeline_id: str,
    baseline: dict[str, Any],
    prefill_tokens: int,
    decode_relative_offset_s: float,
) -> dict[str, Any]:
    runtime, simulator, mini_pipeline, tracer = _build_runtime_case(
        helix_root=helix_root,
        pipeline_id=pipeline_id,
    )
    base_time = simulator.current_time
    target_uid = _issue_fixed_query(
        simulator=simulator,
        runtime=runtime,
        creation_time=base_time,
        input_tokens=TARGET_INPUT_TOKENS,
        output_tokens=TARGET_OUTPUT_TOKENS,
        mini_pipeline=mini_pipeline,
    )
    interferer_arrival_rel_s = baseline["prefill_finish_rel_s"] + decode_relative_offset_s
    interferer_uid = _issue_fixed_query(
        simulator=simulator,
        runtime=runtime,
        creation_time=base_time + interferer_arrival_rel_s,
        input_tokens=prefill_tokens,
        output_tokens=INTERFERER_OUTPUT_TOKENS,
        mini_pipeline=mini_pipeline,
    )
    events = _simulate_to_completion(simulator)

    target = _query(simulator, target_uid)
    decode_index, decode = _max_decode_item(target)
    summary = tracer.summarize_request(
        request_uid=decode.request_uid,
        query_name_by_uid={target_uid: "target", interferer_uid: "interferer"},
    )
    raw_tpot = decode.end_time - decode.start_time
    service_inflation = summary["layer_service_s"] - baseline["decode_layer_service_s"]
    interferer_service = _interferer_prefill_service(tracer, interferer_uid)
    prefill_service = interferer_service["all_batches_containing_prefill_s"]
    accounted_interference = summary["queue_wait_s"] + max(service_inflation, 0.0)

    return {
        "prefill_tokens": prefill_tokens,
        "decode_relative_offset_s": decode_relative_offset_s,
        "interferer_arrival_rel_s": interferer_arrival_rel_s,
        "events": events,
        "target_max_decode_index": decode_index,
        "target_raw_tpot_s": raw_tpot,
        "target_tpot_with_fixed_overhead_s": raw_tpot + 0.005,
        "target_queue_wait_s": summary["queue_wait_s"],
        "target_layer_service_s": summary["layer_service_s"],
        "target_service_inflation_vs_isolated_s": service_inflation,
        "blocker_phase_s": summary["blocker_phase_s"],
        "top_blocker_queries": summary["top_blocker_queries"],
        "interferer_prefill_service": interferer_service,
        "queue_plus_service_inflation_s": accounted_interference,
        "interference_to_prefill_service_ratio": (
            accounted_interference / prefill_service if prefill_service > 0 else None
        ),
    }


def main() -> None:
    helix_root_env = os.environ.get("HELIX_ROOT")
    pipeline_id = os.environ.get("SWEEP_PIPELINE", "helix-fast-link-placement")
    if not helix_root_env:
        raise RuntimeError("set HELIX_ROOT")
    helix_root = Path(helix_root_env)

    baseline = _baseline(helix_root=helix_root, pipeline_id=pipeline_id)
    cases = [
        _run_case(
            helix_root=helix_root,
            pipeline_id=pipeline_id,
            baseline=baseline,
            prefill_tokens=prefill_tokens,
            decode_relative_offset_s=offset_s,
        )
        for prefill_tokens in PREFILL_LENGTHS
        for offset_s in DECODE_RELATIVE_OFFSETS_S
    ]
    result = {
        "pipeline_id": pipeline_id,
        "design": {
            "target_input_tokens": TARGET_INPUT_TOKENS,
            "target_output_tokens": TARGET_OUTPUT_TOKENS,
            "interferer_output_tokens": INTERFERER_OUTPUT_TOKENS,
            "prefill_lengths": list(PREFILL_LENGTHS),
            "decode_relative_offsets_s": list(DECODE_RELATIVE_OFFSETS_S),
            "arrival_definition": "interferer arrival = isolated target Prefill finish + offset",
        },
        "baseline": baseline,
        "cases": cases,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
