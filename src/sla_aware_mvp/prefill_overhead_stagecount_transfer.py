from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import Boundary, GPUNode, Link, Pipeline, RequestSpec, Stage
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_model
from .helix_fixed_reference import (
    _build_helix_simulator,
    _issue_fixed_query,
    _load_helix_runtime,
)
from .helix_queue_diagnostic import HelixQueueTracer


# E22 calibrated this total 8-stage coefficient from isolated TTFT thresholds.
E22_EIGHT_STAGE_OVERHEAD_S_PER_TOKEN = 59.37720874470879e-6
CALIBRATION_STAGE_COUNT = 8
PROMPTS = (256, 512, 1024)
STAGE_LAYER_COUNTS = {
    7: (12, 12, 12, 11, 11, 11, 11),
    8: (10, 10, 10, 10, 10, 10, 10, 10),
    10: (8, 8, 8, 8, 8, 8, 8, 8, 8, 8),
}
LINK_GBPS = 10.0


def build_stage_count_pipeline(stage_count: int) -> Pipeline:
    layer_counts = STAGE_LAYER_COUNTS[stage_count]
    if sum(layer_counts) != 80 or len(layer_counts) != stage_count:
        raise AssertionError("invalid stage partition")
    model = build_helix_model()
    nodes = {
        f"n{i}": GPUNode(
            id=f"n{i}",
            compute_tflops=0.0,
            memory_capacity_bytes=int(40e9),
            hardware_type="A100-40GB",
        )
        for i in range(stage_count)
    }
    links = {
        f"link-{i}": Link(
            id=f"link-{i}",
            source=f"n{i}",
            target=f"n{i + 1}",
            capacity_bytes_per_s=LINK_GBPS * 1e9 / 8.0,
        )
        for i in range(stage_count - 1)
    }
    stages = []
    cursor = 0
    for i, layers in enumerate(layer_counts):
        stages.append(Stage(f"s{i}", cursor, cursor + layers, f"n{i}"))
        cursor += layers
    boundaries = tuple(
        Boundary(f"s{i}", f"s{i + 1}", (f"link-{i}",))
        for i in range(stage_count - 1)
    )
    pipeline = Pipeline(
        id=f"helix-stagecount-{stage_count}",
        model=model,
        nodes=nodes,
        links=links,
        stages=tuple(stages),
        boundaries=boundaries,
    )
    pipeline.validate()
    return pipeline


def simulate_to_completion(simulator) -> int:
    events = 0
    while simulator.query_manager.queries_on_the_fly:
        succeeded, _ = simulator.simulate_next_event()
        events += 1
        if not succeeded:
            raise RuntimeError("HELIX event queue drained with unfinished query")
        if events > 1_000_000:
            raise RuntimeError("HELIX stage-count transfer event limit exceeded")
    return events


def run_case(*, helix_root: Path, stage_count: int, prompt_tokens: int) -> dict:
    pipeline = build_stage_count_pipeline(stage_count)
    profiler = HelixA100Llama2Profiler.from_artifact(helix_root, commit=HELIX_COMMIT)
    runtime = _load_helix_runtime(helix_root)
    simulator, mini_pipeline, _ = _build_helix_simulator(
        pipeline=pipeline,
        runtime=runtime,
    )
    tracer = HelixQueueTracer(simulator)
    tracer.install()
    base_time = simulator.current_time
    query_uid = _issue_fixed_query(
        simulator=simulator,
        runtime=runtime,
        creation_time=base_time,
        input_tokens=prompt_tokens,
        output_tokens=1,
        mini_pipeline=mini_pipeline,
    )
    events = simulate_to_completion(simulator)

    measured_prefill_service_s = 0.0
    measured_prefill_batches = 0
    for batch in tracer.batch_intervals:
        if query_uid not in batch.base_query_uids:
            continue
        matching_phases = [
            phase
            for uid, phase in zip(batch.base_query_uids, batch.phases)
            if uid == query_uid
        ]
        if "Initialization" not in matching_phases:
            continue
        measured_prefill_service_s += batch.duration_s
        measured_prefill_batches += 1

    request = RequestSpec(
        id=f"p{prompt_tokens}",
        arrival_time_s=0.0,
        input_tokens=prompt_tokens,
        output_tokens=1,
    )
    profiled_compute_s = profiler.prefill_time(
        request,
        pipeline,
        n_prefill=1,
        n_decode=0,
    )
    per_stage_overhead_s_per_token = (
        E22_EIGHT_STAGE_OVERHEAD_S_PER_TOKEN / CALIBRATION_STAGE_COUNT
    )
    predicted_overhead_s = (
        per_stage_overhead_s_per_token * stage_count * prompt_tokens
    )
    predicted_service_s = profiled_compute_s + predicted_overhead_s

    return {
        "stage_count": stage_count,
        "layer_counts": list(STAGE_LAYER_COUNTS[stage_count]),
        "prompt_tokens": prompt_tokens,
        "events": events,
        "measured_prefill_batches": measured_prefill_batches,
        "profiled_compute_s": profiled_compute_s,
        "e22_per_stage_overhead_us_per_token": per_stage_overhead_s_per_token * 1e6,
        "predicted_overhead_s": predicted_overhead_s,
        "predicted_prefill_service_s": predicted_service_s,
        "measured_prefill_service_s": measured_prefill_service_s,
        "absolute_error_s": predicted_service_s - measured_prefill_service_s,
        "relative_error": (
            (predicted_service_s - measured_prefill_service_s)
            / measured_prefill_service_s
            if measured_prefill_service_s > 0
            else None
        ),
    }


def main() -> None:
    helix_root_env = os.environ.get("HELIX_ROOT")
    if not helix_root_env:
        raise RuntimeError("set HELIX_ROOT")
    helix_root = Path(helix_root_env)
    rows = [
        run_case(
            helix_root=helix_root,
            stage_count=stage_count,
            prompt_tokens=prompt_tokens,
        )
        for stage_count in STAGE_LAYER_COUNTS
        for prompt_tokens in PROMPTS
    ]
    held_out_stage_rows = [row for row in rows if row["stage_count"] != 8]
    result = {
        "design": {
            "purpose": "out-of-configuration test of whether independently calibrated Prefill runtime overhead scales per compute stage",
            "helix_commit": HELIX_COMMIT,
            "calibration_stage_count": CALIBRATION_STAGE_COUNT,
            "test_stage_counts": list(STAGE_LAYER_COUNTS),
            "prompts": list(PROMPTS),
            "network_gbps": LINK_GBPS,
            "overhead_model": "T_ovhd^P = (h_8/8) * num_stages * prompt_tokens",
            "evaluator_semantics_changed": False,
            "helix_scheduler_changed": False,
            "network_redline_changed": False,
        },
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "held_out_stage_count_cases": len(held_out_stage_rows),
            "max_abs_relative_error_all": max(
                abs(row["relative_error"]) for row in rows
            ),
            "max_abs_relative_error_held_out_stage_counts": max(
                abs(row["relative_error"]) for row in held_out_stage_rows
            ),
        },
        "interpretation_guardrail": "Stage-count scaling is a candidate profiling interface, not a universal systems law. The 7- and 10-stage cases are the actual transfer test; the 8-stage rows are only an in-configuration control. HELIX remains a relative execution reference.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
