from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import SLA
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import _build_helix_simulator, _issue_fixed_query, _load_helix_runtime
from .seed19_tightness_diagnostic import (
    CANDIDATE_UNSAFE,
    FAST_POINTS,
    OFFSET,
    SEED,
    SLOW_POINTS,
    QueueTracer,
    _evaluator_overlap,
    _iteration_record,
)
from .capacity import scale_workload
from .workload import build_helix_azure_conversation_workload


def _helix_trace_fast(*, root: Path, pipeline, workload, intensity: float, target_request_id: str):
    scaled = scale_workload(workload, intensity)
    runtime = _load_helix_runtime(root)
    simulator, mini_pipeline, _ = _build_helix_simulator(pipeline=pipeline, runtime=runtime)
    tracer = QueueTracer(simulator)
    tracer.install()
    base_time = simulator.current_time
    uid_to_name: dict[int, str] = {}
    for request in scaled:
        uid = _issue_fixed_query(
            simulator=simulator,
            runtime=runtime,
            creation_time=base_time + request.arrival_time_s,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            mini_pipeline=mini_pipeline,
        )
        uid_to_name[uid] = request.id

    events = 0
    while simulator.query_manager.queries_on_the_fly:
        succeeded, _ = simulator.simulate_next_event()
        events += 1
        if not succeeded:
            raise RuntimeError("HELIX event queue drained with unfinished queries")
        if events > 5_000_000:
            raise RuntimeError("HELIX diagnostic event limit exceeded")

    global_best = None
    target_best = None
    for query_uid, (_, query) in simulator.query_manager.finished_queries.items():
        request_id = uid_to_name[query_uid]
        for item in query.inference_history[1:]:
            duration = item.end_time - item.start_time
            if global_best is None or duration > global_best[0]:
                global_best = (duration, request_id, item)
            if request_id == target_request_id and (
                target_best is None or duration > target_best[0]
            ):
                target_best = (duration, request_id, item)
    if global_best is None or target_best is None:
        raise RuntimeError("missing decode iteration in HELIX diagnostic")

    global_record = _iteration_record(
        item=global_best[2], request_id=global_best[1], tracer=tracer, names=uid_to_name
    )
    if target_best[2].request_uid == global_best[2].request_uid:
        target_record = global_record
    else:
        target_record = _iteration_record(
            item=target_best[2], request_id=target_best[1], tracer=tracer, names=uid_to_name
        )
    return {
        "pipeline_id": pipeline.id,
        "intensity": intensity,
        "events": events,
        "target_request_max_decode": target_record,
        "global_max_decode": global_record,
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
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    pipelines = {pipeline.id: pipeline for pipeline in build_helix_pipelines()}
    slow = pipelines["helix-slow-link-placement"]
    fast = pipelines["helix-fast-link-placement"]

    evaluator = _evaluator_overlap(pipeline=slow, workload=workload, profiler=profiler, sla=sla)
    target_request_id = evaluator["violation"]["request_id"]
    helix = {
        "slow": [
            _helix_trace_fast(
                root=root,
                pipeline=slow,
                workload=workload,
                intensity=value,
                target_request_id=target_request_id,
            )
            for value in SLOW_POINTS
        ],
        "fast": [
            _helix_trace_fast(
                root=root,
                pipeline=fast,
                workload=workload,
                intensity=value,
                target_request_id=target_request_id,
            )
            for value in FAST_POINTS
        ],
    }
    print(
        json.dumps(
            {
                "design": {
                    "purpose": "E36 optimized seed19 tightness diagnostic",
                    "seed": SEED,
                    "offset": OFFSET,
                    "candidate_unsafe": CANDIDATE_UNSAFE,
                    "slow_points": list(SLOW_POINTS),
                    "fast_points": list(FAST_POINTS),
                    "evaluator_semantics_changed": False,
                    "scheduler_reconstructed": False,
                },
                "workload_requests": len(workload),
                "evaluator_first_overlap_violation": evaluator,
                "helix_queue_diagnostics": helix,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
