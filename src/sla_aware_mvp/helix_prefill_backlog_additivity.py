from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .domain import RequestSpec
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import _issue_fixed_query
from .helix_prefill_decode_sweep import (
    TARGET_INPUT_TOKENS,
    TARGET_OUTPUT_TOKENS,
    _baseline,
    _build_runtime_case,
    _max_decode_item,
    _query,
    _simulate_to_completion,
)


CASES = (
    ((256, 0.02), (256, 0.04)),
    ((512, 0.02), (512, 0.04)),
    ((256, 0.02), (512, 0.04)),
    ((512, 0.02), (1024, 0.04)),
    ((256, 0.01), (256, 0.03), (256, 0.05)),
    ((512, 0.01), (512, 0.03), (512, 0.05)),
)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _run_case(
    *,
    helix_root: Path,
    pipeline_id: str,
    baseline: dict[str, Any],
    specs: tuple[tuple[int, float], ...],
) -> dict[str, Any]:
    pipelines = {pipeline.id: pipeline for pipeline in build_helix_pipelines()}
    pipeline = pipelines[pipeline_id]
    profiler = HelixA100Llama2Profiler.from_artifact(helix_root, commit=HELIX_COMMIT)
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

    interferers: list[tuple[int, int, float]] = []
    for tokens, offset_s in specs:
        arrival_rel_s = baseline["prefill_finish_rel_s"] + offset_s
        uid = _issue_fixed_query(
            simulator=simulator,
            runtime=runtime,
            creation_time=base_time + arrival_rel_s,
            input_tokens=tokens,
            output_tokens=1,
            mini_pipeline=mini_pipeline,
        )
        interferers.append((uid, tokens, arrival_rel_s))

    events = _simulate_to_completion(simulator)
    target = _query(simulator, target_uid)
    decode_index, decode = _max_decode_item(target)
    name_by_uid = {target_uid: "target"}
    for index, (uid, _, _) in enumerate(interferers, start=1):
        name_by_uid[uid] = f"prefill-{index}"
    summary = tracer.summarize_request(
        request_uid=decode.request_uid,
        query_name_by_uid=name_by_uid,
    )

    overlapping: list[dict[str, Any]] = []
    candidate_ub = 0.0
    for index, (uid, tokens, arrival_rel_s) in enumerate(interferers, start=1):
        query = _query(simulator, uid)
        prefill = query.inference_history[0]
        overlap_s = _overlap(decode.start_time, decode.end_time, prefill.start_time, prefill.end_time)
        if overlap_s <= 0:
            continue
        request = RequestSpec(
            id=f"prefill-{index}",
            arrival_time_s=arrival_rel_s,
            input_tokens=tokens,
            output_tokens=1,
        )
        full_compute_s = profiler.prefill_time(request, pipeline, 1, 1)
        candidate_ub += full_compute_s
        overlapping.append(
            {
                "name": f"prefill-{index}",
                "tokens": tokens,
                "prefill_overlap_with_decode_s": overlap_s,
                "full_profiled_prefill_compute_s": full_compute_s,
            }
        )

    raw_tpot = decode.end_time - decode.start_time
    queue_wait = summary["queue_wait_s"]
    return {
        "interferers": [
            {"tokens": tokens, "offset_after_target_prefill_s": offset_s}
            for tokens, offset_s in specs
        ],
        "events": events,
        "target_max_decode_index": decode_index,
        "target_raw_tpot_s": raw_tpot,
        "target_queue_wait_s": queue_wait,
        "target_layer_service_s": summary["layer_service_s"],
        "blocker_phase_s": summary["blocker_phase_s"],
        "top_blocker_queries": summary["top_blocker_queries"],
        "overlapping_prefills": overlapping,
        "candidate_full_prefill_backlog_ub_s": candidate_ub,
        "queue_to_candidate_ub_ratio": queue_wait / candidate_ub if candidate_ub > 0 else None,
        "candidate_bound_holds": queue_wait <= candidate_ub + 1e-9,
    }


def main() -> None:
    helix_root_env = os.environ.get("HELIX_ROOT")
    pipeline_id = os.environ.get("ADDITIVITY_PIPELINE", "helix-fast-link-placement")
    if not helix_root_env:
        raise RuntimeError("set HELIX_ROOT")
    helix_root = Path(helix_root_env)
    baseline = _baseline(helix_root=helix_root, pipeline_id=pipeline_id)
    cases = [
        _run_case(
            helix_root=helix_root,
            pipeline_id=pipeline_id,
            baseline=baseline,
            specs=specs,
        )
        for specs in CASES
    ]
    ratios = [case["queue_to_candidate_ub_ratio"] for case in cases if case["queue_to_candidate_ub_ratio"] is not None]
    result = {
        "pipeline_id": pipeline_id,
        "baseline": baseline,
        "cases": cases,
        "summary": {
            "cases": len(cases),
            "bound_violations": sum(not case["candidate_bound_holds"] for case in cases),
            "max_queue_to_candidate_ub_ratio": max(ratios) if ratios else None,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
