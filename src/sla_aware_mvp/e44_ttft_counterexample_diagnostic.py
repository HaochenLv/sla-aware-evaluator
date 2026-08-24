from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, RequestRuntime, SLA
from .evaluator import _network_bytes_by_link, evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import (
    _build_helix_simulator,
    _issue_fixed_query,
    _load_helix_runtime,
)
from .prefill_debt_budget_ablation import (
    E22_BLOCKING_OVERHEAD_S_PER_TOKEN,
    _budget_trial,
)
from .workload import build_helix_azure_conversation_workload

INTENSITY = 0.0109
SIZE_SCALE = 1.2
TARGET_REQUEST_ID = "azure-00008"


def _scale_sizes(workload, factor: float):
    return tuple(
        replace(
            request,
            input_tokens=max(1, int(round(request.input_tokens * factor))),
            output_tokens=max(1, int(round(request.output_tokens * factor))),
        )
        for request in workload
    )


def _parse_location(name: str):
    kind, uid = name.rsplit("-", 1)
    return kind, int(uid)


def _request_intervals(entry, helix_node_to_logical):
    rows = []
    history = entry.request.location_history
    for index in range(1, len(history)):
        prev_name, start = history[index - 1]
        next_name, end = history[index]
        prev_kind, prev_uid = _parse_location(prev_name)
        next_kind, next_uid = _parse_location(next_name)
        row = {
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "from": prev_name,
            "to": next_name,
        }
        if "ComputeNode" in prev_kind:
            row["class"] = "compute_node_residence"
            row["logical_node"] = helix_node_to_logical.get(prev_uid)
        elif "SourceNode" in prev_kind:
            row["class"] = "source_residence"
        elif "Link" in prev_kind:
            row["class"] = "network_link_service"
            row["link_uid"] = prev_uid
        else:
            row["class"] = "other"
        rows.append(row)
    return rows


def _run_helix_detailed(*, pipeline, workload, root, target_request_id):
    runtime = _load_helix_runtime(root)
    simulator, mini_pipeline, helix_uid_by_node_id = _build_helix_simulator(
        pipeline=pipeline, runtime=runtime
    )
    base_time = simulator.current_time
    query_uid_to_request_id = {}
    for request in workload:
        query_uid = _issue_fixed_query(
            simulator=simulator,
            runtime=runtime,
            creation_time=base_time + request.arrival_time_s,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            mini_pipeline=mini_pipeline,
        )
        query_uid_to_request_id[query_uid] = request.id

    processed = 0
    while simulator.query_manager.queries_on_the_fly:
        succeeded, _ = simulator.simulate_next_event()
        processed += 1
        if not succeeded:
            raise RuntimeError("HELIX event queue drained with unfinished queries")
        if processed > 5_000_000:
            raise RuntimeError("HELIX diagnostic event limit exceeded")

    target_query = None
    target_query_uid = None
    request_uid_metadata = {}
    for query_uid, (_, query) in simulator.query_manager.finished_queries.items():
        request_id = query_uid_to_request_id[query_uid]
        if request_id == target_request_id:
            target_query = query
            target_query_uid = query_uid
        for iteration_index, history in enumerate(query.inference_history):
            request_uid_metadata[history.request_uid] = {
                "request_id": request_id,
                "iteration_index": iteration_index,
                "phase": "prefill" if iteration_index == 0 else "decode",
            }
    if target_query is None:
        raise RuntimeError(f"target query {target_request_id} not found")

    prefill_history = target_query.inference_history[0]
    analyzer = simulator.query_manager.latency_analyzer
    target_entry = analyzer.request_latency[prefill_history.request_uid]
    helix_node_to_logical = {uid: node_id for node_id, uid in helix_uid_by_node_id.items()}
    target_intervals = _request_intervals(target_entry, helix_node_to_logical)

    all_node_intervals = []
    for request_uid, entry in analyzer.request_latency.items():
        meta = request_uid_metadata[request_uid]
        for interval in _request_intervals(entry, helix_node_to_logical):
            if interval["class"] != "compute_node_residence":
                continue
            all_node_intervals.append(
                {
                    **interval,
                    "request_uid": request_uid,
                    **meta,
                }
            )

    return {
        "query_uid": target_query_uid,
        "request_uid": prefill_history.request_uid,
        "raw_prefill_total_s": target_entry.total,
        "raw_prefill_compute_plus_queue_s": target_entry.compute,
        "raw_prefill_network_s": target_entry.network,
        "location_intervals": target_intervals,
        "all_node_intervals": all_node_intervals,
    }


def _stage_diagnostics(*, pipeline, request, profiler, full, isolated):
    stage_by_node = {stage.node_id: stage for stage in pipeline.stages}
    isolated_by_node = {
        row.get("logical_node"): row
        for row in isolated["location_intervals"]
        if row["class"] == "compute_node_residence"
    }
    all_intervals = full["all_node_intervals"]
    rows = []
    for row in full["location_intervals"]:
        if row["class"] != "compute_node_residence":
            continue
        node_id = row["logical_node"]
        stage = stage_by_node[node_id]
        pure_compute = profiler.prefill_stage_time(request, pipeline, stage, 1, 0)
        isolated_row = isolated_by_node[node_id]
        full_residence = row["duration_s"]
        isolated_residence = isolated_row["duration_s"]
        overlaps = []
        for other in all_intervals:
            if other["logical_node"] != node_id:
                continue
            if other["request_uid"] == full["request_uid"]:
                continue
            overlap = min(row["end_s"], other["end_s"]) - max(row["start_s"], other["start_s"])
            if overlap > 1e-9:
                overlaps.append(
                    {
                        "request_id": other["request_id"],
                        "phase": other["phase"],
                        "iteration_index": other["iteration_index"],
                        "overlap_s": overlap,
                    }
                )
        overlaps.sort(key=lambda item: item["overlap_s"], reverse=True)
        rows.append(
            {
                "stage_id": stage.id,
                "logical_node": node_id,
                "layers": [stage.layer_start, stage.layer_end],
                "profile_pure_compute_s": pure_compute,
                "isolated_node_residence_s": isolated_residence,
                "full_node_residence_s": full_residence,
                "isolated_minus_profile_s": isolated_residence - pure_compute,
                "full_minus_isolated_s": full_residence - isolated_residence,
                "full_minus_profile_s": full_residence - pure_compute,
                "overlapping_requests": overlaps[:8],
            }
        )
    return rows


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
        seed=7,
        interval_offset=0,
    )
    base_workload = sample.requests
    sized_workload = _scale_sizes(base_workload, SIZE_SCALE)
    workload = scale_workload(sized_workload, INTENSITY)
    target = next(request for request in workload if request.id == TARGET_REQUEST_ID)
    isolated_target = replace(target, arrival_time_s=0.0)

    pipeline = next(
        item for item in build_helix_pipelines() if item.id == "helix-slow-link-placement"
    )
    base_profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    profiler = ExactHelixDecodeRuntimeProfiler(base_profiler)
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)

    candidate = _budget_trial(
        intensity=INTENSITY,
        pipeline=pipeline,
        workload=sized_workload,
        profiler=profiler,
        sla=sla,
    )
    evaluator_run = evaluate(
        pipeline=pipeline,
        workload=workload,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    target_snapshot = min(
        (
            snapshot
            for snapshot in evaluator_run.trace
            if TARGET_REQUEST_ID in snapshot.request_phase
            and abs(snapshot.time_s - target.arrival_time_s) <= 1e-8
        ),
        key=lambda snapshot: abs(snapshot.time_s - target.arrival_time_s),
    )

    full = _run_helix_detailed(
        pipeline=pipeline,
        workload=workload,
        root=root,
        target_request_id=TARGET_REQUEST_ID,
    )
    isolated = _run_helix_detailed(
        pipeline=pipeline,
        workload=(isolated_target,),
        root=root,
        target_request_id=TARGET_REQUEST_ID,
    )

    profile_compute_s = profiler.prefill_time(
        target,
        pipeline,
        target_snapshot.num_prefill,
        target_snapshot.num_decode,
    )
    runtime = RequestRuntime(target, Phase.PREFILL)
    network_demand = _network_bytes_by_link(runtime, pipeline)
    ideal_internal_network_s = sum(
        data_bytes / pipeline.links[link_id].capacity_bytes_per_s
        for link_id, data_bytes in network_demand.items()
        if data_bytes > 0
    )
    e22_overhead_s = E22_BLOCKING_OVERHEAD_S_PER_TOKEN * target.input_tokens
    evaluator_intrinsic_margin_s = (
        sla.ttft_s
        - sla.queue_overhead_s
        - sla.fixed_overhead_s
        - profile_compute_s
        - ideal_internal_network_s
    )

    stages = _stage_diagnostics(
        pipeline=pipeline,
        request=target,
        profiler=base_profiler,
        full=full,
        isolated=isolated,
    )

    result = {
        "design": {
            "purpose": "diagnose E44 seed7 Slow 1.2x TTFT candidate-optimism counterexample",
            "intensity": INTENSITY,
            "size_scale": SIZE_SCALE,
            "target_request_id": TARGET_REQUEST_ID,
            "candidate_changed": False,
            "production_evaluator_changed": False,
            "network_redline_changed": False,
        },
        "target": {
            "arrival_time_s": target.arrival_time_s,
            "input_tokens": target.input_tokens,
            "output_tokens": target.output_tokens,
            "evaluator_snapshot_num_prefill": target_snapshot.num_prefill,
            "evaluator_snapshot_num_decode": target_snapshot.num_decode,
            "evaluator_snapshot_phases": target_snapshot.request_phase,
        },
        "e31_candidate": candidate,
        "evaluator_prefill_budget_decomposition": {
            "ttft_limit_s": sla.ttft_s,
            "profile_compute_s": profile_compute_s,
            "ideal_internal_network_s": ideal_internal_network_s,
            "fixed_overhead_s": sla.fixed_overhead_s,
            "queue_overhead_s": sla.queue_overhead_s,
            "intrinsic_margin_after_profile_network_fixed_s": evaluator_intrinsic_margin_s,
            "e22_profiled_blocking_overhead_s": e22_overhead_s,
            "e22_minus_margin_s": e22_overhead_s - evaluator_intrinsic_margin_s,
        },
        "helix_full_workload": {
            "raw_prefill_total_s": full["raw_prefill_total_s"],
            "aligned_ttft_with_fixed_s": full["raw_prefill_total_s"] + sla.fixed_overhead_s,
            "compute_plus_queue_s": full["raw_prefill_compute_plus_queue_s"],
            "network_s": full["raw_prefill_network_s"],
        },
        "helix_isolated_target": {
            "raw_prefill_total_s": isolated["raw_prefill_total_s"],
            "aligned_ttft_with_fixed_s": isolated["raw_prefill_total_s"] + sla.fixed_overhead_s,
            "compute_plus_queue_s": isolated["raw_prefill_compute_plus_queue_s"],
            "network_s": isolated["raw_prefill_network_s"],
        },
        "full_minus_isolated": {
            "prefill_total_s": full["raw_prefill_total_s"] - isolated["raw_prefill_total_s"],
            "compute_plus_queue_s": full["raw_prefill_compute_plus_queue_s"] - isolated["raw_prefill_compute_plus_queue_s"],
            "network_s": full["raw_prefill_network_s"] - isolated["raw_prefill_network_s"],
        },
        "isolated_minus_evaluator_components": {
            "compute_plus_queue_minus_profile_compute_s": isolated["raw_prefill_compute_plus_queue_s"] - profile_compute_s,
            "network_minus_ideal_internal_network_s": isolated["raw_prefill_network_s"] - ideal_internal_network_s,
            "raw_total_minus_profile_plus_ideal_network_s": isolated["raw_prefill_total_s"] - profile_compute_s - ideal_internal_network_s,
        },
        "stage_diagnostics": stages,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
