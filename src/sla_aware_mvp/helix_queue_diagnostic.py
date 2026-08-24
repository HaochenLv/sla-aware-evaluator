from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capacity import scale_workload
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import (
    _build_helix_simulator,
    _issue_fixed_query,
    _load_helix_runtime,
)
from .workload import build_helix_azure_conversation_workload


@dataclass(frozen=True)
class QueueInterval:
    request_uid: int
    base_query_uid: int
    phase: str
    node_uid: int
    layer_id: int
    enqueue_s: float
    start_s: float

    @property
    def wait_s(self) -> float:
        return max(self.start_s - self.enqueue_s, 0.0)


@dataclass(frozen=True)
class BatchInterval:
    node_uid: int
    layer_id: int
    start_s: float
    end_s: float
    duration_s: float
    request_uids: tuple[int, ...]
    base_query_uids: tuple[int, ...]
    phases: tuple[str, ...]


class HelixQueueTracer:
    """Observe HELIX queueing without changing its scheduling semantics."""

    def __init__(self, simulator: Any) -> None:
        self.simulator = simulator
        self.enqueue_time: dict[tuple[int, int, int], float] = {}
        self.queue_intervals: list[QueueInterval] = []
        self.batch_intervals: list[BatchInterval] = []

    @staticmethod
    def _phase_name(request: Any) -> str:
        return getattr(request.phase, "name", str(request.phase))

    def install(self) -> None:
        for node in self.simulator.compute_nodes.values():
            original_receive = node.receive_request
            original_start = node.start_execution
            original_finish = node.finish_execution

            def receive_request(this, request, *, _orig=original_receive):
                _orig(request)
                layers = request.get_current_pipeline_stage().layers_to_infer
                first_layer = min(layers)
                self.enqueue_time[(request.request_uid, this.node_uid, first_layer)] = (
                    self.simulator.current_time
                )

            def start_execution(this, requests, *, _orig=original_start):
                start_s = self.simulator.current_time
                layer_id = this.current_layer_id
                for request in requests:
                    key = (request.request_uid, this.node_uid, layer_id)
                    enqueue_s = self.enqueue_time.pop(key, start_s)
                    self.queue_intervals.append(
                        QueueInterval(
                            request_uid=request.request_uid,
                            base_query_uid=request.base_query_uid,
                            phase=self._phase_name(request),
                            node_uid=this.node_uid,
                            layer_id=layer_id,
                            enqueue_s=enqueue_s,
                            start_s=start_s,
                        )
                    )
                batch = _orig(requests)
                self.batch_intervals.append(
                    BatchInterval(
                        node_uid=this.node_uid,
                        layer_id=layer_id,
                        start_s=start_s,
                        end_s=start_s + batch.duration,
                        duration_s=batch.duration,
                        request_uids=tuple(request.request_uid for request in batch.requests),
                        base_query_uids=tuple(
                            request.base_query_uid for request in batch.requests
                        ),
                        phases=tuple(self._phase_name(request) for request in batch.requests),
                    )
                )
                return batch

            def finish_execution(this, inference_batch_handle, *, _orig=original_finish):
                old_layer = this.current_layer_id
                batch, trigger_network_send = _orig(inference_batch_handle)
                if old_layer < max(this.in_vram_model_layers):
                    next_layer = old_layer + 1
                    now = self.simulator.current_time
                    for request in batch.requests:
                        self.enqueue_time[
                            (request.request_uid, this.node_uid, next_layer)
                        ] = now
                return batch, trigger_network_send

            node.receive_request = types.MethodType(receive_request, node)
            node.start_execution = types.MethodType(start_execution, node)
            node.finish_execution = types.MethodType(finish_execution, node)

    def summarize_request(
        self,
        *,
        request_uid: int,
        query_name_by_uid: dict[int, str],
    ) -> dict[str, Any]:
        waits = [item for item in self.queue_intervals if item.request_uid == request_uid]
        batches = [
            item for item in self.batch_intervals if request_uid in item.request_uids
        ]
        total_queue = sum(item.wait_s for item in waits)
        total_service = sum(item.duration_s for item in batches)

        blocker_phase_s = {"prefill": 0.0, "decode": 0.0, "mixed": 0.0, "uncovered": 0.0}
        blocker_query_s: dict[str, float] = {}
        wait_details: list[dict[str, Any]] = []
        for wait in waits:
            if wait.wait_s <= 1e-12:
                continue
            covered = 0.0
            local_blockers: list[dict[str, Any]] = []
            for batch in self.batch_intervals:
                if batch.node_uid != wait.node_uid:
                    continue
                overlap = max(
                    0.0,
                    min(wait.start_s, batch.end_s) - max(wait.enqueue_s, batch.start_s),
                )
                if overlap <= 0:
                    continue
                covered += overlap
                unique_phases = set(batch.phases)
                if unique_phases == {"Initialization"}:
                    kind = "prefill"
                elif unique_phases == {"Increment"}:
                    kind = "decode"
                else:
                    kind = "mixed"
                blocker_phase_s[kind] += overlap
                blocker_names = sorted(
                    {
                        query_name_by_uid.get(uid, str(uid))
                        for uid in batch.base_query_uids
                    }
                )
                for name in blocker_names:
                    blocker_query_s[name] = blocker_query_s.get(name, 0.0) + overlap
                local_blockers.append(
                    {
                        "kind": kind,
                        "overlap_s": overlap,
                        "queries": blocker_names,
                        "layer_id": batch.layer_id,
                    }
                )
            uncovered = max(wait.wait_s - covered, 0.0)
            blocker_phase_s["uncovered"] += uncovered
            wait_details.append(
                {
                    "node_uid": wait.node_uid,
                    "layer_id": wait.layer_id,
                    "wait_s": wait.wait_s,
                    "blockers": local_blockers,
                    "uncovered_s": uncovered,
                }
            )

        return {
            "queue_wait_s": total_queue,
            "layer_service_s": total_service,
            "blocker_phase_s": blocker_phase_s,
            "top_blocker_queries": sorted(
                blocker_query_s.items(), key=lambda item: item[1], reverse=True
            )[:8],
            "positive_wait_layers": sorted(
                wait_details, key=lambda item: item["wait_s"], reverse=True
            )[:12],
        }


def _run_one(*, helix_root: Path, pipeline_id: str, intensity: float) -> dict[str, Any]:
    pipelines = {pipeline.id: pipeline for pipeline in build_helix_pipelines()}
    pipeline = pipelines[pipeline_id]
    sample = build_helix_azure_conversation_workload(
        helix_root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = scale_workload(sample.requests, intensity)
    base_rate = (len(sample.requests) - 1) / (
        sample.requests[-1].arrival_time_s - sample.requests[0].arrival_time_s
    )

    runtime = _load_helix_runtime(helix_root)
    simulator, mini_pipeline, _ = _build_helix_simulator(
        pipeline=pipeline, runtime=runtime
    )
    tracer = HelixQueueTracer(simulator)
    tracer.install()

    base_time = simulator.current_time
    query_uid_to_request_id: dict[int, str] = {}
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

    events = 0
    while simulator.query_manager.queries_on_the_fly:
        succeeded, _ = simulator.simulate_next_event()
        events += 1
        if not succeeded:
            raise RuntimeError("HELIX event queue drained with unfinished queries")
        if events > 5_000_000:
            raise RuntimeError("HELIX queue diagnostic event limit exceeded")

    max_decode: dict[str, Any] | None = None
    per_query_max: dict[str, float] = {}
    for query_uid, (_, query) in simulator.query_manager.finished_queries.items():
        request_id = query_uid_to_request_id[query_uid]
        for decode_index, item in enumerate(query.inference_history[1:], start=1):
            duration = item.end_time - item.start_time
            per_query_max[request_id] = max(per_query_max.get(request_id, 0.0), duration)
            if max_decode is None or duration > max_decode["raw_tpot_s"]:
                max_decode = {
                    "request_id": request_id,
                    "query_uid": query_uid,
                    "decode_index": decode_index,
                    "request_uid": item.request_uid,
                    "raw_tpot_s": duration,
                    "start_s": item.start_time,
                    "end_s": item.end_time,
                }
    if max_decode is None:
        raise RuntimeError("no decode iteration found")

    request_summary = tracer.summarize_request(
        request_uid=max_decode["request_uid"],
        query_name_by_uid=query_uid_to_request_id,
    )
    residual = max_decode["raw_tpot_s"] - request_summary["queue_wait_s"] - request_summary["layer_service_s"]

    return {
        "pipeline_id": pipeline_id,
        "intensity": intensity,
        "arrival_rate_rps": intensity * base_rate,
        "requests": len(workload),
        "events": events,
        "max_decode_iteration": {
            **max_decode,
            "tpot_with_fixed_overhead_s": max_decode["raw_tpot_s"] + 0.005,
            **request_summary,
            "network_and_other_residual_s": residual,
        },
        "per_query_max_raw_tpot_s": per_query_max,
    }


def main() -> None:
    helix_root_env = os.environ.get("HELIX_ROOT")
    pipeline_id = os.environ.get("DIAG_PIPELINE")
    intensity_text = os.environ.get("DIAG_INTENSITY")
    if not helix_root_env or not pipeline_id or not intensity_text:
        raise RuntimeError("set HELIX_ROOT, DIAG_PIPELINE, and DIAG_INTENSITY")
    result = _run_one(
        helix_root=Path(helix_root_env),
        pipeline_id=pipeline_id,
        intensity=float(intensity_text),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
