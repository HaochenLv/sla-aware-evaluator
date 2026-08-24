from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capacity import scale_workload
from .domain import EvaluatorConfig, Phase, SLA
from .evaluator import evaluate
from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, build_helix_pipelines
from .helix_fixed_reference import _build_helix_simulator, _issue_fixed_query, _load_helix_runtime
from .prefill_debt_budget_ablation import (
    _budget_consistent_violation,
    _prefill_blocking_service_s,
)
from .workload import build_helix_azure_conversation_workload


SEED = 19
OFFSET = 0
CANDIDATE_UNSAFE = 0.0153
SLOW_SAFE_EDGE = 0.0183779296875
SLOW_UNSAFE_EDGE = 0.01838091796875
SLOW_POINTS = (CANDIDATE_UNSAFE, CANDIDATE_UNSAFE * 1.05, SLOW_SAFE_EDGE, SLOW_UNSAFE_EDGE)
FAST_POINTS = (CANDIDATE_UNSAFE, CANDIDATE_UNSAFE * 1.05)


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


class QueueTracer:
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
                first_layer = min(request.get_current_pipeline_stage().layers_to_infer)
                self.enqueue_time[(request.request_uid, this.node_uid, first_layer)] = self.simulator.current_time

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
                        base_query_uids=tuple(request.base_query_uid for request in batch.requests),
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
                        self.enqueue_time[(request.request_uid, this.node_uid, next_layer)] = now
                return batch, trigger_network_send

            node.receive_request = types.MethodType(receive_request, node)
            node.start_execution = types.MethodType(start_execution, node)
            node.finish_execution = types.MethodType(finish_execution, node)

    def summarize_iteration(self, request_uid: int, query_name_by_uid: dict[int, str]) -> dict[str, Any]:
        waits = [item for item in self.queue_intervals if item.request_uid == request_uid]
        batches = [item for item in self.batch_intervals if request_uid in item.request_uids]
        total_queue = sum(item.wait_s for item in waits)
        total_service = sum(item.duration_s for item in batches)
        blocker_phase_s = {"prefill": 0.0, "decode": 0.0, "mixed": 0.0, "uncovered": 0.0}
        blocker_query_s: dict[str, float] = {}
        for wait in waits:
            if wait.wait_s <= 1e-12:
                continue
            covered = 0.0
            for batch in self.batch_intervals:
                if batch.node_uid != wait.node_uid:
                    continue
                overlap = max(0.0, min(wait.start_s, batch.end_s) - max(wait.enqueue_s, batch.start_s))
                if overlap <= 0:
                    continue
                covered += overlap
                phases = set(batch.phases)
                if phases == {"Initialization"}:
                    kind = "prefill"
                elif phases == {"Increment"}:
                    kind = "decode"
                else:
                    kind = "mixed"
                blocker_phase_s[kind] += overlap
                for uid in set(batch.base_query_uids):
                    name = query_name_by_uid.get(uid, str(uid))
                    blocker_query_s[name] = blocker_query_s.get(name, 0.0) + overlap
            blocker_phase_s["uncovered"] += max(wait.wait_s - covered, 0.0)
        return {
            "queue_wait_s": total_queue,
            "layer_service_s": total_service,
            "blocker_phase_s": blocker_phase_s,
            "top_blocker_queries": sorted(blocker_query_s.items(), key=lambda item: item[1], reverse=True)[:8],
        }


def _snapshot_summary(snapshot) -> dict[str, Any]:
    return {
        "time_s": snapshot.time_s,
        "event_types": list(snapshot.event_types),
        "num_prefill": snapshot.num_prefill,
        "num_decode": snapshot.num_decode,
        "active_prefill_ids": sorted(rid for rid, phase in snapshot.request_phase.items() if phase == Phase.PREFILL.value),
        "active_decode_ids": sorted(rid for rid, phase in snapshot.request_phase.items() if phase == Phase.DECODE.value),
    }


def _evaluator_overlap(*, pipeline, workload, profiler, sla) -> dict[str, Any]:
    scaled = scale_workload(workload, CANDIDATE_UNSAFE)
    request_by_id = {request.id: request for request in scaled}
    run = evaluate(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        config=EvaluatorConfig(record_trace=True),
        profiler=profiler,
    )
    if not run.feasible:
        raise RuntimeError("base evaluator unexpectedly infeasible before E31 observational debt")

    for index, snapshot in enumerate(run.trace):
        violation = _budget_consistent_violation(
            snapshot=snapshot,
            request_by_id=request_by_id,
            pipeline=pipeline,
            profiler=profiler,
            sla=sla,
        )
        if violation is None:
            continue
        prefill_ids = [rid for rid, phase in snapshot.request_phase.items() if phase == Phase.PREFILL.value]
        decode_ids = [rid for rid, phase in snapshot.request_phase.items() if phase == Phase.DECODE.value]
        n_prefill = len(prefill_ids)
        n_decode = len(decode_ids)
        prefill_rows = []
        for rid in prefill_ids:
            request = request_by_id[rid]
            prefill_rows.append(
                {
                    "request_id": rid,
                    "arrival_time_s": request.arrival_time_s,
                    "age_at_overlap_s": snapshot.time_s - request.arrival_time_s,
                    "input_tokens": request.input_tokens,
                    "output_tokens": request.output_tokens,
                    "full_blocking_service_debt_s": _prefill_blocking_service_s(
                        request=request,
                        pipeline=pipeline,
                        profiler=profiler,
                        n_prefill=n_prefill,
                        n_decode=n_decode,
                    ),
                }
            )
        decode_rows = []
        for rid in decode_ids:
            request = request_by_id[rid]
            context = snapshot.request_context[rid]
            decode_rows.append(
                {
                    "request_id": rid,
                    "arrival_time_s": request.arrival_time_s,
                    "input_tokens": request.input_tokens,
                    "output_tokens": request.output_tokens,
                    "decode_progress": snapshot.request_progress[rid],
                    "context_tokens": context,
                    "decode_compute_s": profiler.decode_time_per_token(
                        request, context, pipeline, n_prefill, n_decode
                    ),
                }
            )
        return {
            "candidate_intensity": CANDIDATE_UNSAFE,
            "violation": violation,
            "snapshot": _snapshot_summary(snapshot),
            "previous_snapshot": _snapshot_summary(run.trace[index - 1]) if index > 0 else None,
            "active_prefills": prefill_rows,
            "active_decodes": decode_rows,
        }
    raise RuntimeError("E31 candidate violation not found")


def _iteration_record(*, item, request_id: str, tracer: QueueTracer, names: dict[int, str]) -> dict[str, Any]:
    duration = item.end_time - item.start_time
    summary = tracer.summarize_iteration(item.request_uid, names)
    return {
        "request_id": request_id,
        "request_uid": item.request_uid,
        "raw_tpot_s": duration,
        "tpot_plus_fixed_s": duration + 0.005,
        "start_s": item.start_time,
        "end_s": item.end_time,
        **summary,
        "network_and_other_residual_s": duration - summary["queue_wait_s"] - summary["layer_service_s"],
    }


def _helix_trace(*, root: Path, pipeline, workload, intensity: float, target_request_id: str) -> dict[str, Any]:
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

    global_max = None
    target_max = None
    for query_uid, (_, query) in simulator.query_manager.finished_queries.items():
        request_id = uid_to_name[query_uid]
        for item in query.inference_history[1:]:
            record = _iteration_record(item=item, request_id=request_id, tracer=tracer, names=uid_to_name)
            if global_max is None or record["raw_tpot_s"] > global_max["raw_tpot_s"]:
                global_max = record
            if request_id == target_request_id and (
                target_max is None or record["raw_tpot_s"] > target_max["raw_tpot_s"]
            ):
                target_max = record
    if global_max is None or target_max is None:
        raise RuntimeError("missing decode iteration in HELIX diagnostic")
    return {
        "pipeline_id": pipeline.id,
        "intensity": intensity,
        "events": events,
        "target_request_max_decode": target_max,
        "global_max_decode": global_max,
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

    evaluator = _evaluator_overlap(
        pipeline=slow,
        workload=workload,
        profiler=profiler,
        sla=sla,
    )
    target_request_id = evaluator["violation"]["request_id"]

    helix = {
        "slow": [
            _helix_trace(
                root=root,
                pipeline=slow,
                workload=workload,
                intensity=value,
                target_request_id=target_request_id,
            )
            for value in SLOW_POINTS
        ],
        "fast": [
            _helix_trace(
                root=root,
                pipeline=fast,
                workload=workload,
                intensity=value,
                target_request_id=target_request_id,
            )
            for value in FAST_POINTS
        ],
    }
    result = {
        "design": {
            "purpose": "diagnose why seed19 Slow is about 21% more conservative than pinned HELIX without changing evaluator semantics",
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
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
