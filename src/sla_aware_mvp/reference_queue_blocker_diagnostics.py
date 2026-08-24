from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .capacity import scale_workload
from .domain import Pipeline, RequestSpec, SLA
from .reference import (
    ReferenceConfig,
    ReferenceEvaluationResult,
    ReferenceTraceEvent,
    StageProfiler,
)
from .reference_diagnostics import ReferenceTraceRecorder, _failing_token_index


ReferenceEvaluator = Callable[..., ReferenceEvaluationResult]


@dataclass(frozen=True)
class StageQueueBlockerAttribution:
    stage_id: str
    gpu_queue_wait_s: float
    blocked_by_prefill_s: float
    blocked_by_decode_s: float
    idle_or_sync_s: float


@dataclass(frozen=True)
class QueueBlockerDiagnostic:
    pipeline_id: str
    intensity: float
    request_id: str
    token_index: int
    observed_tpot_s: float
    gpu_queue_wait_s: float
    blocked_by_prefill_s: float
    blocked_by_decode_s: float
    idle_or_sync_s: float
    per_stage: tuple[StageQueueBlockerAttribution, ...]


def _index_node_events(
    events: Sequence[ReferenceTraceEvent],
) -> dict[int, dict[str, ReferenceTraceEvent]]:
    indexed: dict[int, dict[str, ReferenceTraceEvent]] = {}
    for event in events:
        if event.operation_seq is None or event.resource_kind != "node":
            continue
        if event.event not in {"queue_enqueue", "service_start", "service_complete"}:
            continue
        indexed.setdefault(int(event.operation_seq), {})[event.event] = event
    return indexed


def _node_service_intervals(
    indexed: dict[int, dict[str, ReferenceTraceEvent]],
) -> tuple[tuple[str, float, float, str], ...]:
    # A Decode microbatch emits one event pair per member. Collapse identical
    # node/time/phase intervals so one physical service interval is counted once.
    intervals: set[tuple[str, float, float, str]] = set()
    for op_events in indexed.values():
        start = op_events.get("service_start")
        complete = op_events.get("service_complete")
        if start is None or complete is None:
            continue
        intervals.add(
            (start.resource_id, start.time_s, complete.time_s, start.phase)
        )
    return tuple(sorted(intervals))


def diagnose_gpu_queue_blockers(
    *,
    evaluator: ReferenceEvaluator,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    profiler: StageProfiler,
    intensity: float,
    config: ReferenceConfig | None = None,
) -> tuple[QueueBlockerDiagnostic, ReferenceEvaluationResult]:
    """Explain the failing Decode token's GPU queue wait by blocker phase.

    This is observational only. For each node queue interval of the first TPOT-
    violating token, it measures how much time overlaps an actual Prefill service,
    an actual Decode service, or no node service (idle/synchronization/policy wait).
    The three components close exactly to the measured GPU queue wait up to epsilon.
    """
    if intensity <= 0:
        raise ValueError("intensity must be positive")
    config = config or ReferenceConfig()
    recorder = ReferenceTraceRecorder()
    run = evaluator(
        pipeline=pipeline,
        workload=scale_workload(workload, intensity),
        sla=sla,
        profiler=profiler,
        config=config,
        trace_sink=recorder,
    )
    violation = run.first_violation
    if violation is None or violation.kind != "tpot" or violation.request_id is None:
        raise ValueError("queue-blocker diagnostic requires a TPOT-violating run")

    token_index = _failing_token_index(
        recorder.events,
        violation.request_id,
        violation.time_s,
        config.time_epsilon,
    )
    if token_index is None:
        raise RuntimeError("could not identify failing Decode token")

    token_operation_ids = sorted(
        {
            int(event.operation_seq)
            for event in recorder.events
            if event.request_id == violation.request_id
            and event.phase == "decode"
            and event.token_index == token_index
            and event.operation_seq is not None
            and event.resource_kind == "node"
        }
    )
    indexed = _index_node_events(recorder.events)
    service_intervals = _node_service_intervals(indexed)

    stage_results: list[StageQueueBlockerAttribution] = []
    total_queue = 0.0
    total_prefill = 0.0
    total_decode = 0.0
    total_idle = 0.0

    for operation_id in token_operation_ids:
        op_events = indexed.get(operation_id, {})
        enqueue = op_events.get("queue_enqueue")
        start = op_events.get("service_start")
        complete = op_events.get("service_complete")
        if enqueue is None or start is None or complete is None:
            raise RuntimeError("incomplete node operation in Reference trace")
        queue_start = enqueue.time_s
        queue_end = start.time_s
        wait = max(queue_end - queue_start, 0.0)
        blocked_prefill = 0.0
        blocked_decode = 0.0
        for resource_id, service_start, service_end, phase in service_intervals:
            if resource_id != start.resource_id:
                continue
            overlap = max(
                min(queue_end, service_end) - max(queue_start, service_start),
                0.0,
            )
            if overlap <= config.time_epsilon:
                continue
            if phase == "prefill":
                blocked_prefill += overlap
            elif phase == "decode":
                blocked_decode += overlap
        blocked = blocked_prefill + blocked_decode
        if blocked > wait + config.time_epsilon:
            raise RuntimeError("queue blocker attribution double-counted node service")
        idle_or_sync = max(wait - blocked, 0.0)
        if idle_or_sync <= config.time_epsilon:
            idle_or_sync = 0.0
        stage_results.append(
            StageQueueBlockerAttribution(
                stage_id=start.stage_id or "unknown-stage",
                gpu_queue_wait_s=wait,
                blocked_by_prefill_s=blocked_prefill,
                blocked_by_decode_s=blocked_decode,
                idle_or_sync_s=idle_or_sync,
            )
        )
        total_queue += wait
        total_prefill += blocked_prefill
        total_decode += blocked_decode
        total_idle += idle_or_sync

    closure = total_prefill + total_decode + total_idle
    if abs(total_queue - closure) > max(config.time_epsilon * 10, 1e-8):
        raise RuntimeError("GPU queue blocker attribution did not close")

    return (
        QueueBlockerDiagnostic(
            pipeline_id=pipeline.id,
            intensity=intensity,
            request_id=violation.request_id,
            token_index=token_index,
            observed_tpot_s=violation.observed,
            gpu_queue_wait_s=total_queue,
            blocked_by_prefill_s=total_prefill,
            blocked_by_decode_s=total_decode,
            idle_or_sync_s=total_idle,
            per_stage=tuple(stage_results),
        ),
        run,
    )
