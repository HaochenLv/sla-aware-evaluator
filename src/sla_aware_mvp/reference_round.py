from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Mapping, Sequence

from .capacity import scale_workload
from .domain import Phase, Pipeline, RequestSpec, SLA, Stage, sorted_workload
from .reference import (
    ReferenceCapacityResult,
    ReferenceCapacityTrial,
    ReferenceConfig,
    ReferenceEvaluationResult,
    ReferenceTraceEvent,
    ReferenceTraceSink,
    ReferenceViolation,
    StageProfiler,
)


@dataclass
class _RequestState:
    spec: RequestSpec
    phase: Phase = Phase.PREFILL
    completed_tokens: int = 0
    token_ready_s: float | None = None
    ttft_s: float | None = None
    max_tpot_s: float = 0.0


@dataclass(frozen=True)
class _ComputeOp:
    request_id: str
    phase: Phase
    stage_index: int
    token_index: int | None
    enqueue_seq: int
    batch_id: int | None = None


@dataclass(frozen=True)
class _DecodeBatch:
    batch_id: int
    stage_index: int
    members: tuple[_ComputeOp, ...]


@dataclass(frozen=True)
class _LinkOp:
    request_id: str
    phase: Phase
    stage_index: int
    token_index: int | None
    route_link_ids: tuple[str, ...]
    route_position: int
    enqueue_seq: int
    batch_id: int | None = None


@dataclass
class _NodeServer:
    queue: Deque[_ComputeOp | _DecodeBatch] = field(default_factory=deque)
    busy: bool = False


@dataclass
class _LinkServer:
    queue: Deque[_LinkOp] = field(default_factory=deque)
    busy: bool = False


def _static_memory_by_node(pipeline: Pipeline) -> dict[str, float]:
    layers_by_node = pipeline.layers_by_node()
    result: dict[str, float] = {}
    for node_id, node in pipeline.nodes.items():
        layer_fraction = layers_by_node[node_id] / pipeline.model.num_layers
        runtime_reserve = (
            node.workspace_bytes + node.memory_margin_bytes
            if layers_by_node[node_id] > 0
            else 0
        )
        result[node_id] = pipeline.model.weight_bytes * layer_fraction + runtime_reserve
    return result


def _counts(states: Mapping[str, _RequestState]) -> tuple[int, int]:
    return (
        sum(state.phase == Phase.PREFILL for state in states.values()),
        sum(state.phase == Phase.DECODE for state in states.values()),
    )


def _memory_by_node(
    states: Mapping[str, _RequestState], pipeline: Pipeline
) -> dict[str, float]:
    memory = _static_memory_by_node(pipeline)
    layers_by_node = pipeline.layers_by_node()
    for state in states.values():
        context = (
            state.spec.input_tokens
            if state.phase == Phase.PREFILL
            else state.spec.input_tokens + state.completed_tokens
        )
        for node_id, num_layers in layers_by_node.items():
            memory[node_id] += (
                pipeline.model.kv_bytes_per_token_per_layer * context * num_layers
            )
    return memory


def evaluate_reference_round(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    profiler: StageProfiler,
    config: ReferenceConfig | None = None,
    trace_sink: ReferenceTraceSink | None = None,
) -> ReferenceEvaluationResult:
    """Explicit-service reference with deterministic end-to-end Decode rounds.

    At most one Decode cohort may be in flight end-to-end. Requests that become
    Decode-ready during a round wait and join the next round. Prefill FIFO,
    explicit FCFS links, profiler inputs, SLA checks, and memory accounting are
    unchanged from Reference v1.
    """
    config = config or ReferenceConfig()
    pipeline.validate()
    requests = sorted_workload(workload)
    if not requests:
        return ReferenceEvaluationResult(
            True, None, (), 0.0, 0, 0, 0, 0, 0, {}, {}, {}
        )

    ordered_stages = tuple(
        sorted(pipeline.stages, key=lambda stage: stage.layer_start)
    )
    stage0_node_id = ordered_stages[0].node_id
    boundary_by_pair = {
        (boundary.upstream_stage_id, boundary.downstream_stage_id): boundary
        for boundary in pipeline.boundaries
    }
    for boundary in pipeline.boundaries:
        for link_id in boundary.route_link_ids:
            if pipeline.links[link_id].capacity_bytes_per_s <= 0:
                raise ValueError("reference evaluator requires positive link capacity")

    node_servers = {node_id: _NodeServer() for node_id in pipeline.nodes}
    link_servers = {link_id: _LinkServer() for link_id in pipeline.links}
    states: dict[str, _RequestState] = {}
    ready_decode: dict[str, _ComputeOp] = {}
    batch_expected: dict[int, tuple[str, ...]] = {}
    batch_stage_arrivals: dict[tuple[int, int], dict[str, _ComputeOp]] = {}

    peak_memory = {node_id: 0.0 for node_id in pipeline.nodes}
    ttft_by_request: dict[str, float] = {}
    max_tpot_by_request: dict[str, float] = {}
    completed_requests = 0
    completed_tokens = 0
    peak_prefill = 0
    peak_decode = 0
    processed_events = 0

    completion_heap: list[tuple[float, int, str, str, str, object]] = []
    seq = 0
    enqueue_seq = 0
    batch_seq = 0
    inflight_batch_id: int | None = None
    batch_remaining: dict[int, set[str]] = {}
    arrival_index = 0
    current_time = requests[0].arrival_time_s

    def next_enqueue_seq() -> int:
        nonlocal enqueue_seq
        enqueue_seq += 1
        return enqueue_seq

    def emit_compute(
        *,
        event: str,
        operation: _ComputeOp,
        resource_kind: str,
        resource_id: str,
        batch_size: int | None = None,
    ) -> None:
        if trace_sink is None:
            return
        trace_sink.record(
            ReferenceTraceEvent(
                time_s=current_time,
                event=event,
                resource_kind=resource_kind,
                resource_id=resource_id,
                request_id=operation.request_id,
                phase=operation.phase.value,
                stage_id=ordered_stages[operation.stage_index].id,
                token_index=operation.token_index,
                operation_seq=operation.enqueue_seq,
                batch_size=batch_size,
            )
        )

    def emit_link(
        *,
        event: str,
        operation: _LinkOp,
        resource_id: str,
        batch_size: int | None = None,
    ) -> None:
        if trace_sink is None:
            return
        trace_sink.record(
            ReferenceTraceEvent(
                time_s=current_time,
                event=event,
                resource_kind="link",
                resource_id=resource_id,
                request_id=operation.request_id,
                phase=operation.phase.value,
                stage_id=ordered_stages[operation.stage_index].id,
                token_index=operation.token_index,
                operation_seq=operation.enqueue_seq,
                batch_size=batch_size,
            )
        )

    def emit_token(
        *,
        event: str,
        request_id: str,
        token_index: int,
        operation_seq: int | None,
    ) -> None:
        if trace_sink is None:
            return
        trace_sink.record(
            ReferenceTraceEvent(
                time_s=current_time,
                event=event,
                resource_kind="token",
                resource_id="",
                request_id=request_id,
                phase=Phase.DECODE.value,
                stage_id=ordered_stages[0].id,
                token_index=token_index,
                operation_seq=operation_seq,
                batch_size=None,
            )
        )

    def push_event(
        when: float,
        event_kind: str,
        resource_kind: str,
        resource_id: str,
        payload: object,
    ) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(
            completion_heap,
            (when, seq, event_kind, resource_kind, resource_id, payload),
        )

    def enqueue_prefill(request_id: str, stage_index: int) -> None:
        op = _ComputeOp(
            request_id,
            Phase.PREFILL,
            stage_index,
            None,
            next_enqueue_seq(),
        )
        stage = ordered_stages[stage_index]
        node_servers[stage.node_id].queue.append(op)
        emit_compute(
            event="queue_enqueue",
            operation=op,
            resource_kind="node",
            resource_id=stage.node_id,
        )

    def mark_decode_ready(request_id: str) -> None:
        state = states[request_id]
        token_index = state.completed_tokens
        state.token_ready_s = current_time
        op = _ComputeOp(
            request_id,
            Phase.DECODE,
            0,
            token_index,
            next_enqueue_seq(),
        )
        ready_decode[request_id] = op
        emit_token(
            event="token_start",
            request_id=request_id,
            token_index=token_index,
            operation_seq=None,
        )
        emit_compute(
            event="queue_enqueue",
            operation=op,
            resource_kind="node",
            resource_id=stage0_node_id,
        )

    def enqueue_link(
        *,
        request_id: str,
        phase: Phase,
        stage_index: int,
        token_index: int | None,
        route_link_ids: tuple[str, ...],
        route_position: int,
        batch_id: int | None,
    ) -> None:
        link_id = route_link_ids[route_position]
        op = _LinkOp(
            request_id,
            phase,
            stage_index,
            token_index,
            route_link_ids,
            route_position,
            next_enqueue_seq(),
            batch_id,
        )
        link_servers[link_id].queue.append(op)
        emit_link(event="queue_enqueue", operation=op, resource_id=link_id)

    def arrive_decode_stage(
        *, request_id: str, token_index: int, batch_id: int, stage_index: int
    ) -> None:
        op = _ComputeOp(
            request_id,
            Phase.DECODE,
            stage_index,
            token_index,
            next_enqueue_seq(),
            batch_id,
        )
        stage = ordered_stages[stage_index]
        emit_compute(
            event="queue_enqueue",
            operation=op,
            resource_kind="node",
            resource_id=stage.node_id,
        )
        key = (batch_id, stage_index)
        arrived = batch_stage_arrivals.setdefault(key, {})
        arrived[request_id] = op
        expected = batch_expected[batch_id]
        if all(member_id in arrived for member_id in expected):
            members = tuple(arrived[member_id] for member_id in expected)
            node_servers[stage.node_id].queue.append(
                _DecodeBatch(batch_id, stage_index, members)
            )
            del batch_stage_arrivals[key]

    def route_after_prefill(op: _ComputeOp) -> None:
        if op.stage_index >= len(ordered_stages) - 1:
            push_event(
                current_time + sla.queue_overhead_s + sla.fixed_overhead_s,
                "phase_finalize",
                "timer",
                "",
                op,
            )
            return
        upstream = ordered_stages[op.stage_index]
        downstream = ordered_stages[op.stage_index + 1]
        boundary = boundary_by_pair[(upstream.id, downstream.id)]
        if boundary.route_link_ids:
            enqueue_link(
                request_id=op.request_id,
                phase=Phase.PREFILL,
                stage_index=op.stage_index,
                token_index=None,
                route_link_ids=boundary.route_link_ids,
                route_position=0,
                batch_id=None,
            )
        else:
            enqueue_prefill(op.request_id, op.stage_index + 1)

    def route_after_decode(op: _ComputeOp) -> None:
        if op.batch_id is None or op.token_index is None:
            raise RuntimeError("decode operation missing batch/token identity")
        if op.stage_index >= len(ordered_stages) - 1:
            push_event(
                current_time + sla.queue_overhead_s + sla.fixed_overhead_s,
                "phase_finalize",
                "timer",
                "",
                op,
            )
            return
        upstream = ordered_stages[op.stage_index]
        downstream = ordered_stages[op.stage_index + 1]
        boundary = boundary_by_pair[(upstream.id, downstream.id)]
        if boundary.route_link_ids:
            enqueue_link(
                request_id=op.request_id,
                phase=Phase.DECODE,
                stage_index=op.stage_index,
                token_index=op.token_index,
                route_link_ids=boundary.route_link_ids,
                route_position=0,
                batch_id=op.batch_id,
            )
        else:
            arrive_decode_stage(
                request_id=op.request_id,
                token_index=op.token_index,
                batch_id=op.batch_id,
                stage_index=op.stage_index + 1,
            )

    def finalize_operation(
        op: _ComputeOp, batch_violations: list[ReferenceViolation]
    ) -> None:
        nonlocal completed_requests, completed_tokens, inflight_batch_id
        state = states[op.request_id]
        if op.phase == Phase.PREFILL:
            observed = current_time - state.spec.arrival_time_s
            state.ttft_s = observed
            ttft_by_request[state.spec.id] = observed
            if observed > sla.ttft_s + config.time_epsilon:
                batch_violations.append(
                    ReferenceViolation(
                        current_time,
                        "ttft",
                        "request",
                        observed,
                        sla.ttft_s,
                        state.spec.id,
                    )
                )
            state.phase = Phase.DECODE
            mark_decode_ready(state.spec.id)
            return

        if state.token_ready_s is None or op.token_index is None:
            raise RuntimeError("decode token completed without ready time")
        observed = current_time - state.token_ready_s
        emit_token(
            event="token_complete",
            request_id=state.spec.id,
            token_index=op.token_index,
            operation_seq=op.enqueue_seq,
        )
        state.max_tpot_s = max(state.max_tpot_s, observed)
        max_tpot_by_request[state.spec.id] = state.max_tpot_s
        if observed > sla.tpot_s + config.time_epsilon:
            batch_violations.append(
                ReferenceViolation(
                    current_time,
                    "tpot",
                    "request",
                    observed,
                    sla.tpot_s,
                    state.spec.id,
                )
            )
        state.completed_tokens += 1
        completed_tokens += 1
        if state.completed_tokens >= state.spec.output_tokens:
            completed_requests += 1
            ready_decode.pop(state.spec.id, None)
            del states[state.spec.id]
        else:
            mark_decode_ready(state.spec.id)

        if op.batch_id is None:
            raise RuntimeError("decode completion missing batch identity")
        remaining = batch_remaining[op.batch_id]
        remaining.discard(op.request_id)
        if not remaining:
            del batch_remaining[op.batch_id]
            if inflight_batch_id != op.batch_id:
                raise RuntimeError("decode round identity mismatch")
            inflight_batch_id = None

    def queue_head_seq(server: _NodeServer) -> int | None:
        if not server.queue:
            return None
        item = server.queue[0]
        if isinstance(item, _DecodeBatch):
            return min(member.enqueue_seq for member in item.members)
        return item.enqueue_seq

    def start_node_item(
        node_id: str, item: _ComputeOp | _DecodeBatch
    ) -> None:
        server = node_servers[node_id]
        server.busy = True
        n_prefill, n_decode_active = _counts(states)
        if isinstance(item, _DecodeBatch):
            members = item.members
            stage = ordered_stages[item.stage_index]
            batch_size = len(members)
            durations: list[float] = []
            for op in members:
                state = states[op.request_id]
                if op.token_index is None:
                    raise RuntimeError("decode batch member missing token index")
                context = state.spec.input_tokens + op.token_index
                durations.append(
                    profiler.decode_stage_time_per_token(
                        state.spec,
                        context,
                        pipeline,
                        stage,
                        n_prefill,
                        batch_size,
                    )
                )
            service_s = max(durations, default=0.0)
            if service_s < 0:
                raise ValueError("profile service time cannot be negative")
            for op in members:
                emit_compute(
                    event="service_start",
                    operation=op,
                    resource_kind="node",
                    resource_id=node_id,
                    batch_size=batch_size,
                )
            push_event(
                current_time + service_s,
                "server_complete",
                "node",
                node_id,
                item,
            )
            return

        stage = ordered_stages[item.stage_index]
        state = states[item.request_id]
        service_s = profiler.prefill_stage_time(
            state.spec,
            pipeline,
            stage,
            n_prefill,
            n_decode_active,
        )
        if service_s < 0:
            raise ValueError("profile service time cannot be negative")
        emit_compute(
            event="service_start",
            operation=item,
            resource_kind="node",
            resource_id=node_id,
            batch_size=1,
        )
        push_event(
            current_time + service_s,
            "server_complete",
            "node",
            node_id,
            item,
        )

    def start_idle_servers() -> None:
        nonlocal batch_seq, inflight_batch_id
        for node_id in sorted(node_servers):
            server = node_servers[node_id]
            if server.busy:
                continue
            item: _ComputeOp | _DecodeBatch | None = None
            if (
                node_id == stage0_node_id
                and ready_decode
                and inflight_batch_id is None
            ):
                head_seq = queue_head_seq(server)
                eligible = [
                    op
                    for op in ready_decode.values()
                    if head_seq is None or op.enqueue_seq < head_seq
                ]
                if eligible:
                    eligible.sort(key=lambda op: (op.enqueue_seq, op.request_id))
                    batch_seq += 1
                    batch_id = batch_seq
                    members = tuple(
                        _ComputeOp(
                            op.request_id,
                            Phase.DECODE,
                            0,
                            op.token_index,
                            op.enqueue_seq,
                            batch_id,
                        )
                        for op in eligible
                    )
                    for op in eligible:
                        del ready_decode[op.request_id]
                    batch_expected[batch_id] = tuple(
                        op.request_id for op in members
                    )
                    batch_remaining[batch_id] = set(batch_expected[batch_id])
                    inflight_batch_id = batch_id
                    item = _DecodeBatch(batch_id, 0, members)
            if item is None and server.queue:
                item = server.queue.popleft()
            if item is not None:
                start_node_item(node_id, item)

        for link_id in sorted(link_servers):
            server = link_servers[link_id]
            if server.busy or not server.queue:
                continue
            op = server.queue.popleft()
            server.busy = True
            state = states[op.request_id]
            transfer_tokens = (
                state.spec.input_tokens if op.phase == Phase.PREFILL else 1
            )
            data_bytes = (
                pipeline.model.activation_bytes_per_token * transfer_tokens
            )
            service_s = (
                data_bytes / pipeline.links[link_id].capacity_bytes_per_s
            )
            emit_link(
                event="service_start",
                operation=op,
                resource_id=link_id,
                batch_size=1,
            )
            push_event(
                current_time + service_s,
                "server_complete",
                "link",
                link_id,
                op,
            )

    while arrival_index < len(requests) or states or completion_heap:
        if processed_events >= config.max_events:
            raise RuntimeError("reference event limit exceeded")

        next_arrival = (
            requests[arrival_index].arrival_time_s
            if arrival_index < len(requests)
            else float("inf")
        )
        next_completion = (
            completion_heap[0][0] if completion_heap else float("inf")
        )
        next_time = min(next_arrival, next_completion)
        if next_time == float("inf"):
            if states:
                raise RuntimeError("reference round deadlocked with active requests")
            break
        if next_time < current_time - config.time_epsilon:
            raise RuntimeError("reference event time moved backwards")
        current_time = next_time
        batch_violations: list[ReferenceViolation] = []

        due: list[tuple[float, int, str, str, str, object]] = []
        while (
            completion_heap
            and completion_heap[0][0] <= current_time + config.time_epsilon
        ):
            due.append(heapq.heappop(completion_heap))

        for _, _, event_kind, resource_kind, resource_id, _ in due:
            if event_kind != "server_complete":
                continue
            if resource_kind == "node":
                node_servers[resource_id].busy = False
            elif resource_kind == "link":
                link_servers[resource_id].busy = False

        for _, _, event_kind, resource_kind, resource_id, payload in due:
            processed_events += 1
            if event_kind == "phase_finalize":
                finalize_operation(payload, batch_violations)  # type: ignore[arg-type]
                continue

            if resource_kind == "node":
                if isinstance(payload, _DecodeBatch):
                    for op in payload.members:
                        emit_compute(
                            event="service_complete",
                            operation=op,
                            resource_kind="node",
                            resource_id=resource_id,
                            batch_size=len(payload.members),
                        )
                        route_after_decode(op)
                else:
                    op = payload
                    emit_compute(
                        event="service_complete",
                        operation=op,
                        resource_kind="node",
                        resource_id=resource_id,
                        batch_size=1,
                    )
                    route_after_prefill(op)
            else:
                op = payload
                emit_link(
                    event="service_complete",
                    operation=op,
                    resource_id=resource_id,
                    batch_size=1,
                )
                if op.route_position + 1 < len(op.route_link_ids):
                    enqueue_link(
                        request_id=op.request_id,
                        phase=op.phase,
                        stage_index=op.stage_index,
                        token_index=op.token_index,
                        route_link_ids=op.route_link_ids,
                        route_position=op.route_position + 1,
                        batch_id=op.batch_id,
                    )
                elif op.phase == Phase.PREFILL:
                    enqueue_prefill(op.request_id, op.stage_index + 1)
                else:
                    if op.batch_id is None or op.token_index is None:
                        raise RuntimeError(
                            "decode link operation missing batch/token identity"
                        )
                    arrive_decode_stage(
                        request_id=op.request_id,
                        token_index=op.token_index,
                        batch_id=op.batch_id,
                        stage_index=op.stage_index + 1,
                    )

        while (
            completion_heap
            and completion_heap[0][0] <= current_time + config.time_epsilon
            and completion_heap[0][2] == "phase_finalize"
        ):
            _, _, _, _, _, payload = heapq.heappop(completion_heap)
            processed_events += 1
            finalize_operation(payload, batch_violations)  # type: ignore[arg-type]

        arrivals: list[RequestSpec] = []
        while (
            arrival_index < len(requests)
            and requests[arrival_index].arrival_time_s
            <= current_time + config.time_epsilon
        ):
            arrivals.append(requests[arrival_index])
            arrival_index += 1
        for request in sorted(arrivals, key=lambda item: item.id):
            processed_events += 1
            states[request.id] = _RequestState(request)
            enqueue_prefill(request.id, 0)

        n_prefill, n_decode = _counts(states)
        peak_prefill = max(peak_prefill, n_prefill)
        peak_decode = max(peak_decode, n_decode)
        memory = _memory_by_node(states, pipeline)
        for node_id, used in memory.items():
            peak_memory[node_id] = max(peak_memory[node_id], used)
            capacity = pipeline.nodes[node_id].memory_capacity_bytes
            if used > capacity:
                batch_violations.append(
                    ReferenceViolation(
                        current_time,
                        "memory",
                        node_id,
                        used,
                        capacity,
                        None,
                    )
                )

        if batch_violations:
            batch_violations.sort(
                key=lambda item: (item.kind, item.object_id, item.request_id or "")
            )
            return ReferenceEvaluationResult(
                False,
                batch_violations[0],
                tuple(batch_violations),
                current_time,
                processed_events,
                completed_requests,
                completed_tokens,
                peak_prefill,
                peak_decode,
                peak_memory,
                ttft_by_request,
                max_tpot_by_request,
            )

        start_idle_servers()

    return ReferenceEvaluationResult(
        True,
        None,
        (),
        current_time,
        processed_events,
        completed_requests,
        completed_tokens,
        peak_prefill,
        peak_decode,
        peak_memory,
        ttft_by_request,
        max_tpot_by_request,
    )


def find_reference_capacity_round(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    profiler: StageProfiler,
    config: ReferenceConfig | None = None,
    initial_intensity: float = 0.25,
    tolerance: float = 0.02,
    max_intensity: float = 16.0,
    verification_grid_points: int = 9,
) -> ReferenceCapacityResult:
    if initial_intensity <= 0 or tolerance <= 0 or max_intensity <= 0:
        raise ValueError("capacity-search parameters must be positive")
    if verification_grid_points < 3:
        raise ValueError("verification_grid_points must be at least 3")
    config = config or ReferenceConfig()
    trials: list[ReferenceCapacityTrial] = []
    cache: dict[float, ReferenceEvaluationResult] = {}

    def run(intensity: float) -> ReferenceEvaluationResult:
        key = float(intensity)
        if key not in cache:
            result = evaluate_reference_round(
                pipeline=pipeline,
                workload=scale_workload(workload, key),
                sla=sla,
                config=config,
                profiler=profiler,
            )
            cache[key] = result
            trials.append(
                ReferenceCapacityTrial(
                    key,
                    result.feasible,
                    result.first_violation.kind
                    if result.first_violation
                    else None,
                )
            )
        return cache[key]

    high = min(initial_intensity, max_intensity)
    high_run = run(high)
    if high_run.feasible:
        low = high
        while high < max_intensity:
            candidate = min(high * 2.0, max_intensity)
            if candidate == high:
                break
            candidate_run = run(candidate)
            if not candidate_run.feasible:
                high = candidate
                high_run = candidate_run
                break
            low = candidate
            high = candidate
        if high_run.feasible:
            sampled = tuple(
                trial.intensity
                for trial in sorted(trials, key=lambda item: item.intensity)
            )
            return ReferenceCapacityResult(
                high,
                None,
                tuple(trials),
                high_run,
                None,
                True,
                sampled,
                True,
                max_intensity,
            )
    else:
        unsafe_anchor = high
        unsafe_anchor_run = high_run
        low = high / 2.0
        while low > 1e-9:
            low_run = run(low)
            if low_run.feasible:
                high = unsafe_anchor
                high_run = unsafe_anchor_run
                break
            unsafe_anchor = low
            unsafe_anchor_run = low_run
            low /= 2.0
        else:
            raise RuntimeError("no safe positive reference intensity found")

    bracket_low, bracket_high = low, high
    probe_intensities: list[float] = []
    for index in range(1, verification_grid_points):
        intensity = bracket_low + (bracket_high - bracket_low) * (
            index / verification_grid_points
        )
        probe_intensities.append(intensity)
        run(intensity)

    probe = bracket_high * 2.0
    while (
        probe <= max_intensity
        and len(probe_intensities) < verification_grid_points + 4
    ):
        probe_intensities.append(probe)
        run(probe)
        probe *= 2.0

    ordered = sorted(trials, key=lambda trial: trial.intensity)
    seen_unsafe = False
    for trial in ordered:
        if not trial.feasible:
            seen_unsafe = True
        elif seen_unsafe:
            raise RuntimeError(
                "non-monotonic reference feasibility observed on sampled intensities"
            )

    safe_trials = [trial for trial in ordered if trial.feasible]
    unsafe_trials = [trial for trial in ordered if not trial.feasible]
    if not safe_trials or not unsafe_trials:
        raise RuntimeError("reference capacity search failed to retain a bracket")
    low = max(trial.intensity for trial in safe_trials)
    high = min(
        trial.intensity for trial in unsafe_trials if trial.intensity > low
    )
    safe_run = run(low)
    high_run = run(high)

    while high - low > tolerance * max(high, 1.0):
        mid = (low + high) / 2.0
        mid_run = run(mid)
        if mid_run.feasible:
            low, safe_run = mid, mid_run
        else:
            high, high_run = mid, mid_run

    return ReferenceCapacityResult(
        low,
        high,
        tuple(trials),
        safe_run,
        high_run,
        True,
        tuple(probe_intensities),
        False,
        max_intensity,
    )
