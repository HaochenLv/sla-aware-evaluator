from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Mapping, Protocol, Sequence

from .capacity import scale_workload
from .domain import Phase, Pipeline, RequestSpec, SLA, Stage, sorted_workload


class StageProfiler(Protocol):
    def prefill_stage_time(
        self,
        request: RequestSpec,
        pipeline: Pipeline,
        stage: Stage,
        n_prefill: int,
        n_decode: int,
    ) -> float: ...

    def decode_stage_time_per_token(
        self,
        request: RequestSpec,
        context_tokens: int,
        pipeline: Pipeline,
        stage: Stage,
        n_prefill: int,
        n_decode: int,
    ) -> float: ...


@dataclass(frozen=True)
class ReferenceConfig:
    """Execution policy for the independent reference evaluator.

    v0 deliberately does not reuse the Conservative Evaluator's
    SLA->required-bandwidth reservation formula. It models explicit GPU service
    and explicit FCFS link transfers. Decode compute uses a deterministic
    microbatch rule: when a GPU becomes idle and the FIFO head is Decode, the
    contiguous Decode operations for that same stage are served as one batch.
    """

    time_epsilon: float = 1e-9
    max_events: int = 5_000_000


@dataclass(frozen=True)
class ReferenceViolation:
    time_s: float
    kind: str
    object_id: str
    observed: float
    limit: float
    request_id: str | None = None


@dataclass(frozen=True)
class ReferenceEvaluationResult:
    feasible: bool
    first_violation: ReferenceViolation | None
    first_violations: tuple[ReferenceViolation, ...]
    final_time_s: float
    processed_events: int
    completed_requests: int
    completed_tokens: int
    peak_prefill: int
    peak_decode: int
    peak_memory_bytes: Mapping[str, float]
    ttft_s_by_request: Mapping[str, float]
    max_tpot_s_by_request: Mapping[str, float]


@dataclass(frozen=True)
class ReferenceCapacityTrial:
    intensity: float
    feasible: bool
    violation_kind: str | None


@dataclass(frozen=True)
class ReferenceCapacityResult:
    safe_intensity: float
    unsafe_intensity: float
    trials: tuple[ReferenceCapacityTrial, ...]
    representative_safe_run: ReferenceEvaluationResult
    representative_unsafe_run: ReferenceEvaluationResult
    monotonicity_verified_on_samples: bool
    verification_probe_intensities: tuple[float, ...]


@dataclass
class _RequestState:
    spec: RequestSpec
    phase: Phase = Phase.PREFILL
    completed_tokens: int = 0
    token_start_s: float | None = None
    ttft_s: float | None = None
    max_tpot_s: float = 0.0


@dataclass(frozen=True)
class _Operation:
    request_id: str
    phase: Phase
    kind: str  # compute | link
    stage_index: int
    token_index: int | None = None
    route_link_ids: tuple[str, ...] = ()
    route_position: int = 0
    enqueue_seq: int = 0


@dataclass
class _Server:
    queue: Deque[_Operation] = field(default_factory=deque)
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
        if state.phase == Phase.PREFILL:
            context = state.spec.input_tokens
        else:
            context = state.spec.input_tokens + state.completed_tokens
        for node_id, num_layers in layers_by_node.items():
            memory[node_id] += (
                pipeline.model.kv_bytes_per_token_per_layer * context * num_layers
            )
    return memory


def evaluate_reference(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    profiler: StageProfiler,
    config: ReferenceConfig | None = None,
) -> ReferenceEvaluationResult:
    """Explicit-service finite-workload reference evaluator v0.

    Independence from the Conservative Evaluator is intentional:

    - GPU stages are queued servers rather than a residual-SLA reservation.
    - physical links are queued FCFS byte-transfer servers with service D/B.
    - Decode GPU work uses deterministic queued microbatches rather than active-
      request bandwidth commitments.
    - TTFT is aligned with the current research model and measured at end of
      Prefill (plus configured fixed/queue overhead).
    - strong TPOT is measured as every end-to-end Decode-token interval.

    This is a reference *simulation*, not a claim to reproduce vLLM/HELIX runtime
    scheduling exactly. Its role is to provide a mechanically independent
    ordering target for the Conservative Evaluator.
    """

    config = config or ReferenceConfig()
    pipeline.validate()
    requests = sorted_workload(workload)
    if not requests:
        return ReferenceEvaluationResult(
            True, None, (), 0.0, 0, 0, 0, 0, 0, {}, {}, {}
        )

    ordered_stages = tuple(sorted(pipeline.stages, key=lambda stage: stage.layer_start))
    boundary_by_pair = {
        (boundary.upstream_stage_id, boundary.downstream_stage_id): boundary
        for boundary in pipeline.boundaries
    }
    for boundary in pipeline.boundaries:
        for link_id in boundary.route_link_ids:
            if pipeline.links[link_id].capacity_bytes_per_s <= 0:
                raise ValueError("reference evaluator requires positive link capacity")

    node_servers = {node_id: _Server() for node_id in pipeline.nodes}
    link_servers = {link_id: _Server() for link_id in pipeline.links}
    states: dict[str, _RequestState] = {}
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
    arrival_index = 0
    current_time = requests[0].arrival_time_s

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

    def enqueue_compute(
        request_id: str,
        phase: Phase,
        stage_index: int,
        token_index: int | None,
    ) -> None:
        nonlocal enqueue_seq
        enqueue_seq += 1
        stage = ordered_stages[stage_index]
        node_servers[stage.node_id].queue.append(
            _Operation(
                request_id=request_id,
                phase=phase,
                kind="compute",
                stage_index=stage_index,
                token_index=token_index,
                enqueue_seq=enqueue_seq,
            )
        )

    def enqueue_link(
        request_id: str,
        phase: Phase,
        stage_index: int,
        token_index: int | None,
        route_link_ids: tuple[str, ...],
        route_position: int,
    ) -> None:
        nonlocal enqueue_seq
        enqueue_seq += 1
        link_id = route_link_ids[route_position]
        link_servers[link_id].queue.append(
            _Operation(
                request_id=request_id,
                phase=phase,
                kind="link",
                stage_index=stage_index,
                token_index=token_index,
                route_link_ids=route_link_ids,
                route_position=route_position,
                enqueue_seq=enqueue_seq,
            )
        )

    def route_after_stage(operation: _Operation) -> None:
        if operation.stage_index >= len(ordered_stages) - 1:
            overhead = sla.queue_overhead_s + sla.fixed_overhead_s
            push_event(
                current_time + overhead,
                "phase_finalize",
                "timer",
                "",
                operation,
            )
            return
        upstream = ordered_stages[operation.stage_index]
        downstream = ordered_stages[operation.stage_index + 1]
        boundary = boundary_by_pair[(upstream.id, downstream.id)]
        if boundary.route_link_ids:
            enqueue_link(
                operation.request_id,
                operation.phase,
                operation.stage_index,
                operation.token_index,
                boundary.route_link_ids,
                0,
            )
        else:
            enqueue_compute(
                operation.request_id,
                operation.phase,
                operation.stage_index + 1,
                operation.token_index,
            )

    def start_idle_servers() -> None:
        n_prefill, n_decode_active = _counts(states)

        for node_id in sorted(node_servers):
            server = node_servers[node_id]
            if server.busy or not server.queue:
                continue
            first = server.queue[0]
            batch: list[_Operation] = []
            if first.phase == Phase.DECODE:
                while (
                    server.queue
                    and server.queue[0].phase == Phase.DECODE
                    and server.queue[0].stage_index == first.stage_index
                ):
                    batch.append(server.queue.popleft())
            else:
                batch.append(server.queue.popleft())
            server.busy = True
            stage = ordered_stages[first.stage_index]
            if first.phase == Phase.PREFILL:
                state = states[first.request_id]
                service_s = profiler.prefill_stage_time(
                    state.spec,
                    pipeline,
                    stage,
                    n_prefill,
                    n_decode_active,
                )
            else:
                batch_size = len(batch)
                durations: list[float] = []
                for operation in batch:
                    state = states[operation.request_id]
                    context = state.spec.input_tokens + int(operation.token_index or 0)
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
                service_s = max(durations)
            if service_s < 0:
                raise ValueError("profile service time cannot be negative")
            push_event(
                current_time + service_s,
                "server_complete",
                "node",
                node_id,
                tuple(batch),
            )

        for link_id in sorted(link_servers):
            server = link_servers[link_id]
            if server.busy or not server.queue:
                continue
            operation = server.queue.popleft()
            server.busy = True
            state = states[operation.request_id]
            transfer_tokens = (
                state.spec.input_tokens if operation.phase == Phase.PREFILL else 1
            )
            data_bytes = pipeline.model.activation_bytes_per_token * transfer_tokens
            service_s = data_bytes / pipeline.links[link_id].capacity_bytes_per_s
            push_event(
                current_time + service_s,
                "server_complete",
                "link",
                link_id,
                (operation,),
            )

    while arrival_index < len(requests) or states or completion_heap:
        if processed_events >= config.max_events:
            raise RuntimeError("reference event limit exceeded")

        next_arrival = (
            requests[arrival_index].arrival_time_s
            if arrival_index < len(requests)
            else float("inf")
        )
        next_completion = completion_heap[0][0] if completion_heap else float("inf")
        next_time = min(next_arrival, next_completion)
        if next_time == float("inf"):
            break
        if next_time < current_time - config.time_epsilon:
            raise RuntimeError("reference event time moved backwards")
        current_time = next_time
        batch_violations: list[ReferenceViolation] = []

        due: list[tuple[float, int, str, str, str, object]] = []
        while completion_heap and completion_heap[0][0] <= current_time + config.time_epsilon:
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
            if event_kind == "server_complete":
                operations = tuple(payload)  # type: ignore[arg-type]
                for operation in operations:
                    if resource_kind == "node":
                        route_after_stage(operation)
                    else:
                        if operation.route_position + 1 < len(operation.route_link_ids):
                            enqueue_link(
                                operation.request_id,
                                operation.phase,
                                operation.stage_index,
                                operation.token_index,
                                operation.route_link_ids,
                                operation.route_position + 1,
                            )
                        else:
                            enqueue_compute(
                                operation.request_id,
                                operation.phase,
                                operation.stage_index + 1,
                                operation.token_index,
                            )
                continue

            operation = payload  # type: ignore[assignment]
            state = states[operation.request_id]
            if operation.phase == Phase.PREFILL:
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
                state.token_start_s = current_time
                enqueue_compute(state.spec.id, Phase.DECODE, 0, 0)
            else:
                if state.token_start_s is None:
                    raise RuntimeError("decode token completed without start time")
                observed = current_time - state.token_start_s
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
                    del states[state.spec.id]
                else:
                    state.token_start_s = current_time
                    enqueue_compute(
                        state.spec.id,
                        Phase.DECODE,
                        0,
                        state.completed_tokens,
                    )

        arrivals: list[RequestSpec] = []
        while (
            arrival_index < len(requests)
            and requests[arrival_index].arrival_time_s <= current_time + config.time_epsilon
        ):
            arrivals.append(requests[arrival_index])
            arrival_index += 1
        for request in sorted(arrivals, key=lambda item: item.id):
            processed_events += 1
            states[request.id] = _RequestState(request)
            enqueue_compute(request.id, Phase.PREFILL, 0, None)

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


def find_reference_capacity(
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
    """Use the same sampled-frontier discipline as the Conservative search.

    The evaluator is independent; only the workload-intensity search policy is
    intentionally matched so rank comparisons are not confounded by different
    numerical search procedures.
    """

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
            result = evaluate_reference(
                pipeline=pipeline,
                workload=scale_workload(workload, key),
                sla=sla,
                profiler=profiler,
                config=config,
            )
            cache[key] = result
            trials.append(
                ReferenceCapacityTrial(
                    key,
                    result.feasible,
                    result.first_violation.kind if result.first_violation else None,
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
            raise RuntimeError("no unsafe reference intensity found below max_intensity")
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

    bracket_low = low
    bracket_high = high
    probe_intensities: list[float] = []
    for index in range(1, verification_grid_points):
        intensity = bracket_low + (bracket_high - bracket_low) * (
            index / verification_grid_points
        )
        probe_intensities.append(intensity)
        run(intensity)

    probe = bracket_high * 2.0
    while probe <= max_intensity and len(probe_intensities) < verification_grid_points + 4:
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
    high = min(trial.intensity for trial in unsafe_trials if trial.intensity > low)
    safe_run = run(low)
    high_run = run(high)

    while high - low > tolerance * max(high, 1.0):
        mid = (low + high) / 2.0
        mid_run = run(mid)
        if mid_run.feasible:
            low = mid
            safe_run = mid_run
        else:
            high = mid
            high_run = mid_run

    return ReferenceCapacityResult(
        low,
        high,
        tuple(trials),
        safe_run,
        high_run,
        True,
        tuple(probe_intensities),
    )
