from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .domain import (
    EvaluatorConfig,
    Phase,
    Pipeline,
    RequestRuntime,
    RequestSpec,
    SLA,
    StateSnapshot,
    Violation,
    ViolationKind,
    sorted_workload,
)
from .profiling import AnalyticalProfiler


@dataclass(frozen=True)
class EvaluationResult:
    feasible: bool
    first_violation: Violation | None
    first_violations: tuple[Violation, ...]
    final_time_s: float
    processed_events: int
    peak_prefill: int
    peak_decode: int
    min_link_headroom_bytes_per_s: dict[str, float]
    peak_memory_bytes: dict[str, float]
    trace: tuple[StateSnapshot, ...]


def _counts(active: dict[str, RequestRuntime]) -> tuple[int, int]:
    n_prefill = sum(runtime.phase == Phase.PREFILL for runtime in active.values())
    n_decode = sum(runtime.phase == Phase.DECODE for runtime in active.values())
    return n_prefill, n_decode


def _network_bytes_by_link(
    runtime: RequestRuntime, pipeline: Pipeline
) -> dict[str, float]:
    if runtime.phase == Phase.PREFILL:
        transfer_tokens = runtime.spec.input_tokens
    else:
        transfer_tokens = 1
    bytes_per_boundary = pipeline.model.activation_bytes_per_token * transfer_tokens
    result: dict[str, float] = {}
    for boundary in pipeline.boundaries:
        for link_id in boundary.route_link_ids:
            result[link_id] = result.get(link_id, 0.0) + bytes_per_boundary
    return result


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


def _remaining_network_budget_s(compute_s: float, budget_s: float, sla: SLA) -> float:
    return budget_s - compute_s - sla.queue_overhead_s - sla.fixed_overhead_s


def _progress_service_time_s(
    *,
    runtime: RequestRuntime,
    pipeline: Pipeline,
    compute_s: float,
    budget_s: float,
    sla: SLA,
    config: EvaluatorConfig,
) -> float:
    """Return the wall-clock service interval used by the state machine.

    Previous MVP code advanced requests with compute time only while separately
    reserving network service. That allowed a request to release its commitment
    before the service represented by the commitment could have completed.

    Under the conservative policy, a phase/token with cross-node traffic retains
    its commitment for the full SLA service window. This is intentionally a
    lower-throughput reservation semantics, not a high-fidelity runtime model.
    """
    base = compute_s + sla.queue_overhead_s + sla.fixed_overhead_s
    remaining = _remaining_network_budget_s(compute_s, budget_s, sla)
    if remaining <= config.time_epsilon:
        return max(base, config.time_epsilon)
    demand = _network_bytes_by_link(runtime, pipeline)
    if config.conservative_network_lifetime and any(value > 0 for value in demand.values()):
        return max(budget_s, config.time_epsilon)
    return max(base, config.time_epsilon)


def _account_resources(
    *,
    time_s: float,
    active: dict[str, RequestRuntime],
    pipeline: Pipeline,
    sla: SLA,
    config: EvaluatorConfig,
    profiler: AnalyticalProfiler,
) -> tuple[dict[str, float], dict[str, float], tuple[Violation, ...]]:
    n_prefill, n_decode = _counts(active)
    link_required = {link_id: 0.0 for link_id in pipeline.links}
    violations: list[Violation] = []

    for runtime in active.values():
        if runtime.phase == Phase.PREFILL:
            compute_s = runtime.prefill_compute_s
            budget_s = sla.ttft_s
        else:
            context = runtime.resource_context(
                config.decode_block_size, config.progress_epsilon
            )
            compute_s = profiler.decode_time_per_token(
                runtime.spec, context, pipeline, n_prefill, n_decode
            )
            budget_s = sla.tpot_s
        remaining_s = _remaining_network_budget_s(compute_s, budget_s, sla)
        if remaining_s <= config.time_epsilon:
            violations.append(
                Violation(
                    time_s=time_s,
                    kind=ViolationKind.SLA_TIME,
                    object_id=runtime.phase.value,
                    required=compute_s + sla.queue_overhead_s + sla.fixed_overhead_s,
                    capacity=budget_s,
                    request_id=runtime.spec.id,
                    num_prefill=n_prefill,
                    num_decode=n_decode,
                )
            )
            # A non-positive network budget has no meaningful reservation rate.
            # Continue so simultaneous memory and other-request violations are kept.
            continue

        demand = _network_bytes_by_link(runtime, pipeline)
        normalized_cost = 0.0
        bad_links: set[str] = set()
        for link_id, data_bytes in demand.items():
            capacity = pipeline.links[link_id].capacity_bytes_per_s
            if data_bytes > 0 and capacity <= 0:
                bad_links.add(link_id)
                violations.append(
                    Violation(
                        time_s=time_s,
                        kind=ViolationKind.NETWORK,
                        object_id=link_id,
                        required=math.inf,
                        capacity=capacity,
                        request_id=runtime.spec.id,
                        num_prefill=n_prefill,
                        num_decode=n_decode,
                    )
                )
            elif data_bytes > 0:
                normalized_cost += data_bytes / capacity
        if bad_links or normalized_cost <= 0:
            continue
        for link_id, data_bytes in demand.items():
            if data_bytes <= 0:
                continue
            capacity = pipeline.links[link_id].capacity_bytes_per_s
            weight = (data_bytes / capacity) / normalized_cost
            link_budget_s = weight * remaining_s
            link_required[link_id] += data_bytes / link_budget_s

    for link_id, required in link_required.items():
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        if required > capacity + config.time_epsilon:
            violations.append(
                Violation(
                    time_s=time_s,
                    kind=ViolationKind.NETWORK,
                    object_id=link_id,
                    required=required,
                    capacity=capacity,
                    request_id=None,
                    num_prefill=n_prefill,
                    num_decode=n_decode,
                )
            )

    memory = _static_memory_by_node(pipeline)
    layers_by_node = pipeline.layers_by_node()
    for runtime in active.values():
        context = runtime.resource_context(
            config.decode_block_size, config.progress_epsilon
        )
        for node_id, num_layers in layers_by_node.items():
            memory[node_id] += (
                pipeline.model.kv_bytes_per_token_per_layer * context * num_layers
            )
    for node_id, used in memory.items():
        capacity = pipeline.nodes[node_id].memory_capacity_bytes
        if used > capacity:
            violations.append(
                Violation(
                    time_s=time_s,
                    kind=ViolationKind.MEMORY,
                    object_id=node_id,
                    required=used,
                    capacity=capacity,
                    request_id=None,
                    num_prefill=n_prefill,
                    num_decode=n_decode,
                )
            )

    kind_order = {
        ViolationKind.SLA_TIME: 0,
        ViolationKind.NETWORK: 1,
        ViolationKind.MEMORY: 2,
    }
    violations.sort(
        key=lambda item: (
            kind_order[item.kind],
            item.object_id,
            item.request_id or "",
        )
    )
    return link_required, memory, tuple(violations)


def evaluate(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    config: EvaluatorConfig | None = None,
    profiler: AnalyticalProfiler | None = None,
) -> EvaluationResult:
    config = config or EvaluatorConfig()
    profiler = profiler or AnalyticalProfiler(config)
    pipeline.validate()
    requests = sorted_workload(workload)
    if not requests:
        return EvaluationResult(True, None, (), 0.0, 0, 0, 0, {}, {}, ())

    time_s = requests[0].arrival_time_s
    arrival_index = 0
    active: dict[str, RequestRuntime] = {}
    trace: list[StateSnapshot] = []
    events = 0
    peak_prefill = 0
    peak_decode = 0
    min_headroom = {
        link_id: link.capacity_bytes_per_s for link_id, link in pipeline.links.items()
    }
    peak_memory = {node_id: 0.0 for node_id in pipeline.nodes}

    while arrival_index < len(requests) or active:
        if events >= config.max_events:
            raise RuntimeError("event limit exceeded")

        n_prefill, n_decode = _counts(active)
        for runtime in active.values():
            if runtime.phase == Phase.DECODE:
                context = runtime.resource_context(
                    config.decode_block_size, config.progress_epsilon
                )
                compute_s = profiler.decode_time_per_token(
                    runtime.spec, context, pipeline, n_prefill, n_decode
                )
                runtime.decode_time_per_token_s = _progress_service_time_s(
                    runtime=runtime,
                    pipeline=pipeline,
                    compute_s=compute_s,
                    budget_s=sla.tpot_s,
                    sla=sla,
                    config=config,
                )

        candidates: list[float] = []
        if arrival_index < len(requests):
            candidates.append(requests[arrival_index].arrival_time_s)
        candidates.extend(
            runtime.prefill_finish_s
            for runtime in active.values()
            if runtime.phase == Phase.PREFILL
        )

        decode_event_times: dict[str, tuple[float, float]] = {}
        for request_id, runtime in active.items():
            if runtime.phase != Phase.DECODE:
                continue
            progress = runtime.decode_progress
            completed_blocks = runtime.block_index(
                config.decode_block_size, config.progress_epsilon
            )
            next_block = (completed_blocks + 1) * config.decode_block_size
            target = min(float(next_block), float(runtime.spec.output_tokens))
            remaining_tokens = max(target - progress, 0.0)
            event_time = time_s + remaining_tokens * runtime.decode_time_per_token_s
            decode_event_times[request_id] = (event_time, target)
            candidates.append(event_time)

        next_time = min(candidates)
        if next_time < time_s - config.time_epsilon:
            raise RuntimeError("event time moved backwards")

        elapsed = max(next_time - time_s, 0.0)
        for runtime in active.values():
            if runtime.phase == Phase.DECODE:
                runtime.decode_progress += elapsed / runtime.decode_time_per_token_s
                runtime.decode_progress = min(
                    runtime.decode_progress, float(runtime.spec.output_tokens)
                )
                nearest_boundary = (
                    round(runtime.decode_progress / config.decode_block_size)
                    * config.decode_block_size
                )
                if abs(runtime.decode_progress - nearest_boundary) <= config.progress_epsilon:
                    runtime.decode_progress = min(
                        float(nearest_boundary), float(runtime.spec.output_tokens)
                    )
        time_s = next_time

        event_types: set[str] = set()
        due_decode = {
            request_id: target
            for request_id, (event_time, target) in decode_event_times.items()
            if abs(event_time - time_s) <= config.time_epsilon
        }
        if any(
            target >= active[request_id].spec.output_tokens - config.progress_epsilon
            for request_id, target in due_decode.items()
            if request_id in active
        ):
            event_types.add("Finish")
        if any(
            target < active[request_id].spec.output_tokens - config.progress_epsilon
            for request_id, target in due_decode.items()
            if request_id in active
        ):
            event_types.add("DecodeBlockUpdate")

        for request_id in list(active):
            runtime = active[request_id]
            if (
                runtime.phase == Phase.DECODE
                and runtime.decode_progress
                >= runtime.spec.output_tokens - config.progress_epsilon
            ):
                del active[request_id]

        transitioning = [
            runtime
            for runtime in active.values()
            if runtime.phase == Phase.PREFILL
            and runtime.prefill_finish_s <= time_s + config.time_epsilon
        ]
        if transitioning:
            event_types.add("PrefillToDecode")
        for runtime in transitioning:
            runtime.phase = Phase.DECODE
            runtime.decode_progress = 0.0

        arrivals: list[RequestSpec] = []
        while (
            arrival_index < len(requests)
            and requests[arrival_index].arrival_time_s <= time_s + config.time_epsilon
        ):
            arrivals.append(requests[arrival_index])
            arrival_index += 1
        if arrivals:
            event_types.add("Arrival")
        for request in arrivals:
            active[request.id] = RequestRuntime(request, Phase.PREFILL)
        if arrivals:
            batch_prefill, batch_decode = _counts(active)
            for request in arrivals:
                runtime = active[request.id]
                compute_s = profiler.prefill_time(
                    request, pipeline, batch_prefill, batch_decode
                )
                runtime.prefill_compute_s = compute_s
                duration = _progress_service_time_s(
                    runtime=runtime,
                    pipeline=pipeline,
                    compute_s=compute_s,
                    budget_s=sla.ttft_s,
                    sla=sla,
                    config=config,
                )
                runtime.prefill_finish_s = time_s + duration

        events += 1
        n_prefill, n_decode = _counts(active)
        peak_prefill = max(peak_prefill, n_prefill)
        peak_decode = max(peak_decode, n_decode)
        link_required, memory, violations = _account_resources(
            time_s=time_s,
            active=active,
            pipeline=pipeline,
            sla=sla,
            config=config,
            profiler=profiler,
        )
        for link_id, required in link_required.items():
            headroom = pipeline.links[link_id].capacity_bytes_per_s - required
            min_headroom[link_id] = min(min_headroom[link_id], headroom)
        for node_id, used in memory.items():
            peak_memory[node_id] = max(peak_memory[node_id], used)
        if config.record_trace:
            trace.append(
                StateSnapshot(
                    time_s=time_s,
                    num_prefill=n_prefill,
                    num_decode=n_decode,
                    event_types=tuple(sorted(event_types)),
                    link_required_bytes_per_s=dict(link_required),
                    node_memory_bytes=dict(memory),
                    request_progress={
                        request_id: runtime.decode_progress
                        for request_id, runtime in active.items()
                        if runtime.phase == Phase.DECODE
                    },
                    request_phase={
                        request_id: runtime.phase.value
                        for request_id, runtime in active.items()
                    },
                    request_context={
                        request_id: runtime.resource_context(
                            config.decode_block_size, config.progress_epsilon
                        )
                        for request_id, runtime in active.items()
                    },
                )
            )
        if violations:
            return EvaluationResult(
                False,
                violations[0],
                violations,
                time_s,
                events,
                peak_prefill,
                peak_decode,
                min_headroom,
                peak_memory,
                tuple(trace),
            )

    return EvaluationResult(
        True,
        None,
        (),
        time_s,
        events,
        peak_prefill,
        peak_decode,
        min_headroom,
        peak_memory,
        tuple(trace),
    )
