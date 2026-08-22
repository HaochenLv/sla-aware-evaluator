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
        result[node_id] = (
            pipeline.model.weight_bytes * layer_fraction
            + node.workspace_bytes
            + node.memory_margin_bytes
        )
    return result


def _account_resources(
    *,
    time_s: float,
    active: dict[str, RequestRuntime],
    pipeline: Pipeline,
    sla: SLA,
    config: EvaluatorConfig,
    profiler: AnalyticalProfiler,
) -> tuple[dict[str, float], dict[str, float], Violation | None]:
    n_prefill, n_decode = _counts(active)
    link_required = {link_id: 0.0 for link_id in pipeline.links}

    for runtime in active.values():
        if runtime.phase == Phase.PREFILL:
            compute_s = runtime.prefill_compute_s
            budget_s = sla.ttft_s
        else:
            context = runtime.resource_context(config.decode_block_size)
            compute_s = profiler.decode_time_per_token(
                runtime.spec, context, pipeline, n_prefill, n_decode
            )
            budget_s = sla.tpot_s
        remaining_s = (
            budget_s - compute_s - sla.queue_overhead_s - sla.fixed_overhead_s
        )
        if remaining_s <= config.time_epsilon:
            return link_required, {}, Violation(
                time_s=time_s,
                kind=ViolationKind.SLA_TIME,
                object_id=runtime.phase.value,
                required=compute_s + sla.queue_overhead_s + sla.fixed_overhead_s,
                capacity=budget_s,
                request_id=runtime.spec.id,
                num_prefill=n_prefill,
                num_decode=n_decode,
            )

        demand = _network_bytes_by_link(runtime, pipeline)
        normalized_cost = 0.0
        for link_id, data_bytes in demand.items():
            capacity = pipeline.links[link_id].capacity_bytes_per_s
            if data_bytes > 0 and capacity <= 0:
                return link_required, {}, Violation(
                    time_s=time_s,
                    kind=ViolationKind.NETWORK,
                    object_id=link_id,
                    required=math.inf,
                    capacity=capacity,
                    request_id=runtime.spec.id,
                    num_prefill=n_prefill,
                    num_decode=n_decode,
                )
            normalized_cost += data_bytes / capacity
        if normalized_cost > 0:
            for link_id, data_bytes in demand.items():
                capacity = pipeline.links[link_id].capacity_bytes_per_s
                weight = (data_bytes / capacity) / normalized_cost
                link_budget_s = weight * remaining_s
                link_required[link_id] += data_bytes / link_budget_s

    for link_id, required in link_required.items():
        capacity = pipeline.links[link_id].capacity_bytes_per_s
        if required > capacity + config.time_epsilon:
            return link_required, {}, Violation(
                time_s=time_s,
                kind=ViolationKind.NETWORK,
                object_id=link_id,
                required=required,
                capacity=capacity,
                request_id=None,
                num_prefill=n_prefill,
                num_decode=n_decode,
            )

    memory = _static_memory_by_node(pipeline)
    layers_by_node = pipeline.layers_by_node()
    for runtime in active.values():
        context = runtime.resource_context(config.decode_block_size)
        for node_id, num_layers in layers_by_node.items():
            memory[node_id] += (
                pipeline.model.kv_bytes_per_token_per_layer
                * context
                * num_layers
            )
    for node_id, used in memory.items():
        capacity = pipeline.nodes[node_id].memory_capacity_bytes
        if used > capacity:
            return link_required, memory, Violation(
                time_s=time_s,
                kind=ViolationKind.MEMORY,
                object_id=node_id,
                required=used,
                capacity=capacity,
                request_id=None,
                num_prefill=n_prefill,
                num_decode=n_decode,
            )
    return link_required, memory, None


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
        return EvaluationResult(True, None, 0.0, 0, 0, 0, {}, {}, ())

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
                context = runtime.resource_context(config.decode_block_size)
                runtime.decode_time_per_token_s = profiler.decode_time_per_token(
                    runtime.spec, context, pipeline, n_prefill, n_decode
                )

        candidates: list[float] = []
        if arrival_index < len(requests):
            candidates.append(requests[arrival_index].arrival_time_s)
        candidates.extend(
            runtime.prefill_finish_s
            for runtime in active.values()
            if runtime.phase == Phase.PREFILL
        )
        for runtime in active.values():
            if runtime.phase != Phase.DECODE:
                continue
            progress = runtime.decode_progress
            completed_blocks = math.floor(
                (progress + config.progress_epsilon) / config.decode_block_size
            )
            next_block = (completed_blocks + 1) * config.decode_block_size
            target = min(float(next_block), float(runtime.spec.output_tokens))
            remaining_tokens = max(target - progress, 0.0)
            candidates.append(time_s + remaining_tokens * runtime.decode_time_per_token_s)
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
        time_s = next_time

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
        for request in arrivals:
            active[request.id] = RequestRuntime(request, Phase.PREFILL)
        if arrivals:
            batch_prefill, batch_decode = _counts(active)
            for request in arrivals:
                runtime = active[request.id]
                duration = profiler.prefill_time(
                    request, pipeline, batch_prefill, batch_decode
                )
                runtime.prefill_compute_s = duration
                runtime.prefill_finish_s = time_s + duration

        events += 1
        n_prefill, n_decode = _counts(active)
        peak_prefill = max(peak_prefill, n_prefill)
        peak_decode = max(peak_decode, n_decode)
        link_required, memory, violation = _account_resources(
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
                    link_required_bytes_per_s=dict(link_required),
                    node_memory_bytes=dict(memory),
                    request_progress={
                        request_id: runtime.decode_progress
                        for request_id, runtime in active.items()
                        if runtime.phase == Phase.DECODE
                    },
                )
            )
        if violation is not None:
            return EvaluationResult(
                False,
                violation,
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
        time_s,
        events,
        peak_prefill,
        peak_decode,
        min_headroom,
        peak_memory,
        tuple(trace),
    )
