from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .capacity import scale_workload
from .domain import Phase, Pipeline, RequestSpec, SLA, Stage
from .reference import (
    ReferenceConfig,
    ReferenceEvaluationResult,
    ReferenceTraceEvent,
    evaluate_reference,
)


class StageProfilerLike(Protocol):
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
class DecodeServiceRecord:
    request_id: str
    context_tokens: int
    stage_id: str
    n_prefill: int
    decode_batch_size: int
    service_s: float


class TracingStageProfiler:
    """Non-invasive wrapper that records Decode profile calls."""

    def __init__(self, base: StageProfilerLike):
        self.base = base
        self.decode_records: list[DecodeServiceRecord] = []

    def prefill_stage_time(
        self,
        request: RequestSpec,
        pipeline: Pipeline,
        stage: Stage,
        n_prefill: int,
        n_decode: int,
    ) -> float:
        return self.base.prefill_stage_time(
            request, pipeline, stage, n_prefill, n_decode
        )

    def decode_stage_time_per_token(
        self,
        request: RequestSpec,
        context_tokens: int,
        pipeline: Pipeline,
        stage: Stage,
        n_prefill: int,
        n_decode: int,
    ) -> float:
        service_s = self.base.decode_stage_time_per_token(
            request,
            context_tokens,
            pipeline,
            stage,
            n_prefill,
            n_decode,
        )
        self.decode_records.append(
            DecodeServiceRecord(
                request_id=request.id,
                context_tokens=context_tokens,
                stage_id=stage.id,
                n_prefill=n_prefill,
                decode_batch_size=n_decode,
                service_s=service_s,
            )
        )
        return service_s


class ReferenceTraceRecorder:
    def __init__(self) -> None:
        self.events: list[ReferenceTraceEvent] = []

    def record(self, event: ReferenceTraceEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True)
class _TraceAttribution:
    gpu_queue_wait_s: float
    gpu_service_elapsed_s: float
    link_queue_wait_s: float
    link_service_elapsed_s: float
    configured_overhead_s: float
    fully_attributed_s: float
    unattributed_s: float
    unattributed_fraction: float | None
    decode_batch_sizes: tuple[int, ...]
    stage_gpu_queue_wait_s: tuple[tuple[str, float], ...]
    stage_gpu_service_elapsed_s: tuple[tuple[str, float], ...]
    link_queue_wait_s_by_link: tuple[tuple[str, float], ...]
    link_service_elapsed_s_by_link: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ReferenceTPOTDiagnostic:
    pipeline_id: str
    intensity: float
    request_id: str | None
    violation_kind: str | None
    observed_tpot_s: float | None
    tpot_limit_s: float
    failing_context_tokens: int | None
    failing_output_token_index: int | None
    gpu_profile_service_s: float | None
    gpu_service_elapsed_s: float | None
    gpu_queue_wait_s: float | None
    explicit_link_service_s: float | None
    link_queue_wait_s: float | None
    configured_overhead_s: float
    accounted_service_s: float | None
    fully_attributed_s: float | None
    residual_wait_s: float | None
    residual_wait_fraction: float | None
    unattributed_s: float | None
    unattributed_fraction: float | None
    decode_batch_sizes: tuple[int, ...]
    stage_service_s: tuple[tuple[str, float], ...]
    stage_gpu_queue_wait_s: tuple[tuple[str, float], ...]
    stage_gpu_service_elapsed_s: tuple[tuple[str, float], ...]
    link_queue_wait_s_by_link: tuple[tuple[str, float], ...]
    link_service_elapsed_s_by_link: tuple[tuple[str, float], ...]
    counterfactual_network_multiplier: float
    counterfactual_feasible: bool
    counterfactual_first_violation_kind: str | None
    counterfactual_first_violation_request_id: str | None
    counterfactual_first_violation_observed_s: float | None
    counterfactual_max_tpot_s: float
    counterfactual_gpu_queue_wait_s: float | None
    counterfactual_gpu_service_elapsed_s: float | None
    counterfactual_link_queue_wait_s: float | None
    counterfactual_link_service_elapsed_s: float | None
    counterfactual_unattributed_s: float | None
    counterfactual_unattributed_fraction: float | None
    counterfactual_decode_batch_sizes: tuple[int, ...]


def _with_link_capacity_multiplier(
    pipeline: Pipeline, multiplier: float
) -> Pipeline:
    if multiplier <= 0:
        raise ValueError("network capacity multiplier must be positive")
    links = {
        link_id: replace(
            link,
            capacity_bytes_per_s=link.capacity_bytes_per_s * multiplier,
        )
        for link_id, link in pipeline.links.items()
    }
    return replace(pipeline, links=links)


def _failing_token_index(
    events: Sequence[ReferenceTraceEvent],
    request_id: str,
    violation_time_s: float,
    epsilon: float,
) -> int | None:
    matches = [
        event
        for event in events
        if event.event == "token_complete"
        and event.request_id == request_id
        and event.token_index is not None
        and abs(event.time_s - violation_time_s) <= epsilon
    ]
    if not matches:
        return None
    return max(int(event.token_index) for event in matches)


def _attribute_token(
    *,
    events: Sequence[ReferenceTraceEvent],
    request_id: str,
    token_index: int,
    observed_tpot_s: float,
    configured_overhead_s: float,
    epsilon: float,
) -> _TraceAttribution:
    token_events = [
        event
        for event in events
        if event.request_id == request_id
        and event.phase == Phase.DECODE.value
        and event.token_index == token_index
    ]
    operation_ids = sorted(
        {
            int(event.operation_seq)
            for event in token_events
            if event.operation_seq is not None
            and event.resource_kind in {"node", "link"}
        }
    )

    gpu_queue = 0.0
    gpu_service = 0.0
    link_queue = 0.0
    link_service = 0.0
    batch_sizes: list[int] = []
    stage_queue: dict[str, float] = {}
    stage_service: dict[str, float] = {}
    link_queue_by_id: dict[str, float] = {}
    link_service_by_id: dict[str, float] = {}

    for operation_id in operation_ids:
        op_events = [
            event for event in token_events if event.operation_seq == operation_id
        ]
        enqueue = next(
            (event for event in op_events if event.event == "queue_enqueue"), None
        )
        start = next(
            (event for event in op_events if event.event == "service_start"), None
        )
        complete = next(
            (event for event in op_events if event.event == "service_complete"), None
        )
        if enqueue is None or start is None or complete is None:
            raise RuntimeError(
                "incomplete Reference trace for a queued resource operation"
            )
        wait_s = start.time_s - enqueue.time_s
        service_s = complete.time_s - start.time_s
        if wait_s < -epsilon or service_s < -epsilon:
            raise RuntimeError("Reference trace contains negative queue/service time")
        wait_s = max(wait_s, 0.0)
        service_s = max(service_s, 0.0)

        if start.resource_kind == "node":
            gpu_queue += wait_s
            gpu_service += service_s
            stage_id = start.stage_id or "unknown-stage"
            stage_queue[stage_id] = stage_queue.get(stage_id, 0.0) + wait_s
            stage_service[stage_id] = stage_service.get(stage_id, 0.0) + service_s
            if start.batch_size is not None:
                batch_sizes.append(start.batch_size)
        elif start.resource_kind == "link":
            link_queue += wait_s
            link_service += service_s
            link_id = start.resource_id
            link_queue_by_id[link_id] = link_queue_by_id.get(link_id, 0.0) + wait_s
            link_service_by_id[link_id] = (
                link_service_by_id.get(link_id, 0.0) + service_s
            )

    fully_attributed = (
        gpu_queue
        + gpu_service
        + link_queue
        + link_service
        + configured_overhead_s
    )
    unattributed = observed_tpot_s - fully_attributed
    if abs(unattributed) <= epsilon:
        unattributed = 0.0
    fraction = unattributed / observed_tpot_s if observed_tpot_s > 0 else None
    return _TraceAttribution(
        gpu_queue_wait_s=gpu_queue,
        gpu_service_elapsed_s=gpu_service,
        link_queue_wait_s=link_queue,
        link_service_elapsed_s=link_service,
        configured_overhead_s=configured_overhead_s,
        fully_attributed_s=fully_attributed,
        unattributed_s=unattributed,
        unattributed_fraction=fraction,
        decode_batch_sizes=tuple(batch_sizes),
        stage_gpu_queue_wait_s=tuple(stage_queue.items()),
        stage_gpu_service_elapsed_s=tuple(stage_service.items()),
        link_queue_wait_s_by_link=tuple(link_queue_by_id.items()),
        link_service_elapsed_s_by_link=tuple(link_service_by_id.items()),
    )


def diagnose_reference_tpot(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    profiler: StageProfilerLike,
    intensity: float,
    config: ReferenceConfig | None = None,
    network_capacity_multiplier: float = 1_000_000.0,
) -> tuple[ReferenceTPOTDiagnostic, ReferenceEvaluationResult]:
    """Attribute the first Reference TPOT failure without changing scheduling.

    The Reference evaluator now exposes observational queue/service trace hooks.
    For the failing Decode token we separate exact simulated time into GPU queue,
    GPU service, link queue, link service, configured overhead, and any remaining
    unattributed time. The same attribution is repeated with near-infinite links.

    This is a diagnostic measurement, not a scheduler change. In particular,
    large GPU queue wait can motivate a scheduler review but does not by itself
    establish what a production vLLM/HELIX scheduler would do.
    """

    if intensity <= 0:
        raise ValueError("intensity must be positive")
    config = config or ReferenceConfig()
    scaled = scale_workload(workload, intensity)
    overhead_s = sla.queue_overhead_s + sla.fixed_overhead_s

    tracing = TracingStageProfiler(profiler)
    trace = ReferenceTraceRecorder()
    original = evaluate_reference(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        profiler=tracing,
        config=config,
        trace_sink=trace,
    )

    violation = original.first_violation
    request_id = violation.request_id if violation is not None else None
    violation_kind = violation.kind if violation is not None else None
    observed_tpot_s = (
        violation.observed if violation is not None and violation.kind == "tpot" else None
    )

    failing_context: int | None = None
    output_token_index: int | None = None
    gpu_profile_service_s: float | None = None
    attribution: _TraceAttribution | None = None
    profile_batch_sizes: tuple[int, ...] = ()
    profile_stage_service: tuple[tuple[str, float], ...] = ()

    if request_id is not None and observed_tpot_s is not None and violation is not None:
        output_token_index = _failing_token_index(
            trace.events,
            request_id,
            violation.time_s,
            config.time_epsilon,
        )
        if output_token_index is None:
            raise RuntimeError("could not identify failing Decode token from trace")
        request = next(request for request in scaled if request.id == request_id)
        failing_context = request.input_tokens + output_token_index
        token_records = [
            record
            for record in tracing.decode_records
            if record.request_id == request_id
            and record.context_tokens == failing_context
        ]
        gpu_profile_service_s = sum(record.service_s for record in token_records)
        profile_batch_sizes = tuple(
            record.decode_batch_size for record in token_records
        )
        profile_stage_service = tuple(
            (record.stage_id, record.service_s) for record in token_records
        )
        attribution = _attribute_token(
            events=trace.events,
            request_id=request_id,
            token_index=output_token_index,
            observed_tpot_s=observed_tpot_s,
            configured_overhead_s=overhead_s,
            epsilon=config.time_epsilon,
        )

    counterfactual_pipeline = _with_link_capacity_multiplier(
        pipeline, network_capacity_multiplier
    )
    counterfactual_trace = ReferenceTraceRecorder()
    counterfactual = evaluate_reference(
        pipeline=counterfactual_pipeline,
        workload=scaled,
        sla=sla,
        profiler=profiler,
        config=config,
        trace_sink=counterfactual_trace,
    )
    counterfactual_violation = counterfactual.first_violation
    counterfactual_attribution: _TraceAttribution | None = None
    if (
        counterfactual_violation is not None
        and counterfactual_violation.kind == "tpot"
        and counterfactual_violation.request_id is not None
    ):
        counterfactual_token_index = _failing_token_index(
            counterfactual_trace.events,
            counterfactual_violation.request_id,
            counterfactual_violation.time_s,
            config.time_epsilon,
        )
        if counterfactual_token_index is None:
            raise RuntimeError(
                "could not identify counterfactual failing Decode token from trace"
            )
        counterfactual_attribution = _attribute_token(
            events=counterfactual_trace.events,
            request_id=counterfactual_violation.request_id,
            token_index=counterfactual_token_index,
            observed_tpot_s=counterfactual_violation.observed,
            configured_overhead_s=overhead_s,
            epsilon=config.time_epsilon,
        )

    if attribution is None:
        gpu_service_elapsed_s = None
        gpu_queue_wait_s = None
        link_service_s = None
        link_queue_wait_s = None
        accounted_s = None
        fully_attributed_s = None
        residual_s = None
        residual_fraction = None
        unattributed_s = None
        unattributed_fraction = None
        trace_batch_sizes: tuple[int, ...] = ()
        stage_queue: tuple[tuple[str, float], ...] = ()
        stage_elapsed: tuple[tuple[str, float], ...] = ()
        link_queue_by_link: tuple[tuple[str, float], ...] = ()
        link_service_by_link: tuple[tuple[str, float], ...] = ()
    else:
        gpu_service_elapsed_s = attribution.gpu_service_elapsed_s
        gpu_queue_wait_s = attribution.gpu_queue_wait_s
        link_service_s = attribution.link_service_elapsed_s
        link_queue_wait_s = attribution.link_queue_wait_s
        accounted_s = (
            attribution.gpu_service_elapsed_s
            + attribution.link_service_elapsed_s
            + overhead_s
        )
        fully_attributed_s = attribution.fully_attributed_s
        residual_s = attribution.gpu_queue_wait_s + attribution.link_queue_wait_s
        residual_fraction = (
            residual_s / observed_tpot_s
            if observed_tpot_s is not None and observed_tpot_s > 0
            else None
        )
        unattributed_s = attribution.unattributed_s
        unattributed_fraction = attribution.unattributed_fraction
        trace_batch_sizes = attribution.decode_batch_sizes
        stage_queue = attribution.stage_gpu_queue_wait_s
        stage_elapsed = attribution.stage_gpu_service_elapsed_s
        link_queue_by_link = attribution.link_queue_wait_s_by_link
        link_service_by_link = attribution.link_service_elapsed_s_by_link

    # Profile-call and actual-service batch sizes should agree. Keep the trace
    # version as the primary result because it is tied to queue/service intervals.
    if profile_batch_sizes and trace_batch_sizes != profile_batch_sizes:
        raise RuntimeError("profile and Reference trace Decode batch sizes disagree")

    diagnostic = ReferenceTPOTDiagnostic(
        pipeline_id=pipeline.id,
        intensity=intensity,
        request_id=request_id,
        violation_kind=violation_kind,
        observed_tpot_s=observed_tpot_s,
        tpot_limit_s=sla.tpot_s,
        failing_context_tokens=failing_context,
        failing_output_token_index=output_token_index,
        gpu_profile_service_s=gpu_profile_service_s,
        gpu_service_elapsed_s=gpu_service_elapsed_s,
        gpu_queue_wait_s=gpu_queue_wait_s,
        explicit_link_service_s=link_service_s,
        link_queue_wait_s=link_queue_wait_s,
        configured_overhead_s=overhead_s,
        accounted_service_s=accounted_s,
        fully_attributed_s=fully_attributed_s,
        residual_wait_s=residual_s,
        residual_wait_fraction=residual_fraction,
        unattributed_s=unattributed_s,
        unattributed_fraction=unattributed_fraction,
        decode_batch_sizes=trace_batch_sizes,
        stage_service_s=profile_stage_service,
        stage_gpu_queue_wait_s=stage_queue,
        stage_gpu_service_elapsed_s=stage_elapsed,
        link_queue_wait_s_by_link=link_queue_by_link,
        link_service_elapsed_s_by_link=link_service_by_link,
        counterfactual_network_multiplier=network_capacity_multiplier,
        counterfactual_feasible=counterfactual.feasible,
        counterfactual_first_violation_kind=(
            counterfactual_violation.kind if counterfactual_violation else None
        ),
        counterfactual_first_violation_request_id=(
            counterfactual_violation.request_id if counterfactual_violation else None
        ),
        counterfactual_first_violation_observed_s=(
            counterfactual_violation.observed if counterfactual_violation else None
        ),
        counterfactual_max_tpot_s=max(
            counterfactual.max_tpot_s_by_request.values(), default=0.0
        ),
        counterfactual_gpu_queue_wait_s=(
            counterfactual_attribution.gpu_queue_wait_s
            if counterfactual_attribution is not None
            else None
        ),
        counterfactual_gpu_service_elapsed_s=(
            counterfactual_attribution.gpu_service_elapsed_s
            if counterfactual_attribution is not None
            else None
        ),
        counterfactual_link_queue_wait_s=(
            counterfactual_attribution.link_queue_wait_s
            if counterfactual_attribution is not None
            else None
        ),
        counterfactual_link_service_elapsed_s=(
            counterfactual_attribution.link_service_elapsed_s
            if counterfactual_attribution is not None
            else None
        ),
        counterfactual_unattributed_s=(
            counterfactual_attribution.unattributed_s
            if counterfactual_attribution is not None
            else None
        ),
        counterfactual_unattributed_fraction=(
            counterfactual_attribution.unattributed_fraction
            if counterfactual_attribution is not None
            else None
        ),
        counterfactual_decode_batch_sizes=(
            counterfactual_attribution.decode_batch_sizes
            if counterfactual_attribution is not None
            else ()
        ),
    )
    return diagnostic, original
