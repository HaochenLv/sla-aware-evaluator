from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .capacity import scale_workload
from .domain import Pipeline, RequestSpec, SLA, Stage
from .reference import ReferenceConfig, ReferenceEvaluationResult, evaluate_reference


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
    """Non-invasive wrapper that records Decode profile calls.

    The wrapped profiler is otherwise unchanged. This lets the diagnostic path
    inspect the GPU service assigned to the token that first violates TPOT
    without modifying Reference v0's queueing/scheduling semantics.
    """

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
    explicit_link_service_s: float | None
    configured_overhead_s: float
    accounted_service_s: float | None
    residual_wait_s: float | None
    residual_wait_fraction: float | None
    decode_batch_sizes: tuple[int, ...]
    stage_service_s: tuple[tuple[str, float], ...]
    counterfactual_network_multiplier: float
    counterfactual_feasible: bool
    counterfactual_first_violation_kind: str | None
    counterfactual_first_violation_request_id: str | None
    counterfactual_first_violation_observed_s: float | None
    counterfactual_max_tpot_s: float


def _decode_link_service_s(pipeline: Pipeline) -> float:
    data_bytes = pipeline.model.activation_bytes_per_token
    total = 0.0
    for boundary in pipeline.boundaries:
        for link_id in boundary.route_link_ids:
            total += data_bytes / pipeline.links[link_id].capacity_bytes_per_s
    return total


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
    """Diagnose the first Reference TPOT failure without changing its scheduler.

    The original run is replayed with a tracing profiler. For the request/token
    that first violates TPOT we decompose the observed interval into:

        profiled GPU stage service
      + deterministic explicit D/B link service
      + configured fixed/queue overhead
      + residual waiting/synchronization time.

    The residual is intentionally *not* labeled GPU wait: it can include GPU
    queueing, link queueing, and batching/synchronization effects. A second run
    multiplies all link capacities by a large factor. If the TPOT violation and
    latency remain, that is evidence that network transfer/queueing is not the
    dominant cause under Reference v0.
    """

    if intensity <= 0:
        raise ValueError("intensity must be positive")
    config = config or ReferenceConfig()
    scaled = scale_workload(workload, intensity)

    tracing = TracingStageProfiler(profiler)
    original = evaluate_reference(
        pipeline=pipeline,
        workload=scaled,
        sla=sla,
        profiler=tracing,
        config=config,
    )

    violation = original.first_violation
    request_id = violation.request_id if violation is not None else None
    violation_kind = violation.kind if violation is not None else None
    observed_tpot_s = (
        violation.observed if violation is not None and violation.kind == "tpot" else None
    )

    failing_context: int | None = None
    output_token_index: int | None = None
    gpu_service_s: float | None = None
    link_service_s: float | None = None
    accounted_s: float | None = None
    residual_s: float | None = None
    residual_fraction: float | None = None
    batch_sizes: tuple[int, ...] = ()
    stage_service: tuple[tuple[str, float], ...] = ()

    if request_id is not None and observed_tpot_s is not None:
        request = next(request for request in scaled if request.id == request_id)
        request_records = [
            record for record in tracing.decode_records if record.request_id == request_id
        ]
        if request_records:
            failing_context = max(record.context_tokens for record in request_records)
            token_records = [
                record
                for record in request_records
                if record.context_tokens == failing_context
            ]
            output_token_index = failing_context - request.input_tokens
            gpu_service_s = sum(record.service_s for record in token_records)
            link_service_s = _decode_link_service_s(pipeline)
            overhead_s = sla.queue_overhead_s + sla.fixed_overhead_s
            accounted_s = gpu_service_s + link_service_s + overhead_s
            residual_s = observed_tpot_s - accounted_s
            residual_fraction = (
                residual_s / observed_tpot_s if observed_tpot_s > 0 else None
            )
            batch_sizes = tuple(record.decode_batch_size for record in token_records)
            stage_service = tuple(
                (record.stage_id, record.service_s) for record in token_records
            )

    counterfactual_pipeline = _with_link_capacity_multiplier(
        pipeline, network_capacity_multiplier
    )
    counterfactual = evaluate_reference(
        pipeline=counterfactual_pipeline,
        workload=scaled,
        sla=sla,
        profiler=profiler,
        config=config,
    )
    counterfactual_violation = counterfactual.first_violation

    diagnostic = ReferenceTPOTDiagnostic(
        pipeline_id=pipeline.id,
        intensity=intensity,
        request_id=request_id,
        violation_kind=violation_kind,
        observed_tpot_s=observed_tpot_s,
        tpot_limit_s=sla.tpot_s,
        failing_context_tokens=failing_context,
        failing_output_token_index=output_token_index,
        gpu_profile_service_s=gpu_service_s,
        explicit_link_service_s=link_service_s,
        configured_overhead_s=sla.queue_overhead_s + sla.fixed_overhead_s,
        accounted_service_s=accounted_s,
        residual_wait_s=residual_s,
        residual_wait_fraction=residual_fraction,
        decode_batch_sizes=batch_sizes,
        stage_service_s=stage_service,
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
    )
    return diagnostic, original
