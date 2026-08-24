from __future__ import annotations

from typing import Sequence

from .capacity import scale_workload
from .domain import Pipeline, RequestSpec, SLA
from .reference import ReferenceConfig, ReferenceEvaluationResult
from .reference_diagnostics import (
    ReferenceTPOTDiagnostic,
    ReferenceTraceRecorder,
    StageProfilerLike,
    TracingStageProfiler,
    _attribute_token,
    _failing_token_index,
    _with_link_capacity_multiplier,
)
from .reference_round import evaluate_reference_round


def diagnose_reference_round_tpot(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    profiler: StageProfilerLike,
    intensity: float,
    config: ReferenceConfig | None = None,
    network_capacity_multiplier: float = 1_000_000.0,
) -> tuple[ReferenceTPOTDiagnostic, ReferenceEvaluationResult]:
    if intensity <= 0:
        raise ValueError("intensity must be positive")
    config = config or ReferenceConfig()
    scaled = scale_workload(workload, intensity)
    overhead_s = sla.queue_overhead_s + sla.fixed_overhead_s

    tracing = TracingStageProfiler(profiler)
    trace = ReferenceTraceRecorder()
    original = evaluate_reference_round(
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
        violation.observed
        if violation is not None and violation.kind == "tpot"
        else None
    )

    failing_context = None
    output_token_index = None
    gpu_profile_service_s = None
    attribution = None
    profile_batch_sizes = ()
    profile_stage_service = ()

    if request_id is not None and observed_tpot_s is not None and violation is not None:
        output_token_index = _failing_token_index(
            trace.events,
            request_id,
            violation.time_s,
            config.time_epsilon,
        )
        if output_token_index is None:
            raise RuntimeError(
                "could not identify failing Decode token from Reference round trace"
            )
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
    counterfactual = evaluate_reference_round(
        pipeline=counterfactual_pipeline,
        workload=scaled,
        sla=sla,
        profiler=profiler,
        config=config,
        trace_sink=counterfactual_trace,
    )
    counterfactual_violation = counterfactual.first_violation
    counterfactual_attribution = None
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
                "could not identify counterfactual failing Decode token from Reference round trace"
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
        trace_batch_sizes = ()
        stage_queue = ()
        stage_elapsed = ()
        link_queue_by_link = ()
        link_service_by_link = ()
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

    if profile_batch_sizes and trace_batch_sizes != profile_batch_sizes:
        raise RuntimeError(
            "profile and Reference round trace Decode batch sizes disagree"
        )

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
