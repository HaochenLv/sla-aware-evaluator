from __future__ import annotations

from dataclasses import dataclass

from .domain import EvaluatorConfig, Pipeline, RequestSpec, Stage


@dataclass(frozen=True)
class AnalyticalProfiler:
    """Transparent estimated profile used only for the MVP.

    It is deliberately replaceable by a HELIX or measured-profile adapter.
    The stage-level methods expose service demand for the independent reference
    evaluator; the existing pipeline-level methods remain the Conservative
    Evaluator interface.
    """

    config: EvaluatorConfig

    def _stage_base_time(self, pipeline: Pipeline, stage: Stage, tokens: float) -> float:
        node = pipeline.nodes[stage.node_id]
        flops = pipeline.model.flops_per_token_per_layer * stage.num_layers * tokens
        effective_flops_per_s = (
            node.compute_tflops * 1e12 * self.config.compute_efficiency
        )
        return flops / effective_flops_per_s

    def _base_pipeline_time(self, pipeline: Pipeline, tokens: float) -> float:
        return sum(
            self._stage_base_time(pipeline, stage, tokens)
            for stage in pipeline.stages
        )

    def prefill_stage_time(
        self,
        request: RequestSpec,
        pipeline: Pipeline,
        stage: Stage,
        n_prefill: int,
        n_decode: int,
    ) -> float:
        contention = (
            1.0
            + self.config.prefill_contention * max(n_prefill - 1, 0)
            + self.config.mixed_contention * n_decode
        )
        return self._stage_base_time(pipeline, stage, request.input_tokens) * contention

    def prefill_time(
        self, request: RequestSpec, pipeline: Pipeline, n_prefill: int, n_decode: int
    ) -> float:
        return sum(
            self.prefill_stage_time(request, pipeline, stage, n_prefill, n_decode)
            for stage in pipeline.stages
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
        contention = (
            1.0
            + self.config.decode_contention * max(n_decode - 1, 0)
            + self.config.mixed_contention * n_prefill
        )
        context_factor = 1.0 + self.config.decode_context_penalty * (
            context_tokens / self.config.decode_context_scale
        )
        return self._stage_base_time(pipeline, stage, 1.0) * contention * context_factor

    def decode_time_per_token(
        self,
        request: RequestSpec,
        context_tokens: int,
        pipeline: Pipeline,
        n_prefill: int,
        n_decode: int,
    ) -> float:
        return sum(
            self.decode_stage_time_per_token(
                request,
                context_tokens,
                pipeline,
                stage,
                n_prefill,
                n_decode,
            )
            for stage in pipeline.stages
        )
