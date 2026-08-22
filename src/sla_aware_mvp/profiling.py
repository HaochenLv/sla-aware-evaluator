from __future__ import annotations

from dataclasses import dataclass

from .domain import EvaluatorConfig, Pipeline, RequestSpec


@dataclass(frozen=True)
class AnalyticalProfiler:
    """Transparent estimated profile used only for the MVP.

    It is deliberately replaceable by a HELIX or measured-profile adapter.
    """

    config: EvaluatorConfig

    def _base_pipeline_time(self, pipeline: Pipeline, tokens: float) -> float:
        total = 0.0
        for stage in pipeline.stages:
            node = pipeline.nodes[stage.node_id]
            flops = (
                pipeline.model.flops_per_token_per_layer
                * stage.num_layers
                * tokens
            )
            effective_flops_per_s = (
                node.compute_tflops * 1e12 * self.config.compute_efficiency
            )
            total += flops / effective_flops_per_s
        return total

    def prefill_time(
        self, request: RequestSpec, pipeline: Pipeline, n_prefill: int, n_decode: int
    ) -> float:
        contention = (
            1.0
            + self.config.prefill_contention * max(n_prefill - 1, 0)
            + self.config.mixed_contention * n_decode
        )
        return self._base_pipeline_time(pipeline, request.input_tokens) * contention

    def decode_time_per_token(
        self,
        request: RequestSpec,
        context_tokens: int,
        pipeline: Pipeline,
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
        return self._base_pipeline_time(pipeline, 1.0) * contention * context_factor

