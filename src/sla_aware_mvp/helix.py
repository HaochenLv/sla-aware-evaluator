from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .domain import Pipeline, RequestSpec, Stage


@dataclass(frozen=True)
class ProfileProvenance:
    source: str
    source_model: str
    source_gpu: str
    quality_label: str
    measured: bool
    observed_dimensions: frozenset[str]
    commit: str


@dataclass(frozen=True)
class HelixLayerProfile:
    points_ms: tuple[tuple[int, float], ...]
    x_kind: str

    @classmethod
    def from_csv(cls, path: Path, *, x_kind: str) -> "HelixLayerProfile":
        points: list[tuple[int, float]] = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 2:
                    raise ValueError(f"invalid HELIX profile row in {path}: {row}")
                points.append((int(row[0]), float(row[1])))
        if len(points) < 2:
            raise ValueError("HELIX profile requires at least two points")
        if points != sorted(points) or len({x for x, _ in points}) != len(points):
            raise ValueError("HELIX profile x values must be unique and increasing")
        if any(x < 0 or y < 0 for x, y in points):
            raise ValueError("HELIX profile points cannot be negative")
        return cls(tuple(points), x_kind)

    @property
    def min_x(self) -> int:
        return self.points_ms[0][0]

    @property
    def max_x(self) -> int:
        return self.points_ms[-1][0]

    def lookup_seconds_per_layer(self, x: int) -> float:
        """Reproduce HELIX's piecewise-linear lookup and ms-to-s conversion."""
        if x < self.min_x or x > self.max_x:
            raise ValueError(
                f"{self.x_kind}={x} is outside HELIX profile range "
                f"[{self.min_x}, {self.max_x}]"
            )
        for index, (right_x, right_ms) in enumerate(self.points_ms):
            if x == right_x:
                return right_ms * 0.001
            if x < right_x:
                left_x, left_ms = self.points_ms[index - 1]
                interpolated_ms = left_ms + (right_ms - left_ms) * (
                    (x - left_x) / (right_x - left_x)
                )
                return interpolated_ms * 0.001
        raise AssertionError("unreachable profile lookup")


@dataclass(frozen=True)
class HelixA100Llama2Profiler:
    """HELIX public measured proxy with its original observed dimensions.

    Prompt uses request input tokens. Decode uses active decode-token batch size.
    HELIX does not observe context length or mixed prefill/decode interference;
    those arguments are intentionally ignored rather than fabricated.

    Stage-level methods expose the same measured per-layer service demand for the
    independent reference evaluator. Pipeline-level methods are exact sums of the
    stage-level values, so adding this interface does not change Conservative
    Evaluator semantics.
    """

    prompt: HelixLayerProfile
    decode: HelixLayerProfile
    provenance: ProfileProvenance

    @classmethod
    def from_artifact(cls, artifact_root: Path, *, commit: str) -> "HelixA100Llama2Profiler":
        base = artifact_root / "simulator/model_manager/llama2_70b/a100"
        return cls(
            prompt=HelixLayerProfile.from_csv(
                base / "prompt_bs2time.csv", x_kind="prompt_tokens"
            ),
            decode=HelixLayerProfile.from_csv(
                base / "decode_bs2time.csv", x_kind="decode_active_tokens"
            ),
            provenance=ProfileProvenance(
                source="HELIX official artifact",
                source_model="LLaMA-2-70B",
                source_gpu="A100-40GB",
                quality_label="M",
                measured=True,
                observed_dimensions=frozenset(
                    {"phase", "prompt_tokens", "decode_active_tokens", "per_layer"}
                ),
                commit=commit,
            ),
        )

    @staticmethod
    def _validate_pipeline(pipeline: Pipeline) -> None:
        if pipeline.model.name != "LLaMA-2-70B":
            raise ValueError("HELIX LLaMA-2-70B profile cannot be relabeled as another model")
        for stage in pipeline.stages:
            node = pipeline.nodes[stage.node_id]
            if node.hardware_type != "A100-40GB":
                raise ValueError("HELIX A100 profile requires A100-40GB stages")
            if stage.num_layers > 12:
                raise ValueError("HELIX metadata allows at most 12 LLaMA-2 layers per A100")

    def prefill_stage_time(
        self,
        request: RequestSpec,
        pipeline: Pipeline,
        stage: Stage,
        n_prefill: int,
        n_decode: int,
    ) -> float:
        self._validate_pipeline(pipeline)
        per_layer = self.prompt.lookup_seconds_per_layer(request.input_tokens)
        return per_layer * stage.num_layers

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
        self._validate_pipeline(pipeline)
        per_layer = self.decode.lookup_seconds_per_layer(max(n_decode, 1))
        return per_layer * stage.num_layers

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
