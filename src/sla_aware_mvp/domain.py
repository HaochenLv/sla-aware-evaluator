from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


class ViolationKind(str, Enum):
    SLA_TIME = "sla_time"
    NETWORK = "network"
    MEMORY = "memory"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    total_params: float
    weight_bytes_per_param: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    kv_element_bytes: int
    flops_per_token_per_layer: float
    activation_element_bytes: int | None = None

    @property
    def weight_bytes(self) -> float:
        return self.total_params * self.weight_bytes_per_param

    @property
    def activation_bytes_per_token(self) -> int:
        element_bytes = (
            self.activation_element_bytes
            if self.activation_element_bytes is not None
            else self.kv_element_bytes
        )
        return self.hidden_size * element_bytes

    @property
    def kv_bytes_per_token_per_layer(self) -> float:
        head_dim = self.hidden_size / self.num_attention_heads
        return 2 * self.num_kv_heads * head_dim * self.kv_element_bytes


@dataclass(frozen=True)
class GPUNode:
    id: str
    compute_tflops: float
    memory_capacity_bytes: int
    workspace_bytes: int = 0
    memory_margin_bytes: int = 0
    hardware_type: str = "generic"


@dataclass(frozen=True)
class Link:
    id: str
    source: str
    target: str
    capacity_bytes_per_s: float


@dataclass(frozen=True)
class Stage:
    id: str
    layer_start: int
    layer_end: int
    node_id: str

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


@dataclass(frozen=True)
class Boundary:
    upstream_stage_id: str
    downstream_stage_id: str
    route_link_ids: tuple[str, ...]


@dataclass(frozen=True)
class Pipeline:
    id: str
    model: ModelSpec
    nodes: Mapping[str, GPUNode]
    links: Mapping[str, Link]
    stages: tuple[Stage, ...]
    boundaries: tuple[Boundary, ...]

    def validate(self) -> None:
        if not self.stages:
            raise ValueError("pipeline must contain at least one stage")
        ordered = sorted(self.stages, key=lambda stage: stage.layer_start)
        cursor = 0
        stage_ids: set[str] = set()
        for stage in ordered:
            if stage.id in stage_ids:
                raise ValueError(f"duplicate stage id: {stage.id}")
            stage_ids.add(stage.id)
            if stage.node_id not in self.nodes:
                raise ValueError(f"unknown node: {stage.node_id}")
            if stage.layer_start != cursor or stage.layer_end <= stage.layer_start:
                raise ValueError("stages must form consecutive, non-empty layer ranges")
            cursor = stage.layer_end
        if cursor != self.model.num_layers:
            raise ValueError("stages do not cover all model layers")

        stage_by_id = {stage.id: stage for stage in self.stages}
        expected_pairs = tuple(
            (ordered[i].id, ordered[i + 1].id) for i in range(len(ordered) - 1)
        )
        actual_pairs = tuple(
            (boundary.upstream_stage_id, boundary.downstream_stage_id)
            for boundary in self.boundaries
        )
        if len(actual_pairs) != len(set(actual_pairs)):
            raise ValueError("duplicate stage boundary")
        if set(actual_pairs) != set(expected_pairs) or len(actual_pairs) != len(expected_pairs):
            raise ValueError("boundaries must match every adjacent stage pair exactly")
        for boundary in self.boundaries:
            upstream = stage_by_id[boundary.upstream_stage_id]
            downstream = stage_by_id[boundary.downstream_stage_id]
            if upstream.node_id == downstream.node_id:
                if boundary.route_link_ids:
                    raise ValueError("same-node boundary must have an empty route")
                continue
            current = upstream.node_id
            for link_id in boundary.route_link_ids:
                if link_id not in self.links:
                    raise ValueError(f"unknown link: {link_id}")
                link = self.links[link_id]
                if link.source != current:
                    raise ValueError(f"disconnected route at link: {link_id}")
                current = link.target
            if current != downstream.node_id:
                raise ValueError("boundary route does not reach downstream node")

    def layers_by_node(self) -> dict[str, int]:
        result = {node_id: 0 for node_id in self.nodes}
        for stage in self.stages:
            result[stage.node_id] += stage.num_layers
        return result


@dataclass(frozen=True)
class RequestSpec:
    id: str
    arrival_time_s: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SLA:
    ttft_s: float
    tpot_s: float
    queue_overhead_s: float = 0.0
    fixed_overhead_s: float = 0.0


@dataclass(frozen=True)
class EvaluatorConfig:
    decode_block_size: int = 16
    compute_efficiency: float = 0.30
    prefill_contention: float = 0.10
    decode_contention: float = 0.06
    mixed_contention: float = 0.12
    decode_context_scale: float = 4096.0
    decode_context_penalty: float = 0.50
    # Default research semantics: profiling-driven progress. Network remains a
    # conservative SLA-derived reservation/red-line check, but does not itself
    # slow the state machine. Setting this flag to True enables the deliberately
    # stronger full-SLA-window lifetime ablation; it is not the default model.
    conservative_network_lifetime: bool = False
    time_epsilon: float = 1e-9
    progress_epsilon: float = 1e-9
    max_events: int = 1_000_000
    record_trace: bool = False


@dataclass
class RequestRuntime:
    spec: RequestSpec
    phase: Phase
    decode_progress: float = 0.0
    prefill_compute_s: float = 0.0
    prefill_finish_s: float = 0.0
    decode_time_per_token_s: float = 0.0

    def block_index(self, block_size: int, epsilon: float = 0.0) -> int:
        return int((self.decode_progress + epsilon) // block_size)

    def resource_context(self, block_size: int, epsilon: float = 0.0) -> int:
        if self.phase == Phase.PREFILL:
            return self.spec.input_tokens
        completed_block = self.block_index(block_size, epsilon)
        upper_output = min((completed_block + 1) * block_size, self.spec.output_tokens)
        return self.spec.input_tokens + upper_output


@dataclass(frozen=True)
class Violation:
    time_s: float
    kind: ViolationKind
    object_id: str
    required: float
    capacity: float
    request_id: str | None
    num_prefill: int
    num_decode: int


@dataclass(frozen=True)
class StateSnapshot:
    time_s: float
    num_prefill: int
    num_decode: int
    event_types: tuple[str, ...] = ()
    link_required_bytes_per_s: Mapping[str, float] = field(default_factory=dict)
    node_memory_bytes: Mapping[str, float] = field(default_factory=dict)
    request_progress: Mapping[str, float] = field(default_factory=dict)
    request_phase: Mapping[str, str] = field(default_factory=dict)
    request_context: Mapping[str, int] = field(default_factory=dict)


def sorted_workload(workload: Sequence[RequestSpec]) -> tuple[RequestSpec, ...]:
    result = tuple(sorted(workload, key=lambda request: (request.arrival_time_s, request.id)))
    if any(request.arrival_time_s < 0 for request in result):
        raise ValueError("arrival time cannot be negative")
    if any(request.input_tokens <= 0 or request.output_tokens <= 0 for request in result):
        raise ValueError("token lengths must be positive")
    if len({request.id for request in result}) != len(result):
        raise ValueError("request ids must be unique")
    return result
