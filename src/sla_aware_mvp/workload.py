from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

from .domain import RequestSpec


class PrimitiveOnlyUnpickler(pickle.Unpickler):
    """Reject pickle GLOBAL/class construction opcodes from external artifacts."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"pickle global forbidden: {module}.{name}")


def load_primitive_list(path: Path, expected_type: type) -> list:
    with path.open("rb") as stream:
        value = PrimitiveOnlyUnpickler(stream).load()
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected a non-empty primitive list: {path}")
    if any(type(item) is not expected_type for item in value):
        raise ValueError(f"unexpected element type in {path}")
    return value


@dataclass(frozen=True)
class WorkloadProvenance:
    source: str
    kind: str
    commit: str
    generated: bool
    notes: str


@dataclass(frozen=True)
class WorkloadSample:
    requests: tuple[RequestSpec, ...]
    provenance: WorkloadProvenance


def build_helix_azure_conversation_workload(
    artifact_root: Path,
    *,
    commit: str,
    duration_s: int = 30,
    target_request_rate: float = 1.5,
    seed: int = 0,
    interval_offset: int = 0,
) -> WorkloadSample:
    """Build a deterministic finite trace from HELIX's Azure distributions.

    Arrival data contains counts per 3-second interval, not raw timestamps.
    This generator preserves sequential interval shape, rescales its mean, places
    arrivals evenly inside each interval, and samples paired input/output lengths.
    """
    if duration_s <= 0 or duration_s % 3:
        raise ValueError("duration_s must be a positive multiple of 3")
    if target_request_rate <= 0:
        raise ValueError("target_request_rate must be positive")
    base = artifact_root / "simulator/trace_generator"
    arrival_counts = load_primitive_list(
        base / "arrival_rate/azure_conv_arrive_time.pkl", int
    )
    input_lengths = load_primitive_list(
        base / "length_data/azure_conv_input.pkl", int
    )
    output_lengths = load_primitive_list(
        base / "length_data/azure_conv_output.pkl", int
    )
    if len(arrival_counts) != 1200:
        raise ValueError("HELIX Azure arrival series must contain 1200 intervals")
    if len(input_lengths) != len(output_lengths):
        raise ValueError("HELIX Azure input/output arrays must be paired")

    interval_count = duration_s // 3
    selected = [
        arrival_counts[(interval_offset + index) % len(arrival_counts)]
        for index in range(interval_count)
    ]
    source_mean = sum(arrival_counts) / len(arrival_counts)
    scale = 3 * target_request_rate / source_mean
    residual = 0.0
    rng = random.Random(seed)
    requests: list[RequestSpec] = []
    for interval_index, raw_count in enumerate(selected):
        scaled = raw_count * scale + residual
        count = max(round(scaled), 0)
        residual = scaled - count
        for position in range(count):
            arrival = interval_index * 3 + (position + 1) * 3 / (count + 1)
            length_index = rng.randrange(len(input_lengths))
            requests.append(
                RequestSpec(
                    id=f"azure-{len(requests):05d}",
                    arrival_time_s=arrival,
                    input_tokens=input_lengths[length_index],
                    output_tokens=output_lengths[length_index],
                )
            )
    if not requests:
        raise ValueError("selected HELIX workload window produced no requests")
    return WorkloadSample(
        requests=tuple(requests),
        provenance=WorkloadProvenance(
            source="HELIX Azure Conversation artifacts",
            kind="generated_from_interval_counts_and_paired_length_distribution",
            commit=commit,
            generated=True,
            notes="Not raw Azure timestamps; deterministic finite trace derived from HELIX data.",
        ),
    )

