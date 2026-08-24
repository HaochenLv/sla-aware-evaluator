from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .capacity import scale_workload
from .domain import Pipeline, RequestSpec, SLA
from .helix_fixed_reference import HelixReferenceResult, evaluate_helix_fixed_reference


@dataclass(frozen=True)
class HelixCapacityTrial:
    intensity: float
    feasible: bool
    violation_kind: str | None


@dataclass(frozen=True)
class HelixCapacityResult:
    safe_intensity: float
    unsafe_intensity: float | None
    safe_run: HelixReferenceResult
    unsafe_run: HelixReferenceResult | None
    trials: tuple[HelixCapacityTrial, ...]
    right_censored: bool


def find_helix_fixed_capacity(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    helix_root: str | Path,
    initial_intensity: float = 0.25,
    tolerance: float = 0.01,
    max_intensity: float = 16.0,
) -> HelixCapacityResult:
    """Find a finite-workload safe/unsafe request-rate bracket with HELIX replay.

    This wrapper deliberately adds no new serving semantics: each trial only
    rescales the same request arrival timestamps and calls the fixed-pipeline
    HELIX Reference. It assumes the usual monotonic safe-to-unsafe frontier and
    explicitly checks all evaluated trial points for sampled safe re-entry.
    """
    if initial_intensity <= 0 or tolerance <= 0 or max_intensity <= 0:
        raise ValueError("capacity-search parameters must be positive")

    cache: dict[float, HelixReferenceResult] = {}
    trials: list[HelixCapacityTrial] = []

    def run(intensity: float) -> HelixReferenceResult:
        key = float(intensity)
        if key not in cache:
            result = evaluate_helix_fixed_reference(
                pipeline=pipeline,
                workload=scale_workload(workload, key),
                sla=sla,
                helix_root=helix_root,
            )
            cache[key] = result
            trials.append(
                HelixCapacityTrial(
                    intensity=key,
                    feasible=result.feasible,
                    violation_kind=result.first_violation_kind,
                )
            )
        return cache[key]

    high = min(initial_intensity, max_intensity)
    high_run = run(high)

    if high_run.feasible:
        low = high
        low_run = high_run
        while high < max_intensity:
            candidate = min(high * 2.0, max_intensity)
            candidate_run = run(candidate)
            if not candidate_run.feasible:
                high, high_run = candidate, candidate_run
                break
            low, low_run = candidate, candidate_run
            high = candidate
        else:
            return HelixCapacityResult(
                safe_intensity=low,
                unsafe_intensity=None,
                safe_run=low_run,
                unsafe_run=None,
                trials=tuple(trials),
                right_censored=True,
            )
        if high_run.feasible:
            return HelixCapacityResult(
                safe_intensity=low,
                unsafe_intensity=None,
                safe_run=low_run,
                unsafe_run=None,
                trials=tuple(trials),
                right_censored=True,
            )
    else:
        while high > 1e-8 and not high_run.feasible:
            candidate = high / 2.0
            candidate_run = run(candidate)
            if candidate_run.feasible:
                low, low_run = candidate, candidate_run
                break
            high, high_run = candidate, candidate_run
        else:
            raise RuntimeError("no positive safe HELIX intensity found")

    while high - low > tolerance * max(high, 1.0):
        mid = (low + high) / 2.0
        mid_run = run(mid)
        if mid_run.feasible:
            low, low_run = mid, mid_run
        else:
            high, high_run = mid, mid_run

    ordered = sorted(trials, key=lambda trial: trial.intensity)
    seen_unsafe = False
    for trial in ordered:
        if not trial.feasible:
            seen_unsafe = True
        elif seen_unsafe:
            raise RuntimeError(
                "non-monotonic HELIX feasibility observed on sampled intensities"
            )

    return HelixCapacityResult(
        safe_intensity=low,
        unsafe_intensity=high,
        safe_run=low_run,
        unsafe_run=high_run,
        trials=tuple(trials),
        right_censored=False,
    )
