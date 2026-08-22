from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .domain import EvaluatorConfig, Pipeline, RequestSpec, SLA
from .evaluator import EvaluationResult, evaluate
from .profiling import AnalyticalProfiler


@dataclass(frozen=True)
class CapacityTrial:
    intensity: float
    feasible: bool
    violation_kind: str | None


@dataclass(frozen=True)
class CapacityResult:
    safe_intensity: float
    unsafe_intensity: float
    trials: tuple[CapacityTrial, ...]
    representative_safe_run: EvaluationResult | None
    representative_unsafe_run: EvaluationResult


def scale_workload(
    workload: Sequence[RequestSpec], intensity: float
) -> tuple[RequestSpec, ...]:
    if intensity <= 0:
        raise ValueError("intensity must be positive")
    if not workload:
        return ()
    origin = min(request.arrival_time_s for request in workload)
    return tuple(
        RequestSpec(
            id=request.id,
            arrival_time_s=origin + (request.arrival_time_s - origin) / intensity,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
        )
        for request in workload
    )


def find_capacity(
    *,
    pipeline: Pipeline,
    workload: Sequence[RequestSpec],
    sla: SLA,
    config: EvaluatorConfig | None = None,
    profiler: AnalyticalProfiler | None = None,
    initial_intensity: float = 0.25,
    tolerance: float = 0.02,
    max_intensity: float = 256.0,
) -> CapacityResult:
    config = config or EvaluatorConfig()
    profiler = profiler or AnalyticalProfiler(config)
    trials: list[CapacityTrial] = []
    cache: dict[float, EvaluationResult] = {}

    def run(intensity: float) -> EvaluationResult:
        if intensity not in cache:
            result = evaluate(
                pipeline=pipeline,
                workload=scale_workload(workload, intensity),
                sla=sla,
                config=config,
                profiler=profiler,
            )
            cache[intensity] = result
            trials.append(
                CapacityTrial(
                    intensity,
                    result.feasible,
                    result.first_violation.kind.value
                    if result.first_violation is not None
                    else None,
                )
            )
        return cache[intensity]

    low = 0.0
    safe_run: EvaluationResult | None = None
    high = initial_intensity
    high_run = run(high)
    if high_run.feasible:
        low = high
        safe_run = high_run
        while high < max_intensity:
            high *= 2
            high_run = run(high)
            if not high_run.feasible:
                break
            low = high
            safe_run = high_run
        else:
            raise RuntimeError("no unsafe intensity found below max_intensity")

    while high - low > tolerance * max(high, 1.0):
        mid = (low + high) / 2
        mid_run = run(mid)
        if mid_run.feasible:
            low = mid
            safe_run = mid_run
        else:
            high = mid
            high_run = mid_run

    ordered = sorted(trials, key=lambda trial: trial.intensity)
    seen_unsafe = False
    for trial in ordered:
        if not trial.feasible:
            seen_unsafe = True
        elif seen_unsafe:
            raise RuntimeError("non-monotonic feasibility observed; use a grid search")
    return CapacityResult(low, high, tuple(trials), safe_run, high_run)

