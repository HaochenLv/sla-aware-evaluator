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
    monotonicity_verified_on_samples: bool
    verification_probe_intensities: tuple[float, ...]


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
    verification_grid_points: int = 17,
) -> CapacityResult:
    """Find the first sampled safe-to-unsafe workload frontier.

    A pure binary search silently assumes monotonic feasibility. The event-driven
    evaluator contains discrete block and concurrency changes, so this routine now:

    1. brackets one safe and one unsafe point;
    2. probes a coarse grid inside that bracket;
    3. probes several intensities above the first unsafe point for safe re-entry;
    4. refines only after the sampled points are monotonic.

    This is sampled verification, not a mathematical proof of monotonicity. If a
    sampled safe point occurs after an unsafe point, the routine stops and asks the
    caller to use/report a grid rather than returning a misleading binary-search
    capacity.
    """
    if initial_intensity <= 0:
        raise ValueError("initial_intensity must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_intensity <= 0:
        raise ValueError("max_intensity must be positive")
    if verification_grid_points < 3:
        raise ValueError("verification_grid_points must be at least 3")

    config = config or EvaluatorConfig()
    profiler = profiler or AnalyticalProfiler(config)
    trials: list[CapacityTrial] = []
    cache: dict[float, EvaluationResult] = {}

    def run(intensity: float) -> EvaluationResult:
        key = float(intensity)
        if key not in cache:
            result = evaluate(
                pipeline=pipeline,
                workload=scale_workload(workload, key),
                sla=sla,
                config=config,
                profiler=profiler,
            )
            cache[key] = result
            trials.append(
                CapacityTrial(
                    key,
                    result.feasible,
                    result.first_violation.kind.value
                    if result.first_violation is not None
                    else None,
                )
            )
        return cache[key]

    # First construct an explicit safe/unsafe bracket. If the initial point is
    # unsafe, search downward instead of treating zero as an unevaluated safe run.
    high = min(initial_intensity, max_intensity)
    high_run = run(high)
    if high_run.feasible:
        low = high
        safe_run: EvaluationResult | None = high_run
        while high < max_intensity:
            candidate = min(high * 2.0, max_intensity)
            if candidate == high:
                break
            candidate_run = run(candidate)
            if not candidate_run.feasible:
                high = candidate
                high_run = candidate_run
                break
            low = candidate
            safe_run = candidate_run
            high = candidate
        else:
            raise RuntimeError("no unsafe intensity found below max_intensity")
        if high_run.feasible:
            raise RuntimeError("no unsafe intensity found below max_intensity")
    else:
        unsafe_anchor = high
        unsafe_anchor_run = high_run
        low = high / 2.0
        while low > 1e-9:
            low_run = run(low)
            if low_run.feasible:
                safe_run = low_run
                high = unsafe_anchor
                high_run = unsafe_anchor_run
                break
            unsafe_anchor = low
            unsafe_anchor_run = low_run
            low /= 2.0
        else:
            raise RuntimeError("no safe positive intensity found")

    # Coarse grid verification inside the bracket.
    bracket_low = low
    bracket_high = high
    probe_intensities: list[float] = []
    for index in range(1, verification_grid_points):
        intensity = bracket_low + (bracket_high - bracket_low) * (
            index / verification_grid_points
        )
        probe_intensities.append(intensity)
        run(intensity)

    # Also look beyond the first unsafe bracket for obvious safe re-entry.
    probe = bracket_high * 2.0
    while probe <= max_intensity and len(probe_intensities) < verification_grid_points + 4:
        probe_intensities.append(probe)
        run(probe)
        probe *= 2.0

    ordered = sorted(trials, key=lambda trial: trial.intensity)
    seen_unsafe = False
    for trial in ordered:
        if not trial.feasible:
            seen_unsafe = True
        elif seen_unsafe:
            raise RuntimeError(
                "non-monotonic feasibility observed on sampled intensities; "
                "report/use a grid search instead of a scalar capacity"
            )

    # Rebuild the tightest sampled safe/unsafe pair before numerical refinement.
    safe_trials = [trial for trial in ordered if trial.feasible]
    unsafe_trials = [trial for trial in ordered if not trial.feasible]
    if not safe_trials or not unsafe_trials:
        raise RuntimeError("capacity search failed to retain a safe/unsafe bracket")
    low = max(trial.intensity for trial in safe_trials)
    high = min(trial.intensity for trial in unsafe_trials if trial.intensity > low)
    safe_run = run(low)
    high_run = run(high)

    # Refine only after the explicit sampled monotonicity check above.
    while high - low > tolerance * max(high, 1.0):
        mid = (low + high) / 2.0
        mid_run = run(mid)
        if mid_run.feasible:
            low = mid
            safe_run = mid_run
        else:
            high = mid
            high_run = mid_run

    return CapacityResult(
        safe_intensity=low,
        unsafe_intensity=high,
        trials=tuple(trials),
        representative_safe_run=safe_run,
        representative_unsafe_run=high_run,
        monotonicity_verified_on_samples=True,
        verification_probe_intensities=tuple(probe_intensities),
    )
