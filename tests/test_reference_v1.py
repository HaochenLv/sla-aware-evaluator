from __future__ import annotations

import unittest

from sla_aware_mvp.domain import (
    Boundary,
    GPUNode,
    Link,
    ModelSpec,
    Pipeline,
    RequestSpec,
    SLA,
    Stage,
)
from sla_aware_mvp.reference import evaluate_reference
from sla_aware_mvp.reference_diagnostics import ReferenceTraceRecorder
from sla_aware_mvp.reference_v1 import evaluate_reference_v1, find_reference_capacity_v1


class RecordingProfiler:
    def __init__(self, prefill_s: float = 0.0, decode_s: float = 0.1):
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.decode_calls: list[tuple[str, int, int]] = []

    def prefill_stage_time(self, request, pipeline, stage, n_prefill, n_decode):
        return self.prefill_s

    def decode_stage_time_per_token(
        self, request, context_tokens, pipeline, stage, n_prefill, n_decode
    ):
        self.decode_calls.append((stage.id, n_decode, context_tokens))
        return self.decode_s


def _model(num_layers: int) -> ModelSpec:
    return ModelSpec(
        name="v1-toy",
        num_layers=num_layers,
        total_params=1000,
        weight_bytes_per_param=2,
        hidden_size=1,
        num_attention_heads=1,
        num_kv_heads=1,
        kv_element_bytes=1,
        activation_element_bytes=1,
        flops_per_token_per_layer=1.0,
    )


def one_stage_pipeline() -> Pipeline:
    return Pipeline(
        "v1-one",
        _model(1),
        {"a": GPUNode("a", 1.0, 10_000_000)},
        {},
        (Stage("s0", 0, 1, "a"),),
        (),
    )


def two_stage_pipeline(link_capacity: float = 1e9) -> Pipeline:
    return Pipeline(
        "v1-two",
        _model(2),
        {
            "a": GPUNode("a", 1.0, 10_000_000),
            "b": GPUNode("b", 1.0, 10_000_000),
        },
        {"a-b": Link("a-b", "a", "b", link_capacity)},
        (Stage("s0", 0, 1, "a"), Stage("s1", 1, 2, "b")),
        (Boundary("s0", "s1", ("a-b",)),),
    )


class ReferenceV1Tests(unittest.TestCase):
    def test_single_request_matches_v0_latency(self):
        pipeline = two_stage_pipeline(link_capacity=1000.0)
        workload = (RequestSpec("r", 0.0, 10, 2),)
        sla = SLA(ttft_s=1.0, tpot_s=1.0, fixed_overhead_s=0.005)
        v0_profiler = RecordingProfiler(prefill_s=0.1, decode_s=0.02)
        v1_profiler = RecordingProfiler(prefill_s=0.1, decode_s=0.02)
        v0 = evaluate_reference(
            pipeline=pipeline, workload=workload, sla=sla, profiler=v0_profiler
        )
        v1 = evaluate_reference_v1(
            pipeline=pipeline, workload=workload, sla=sla, profiler=v1_profiler
        )
        self.assertEqual(v0.feasible, v1.feasible)
        self.assertAlmostEqual(v0.ttft_s_by_request["r"], v1.ttft_s_by_request["r"], places=9)
        self.assertAlmostEqual(
            v0.max_tpot_s_by_request["r"], v1.max_tpot_s_by_request["r"], places=9
        )

    def test_same_time_decode_ready_requests_form_one_cohort(self):
        pipeline = one_stage_pipeline()
        profiler = RecordingProfiler(prefill_s=0.0, decode_s=0.1)
        result = evaluate_reference_v1(
            pipeline=pipeline,
            workload=(
                RequestSpec("a", 0.0, 1, 2),
                RequestSpec("b", 0.0, 1, 2),
            ),
            sla=SLA(ttft_s=1.0, tpot_s=1.0),
            profiler=profiler,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.completed_requests, 2)
        self.assertTrue(profiler.decode_calls)
        self.assertTrue(all(batch_size == 2 for _, batch_size, _ in profiler.decode_calls))

    def test_late_request_joins_next_decode_cohort_not_inflight_one(self):
        pipeline = one_stage_pipeline()
        profiler = RecordingProfiler(prefill_s=0.0, decode_s=0.1)
        recorder = ReferenceTraceRecorder()
        result = evaluate_reference_v1(
            pipeline=pipeline,
            workload=(
                RequestSpec("a", 0.0, 1, 3),
                RequestSpec("b", 0.05, 1, 2),
            ),
            sla=SLA(ttft_s=1.0, tpot_s=1.0),
            profiler=profiler,
            trace_sink=recorder,
        )
        self.assertTrue(result.feasible)
        starts = [
            event
            for event in recorder.events
            if event.event == "service_start"
            and event.resource_kind == "node"
            and event.phase == "decode"
            and event.stage_id == "s0"
        ]
        first_a = next(event for event in starts if event.request_id == "a" and event.token_index == 0)
        self.assertEqual(first_a.batch_size, 1)
        later = [event for event in starts if event.batch_size == 2]
        self.assertTrue(later)
        self.assertEqual({event.request_id for event in later[:2]}, {"a", "b"})

    def test_trace_hook_does_not_change_v1_result(self):
        pipeline = two_stage_pipeline(link_capacity=1000.0)
        workload = (
            RequestSpec("a", 0.0, 1, 2),
            RequestSpec("b", 0.0, 1, 2),
        )
        sla = SLA(ttft_s=1.0, tpot_s=1.0)
        baseline = evaluate_reference_v1(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=RecordingProfiler(prefill_s=0.0, decode_s=0.01),
        )
        recorder = ReferenceTraceRecorder()
        traced = evaluate_reference_v1(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=RecordingProfiler(prefill_s=0.0, decode_s=0.01),
            trace_sink=recorder,
        )
        self.assertEqual(baseline, traced)
        self.assertGreater(len(recorder.events), 0)

    def test_explicit_link_service_still_can_trigger_tpot(self):
        pipeline = two_stage_pipeline(link_capacity=10.0)
        result = evaluate_reference_v1(
            pipeline=pipeline,
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=1.0, tpot_s=0.05),
            profiler=RecordingProfiler(prefill_s=0.001, decode_s=0.001),
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.first_violation.kind, "tpot")
        self.assertGreater(result.first_violation.observed, 0.1)

    def test_reference_v1_capacity_brackets_frontier(self):
        pipeline = one_stage_pipeline()
        result = find_reference_capacity_v1(
            pipeline=pipeline,
            workload=(
                RequestSpec("a", 0.0, 1, 1),
                RequestSpec("b", 1.0, 1, 1),
                RequestSpec("c", 2.0, 1, 1),
            ),
            sla=SLA(ttft_s=0.15, tpot_s=0.1),
            profiler=RecordingProfiler(prefill_s=0.1, decode_s=0.01),
            initial_intensity=1.0,
            max_intensity=32.0,
            tolerance=0.05,
            verification_grid_points=5,
        )
        self.assertFalse(result.right_censored)
        self.assertIsNotNone(result.unsafe_intensity)
        self.assertLess(result.safe_intensity, result.unsafe_intensity)
        self.assertTrue(result.representative_safe_run.feasible)
        self.assertFalse(result.representative_unsafe_run.feasible)


if __name__ == "__main__":
    unittest.main()
