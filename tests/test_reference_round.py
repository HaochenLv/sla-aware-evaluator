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
from sla_aware_mvp.reference_diagnostics import ReferenceTraceRecorder
from sla_aware_mvp.reference_round import (
    evaluate_reference_round,
    find_reference_capacity_round,
)


class RecordingProfiler:
    def __init__(self, prefill_s=0.0, decode_s=0.1):
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.calls = []

    def prefill_stage_time(self, request, pipeline, stage, n_prefill, n_decode):
        return self.prefill_s

    def decode_stage_time_per_token(
        self, request, context_tokens, pipeline, stage, n_prefill, n_decode
    ):
        self.calls.append((stage.id, n_decode, request.id, context_tokens))
        return self.decode_s


def model(num_layers):
    return ModelSpec(
        "round-toy", num_layers, 1000, 2, 1, 1, 1, 1, 1.0, 1
    )


def one_stage():
    return Pipeline(
        "one",
        model(1),
        {"a": GPUNode("a", 1, 10_000_000)},
        {},
        (Stage("s0", 0, 1, "a"),),
        (),
    )


def two_stage(link=1e9):
    return Pipeline(
        "two",
        model(2),
        {
            "a": GPUNode("a", 1, 10_000_000),
            "b": GPUNode("b", 1, 10_000_000),
        },
        {"a-b": Link("a-b", "a", "b", link)},
        (Stage("s0", 0, 1, "a"), Stage("s1", 1, 2, "b")),
        (Boundary("s0", "s1", ("a-b",)),),
    )


class RoundTests(unittest.TestCase):
    def test_two_stage_late_request_joins_next_round(self):
        trace = ReferenceTraceRecorder()
        result = evaluate_reference_round(
            pipeline=two_stage(),
            workload=(
                RequestSpec("a", 0, 1, 4),
                RequestSpec("b", 0.05, 1, 3),
            ),
            sla=SLA(2, 2),
            profiler=RecordingProfiler(0.0, 0.1),
            trace_sink=trace,
        )
        self.assertTrue(result.feasible)
        starts = [
            event
            for event in trace.events
            if event.event == "service_start"
            and event.resource_kind == "node"
            and event.phase == "decode"
            and event.stage_id == "s0"
        ]
        first = next(
            event
            for event in starts
            if event.request_id == "a" and event.token_index == 0
        )
        self.assertEqual(first.batch_size, 1)
        paired = [event for event in starts if event.batch_size == 2]
        self.assertGreaterEqual(len(paired), 2)
        self.assertEqual({event.request_id for event in paired[:2]}, {"a", "b"})

    def test_round_batch_persists_across_stages(self):
        trace = ReferenceTraceRecorder()
        result = evaluate_reference_round(
            pipeline=two_stage(),
            workload=(
                RequestSpec("a", 0, 1, 2),
                RequestSpec("b", 0, 1, 2),
            ),
            sla=SLA(2, 2),
            profiler=RecordingProfiler(0.0, 0.01),
            trace_sink=trace,
        )
        self.assertTrue(result.feasible)
        for stage in ("s0", "s1"):
            starts = [
                event
                for event in trace.events
                if event.event == "service_start"
                and event.resource_kind == "node"
                and event.phase == "decode"
                and event.stage_id == stage
            ]
            self.assertTrue(starts)
            paired = [event for event in starts if event.batch_size == 2]
            self.assertGreaterEqual(len(paired), 2)
            self.assertEqual(
                {event.request_id for event in paired[:2]}, {"a", "b"}
            )

    def test_single_request_latency_is_finite_and_drains(self):
        result = evaluate_reference_round(
            pipeline=two_stage(1000.0),
            workload=(RequestSpec("r", 0, 10, 2),),
            sla=SLA(1, 1, fixed_overhead_s=0.005),
            profiler=RecordingProfiler(0.1, 0.02),
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.completed_requests, 1)
        self.assertEqual(result.completed_tokens, 2)

    def test_explicit_link_tpot_still_fails(self):
        result = evaluate_reference_round(
            pipeline=two_stage(10.0),
            workload=(RequestSpec("r", 0, 1, 1),),
            sla=SLA(1, 0.05),
            profiler=RecordingProfiler(0.001, 0.001),
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.first_violation.kind, "tpot")

    def test_capacity_frontier_still_brackets(self):
        capacity = find_reference_capacity_round(
            pipeline=one_stage(),
            workload=(
                RequestSpec("a", 0, 1, 1),
                RequestSpec("b", 1, 1, 1),
                RequestSpec("c", 2, 1, 1),
            ),
            sla=SLA(0.15, 0.1),
            profiler=RecordingProfiler(0.1, 0.01),
            initial_intensity=1,
            max_intensity=32,
            tolerance=0.05,
            verification_grid_points=5,
        )
        self.assertFalse(capacity.right_censored)
        self.assertLess(capacity.safe_intensity, capacity.unsafe_intensity)


if __name__ == "__main__":
    unittest.main()
