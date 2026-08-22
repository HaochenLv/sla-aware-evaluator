from __future__ import annotations

import unittest
from dataclasses import replace

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
from sla_aware_mvp.reference import evaluate_reference, find_reference_capacity


class FixedStageProfiler:
    def __init__(self, prefill_s=0.1, decode_s=0.02):
        self.prefill_s = prefill_s
        self.decode_s = decode_s
        self.decode_batch_sizes = []

    def prefill_stage_time(
        self, request, pipeline, stage, n_prefill, n_decode
    ):
        return self.prefill_s

    def decode_stage_time_per_token(
        self, request, context_tokens, pipeline, stage, n_prefill, n_decode
    ):
        self.decode_batch_sizes.append(n_decode)
        return self.decode_s


def build_two_stage_pipeline(link_capacity=1000.0, memory_capacity=10_000_000):
    model = ModelSpec(
        name="toy",
        num_layers=2,
        total_params=1000,
        weight_bytes_per_param=2,
        hidden_size=1,
        num_attention_heads=1,
        num_kv_heads=1,
        kv_element_bytes=1,
        activation_element_bytes=1,
        flops_per_token_per_layer=1.0,
    )
    nodes = {
        "a": GPUNode("a", 1.0, memory_capacity),
        "b": GPUNode("b", 1.0, memory_capacity),
    }
    links = {"a-b": Link("a-b", "a", "b", link_capacity)}
    return Pipeline(
        "toy-two-stage",
        model,
        nodes,
        links,
        (Stage("s0", 0, 1, "a"), Stage("s1", 1, 2, "b")),
        (Boundary("s0", "s1", ("a-b",)),),
    )


def build_one_stage_pipeline(memory_capacity=10_000_000):
    model = ModelSpec(
        name="toy-one",
        num_layers=1,
        total_params=1000,
        weight_bytes_per_param=2,
        hidden_size=1,
        num_attention_heads=1,
        num_kv_heads=1,
        kv_element_bytes=1,
        activation_element_bytes=1,
        flops_per_token_per_layer=1.0,
    )
    nodes = {"a": GPUNode("a", 1.0, memory_capacity)}
    return Pipeline(
        "toy-one-stage",
        model,
        nodes,
        {},
        (Stage("s0", 0, 1, "a"),),
        (),
    )


class ReferenceEvaluatorTests(unittest.TestCase):
    def test_single_request_explicit_compute_and_link_latency(self):
        pipeline = build_two_stage_pipeline(link_capacity=1000.0)
        profiler = FixedStageProfiler(prefill_s=0.1, decode_s=0.02)
        result = evaluate_reference(
            pipeline=pipeline,
            workload=(RequestSpec("r", 0.0, 10, 2),),
            sla=SLA(ttft_s=0.3, tpot_s=0.1, fixed_overhead_s=0.005),
            profiler=profiler,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.completed_requests, 1)
        self.assertEqual(result.completed_tokens, 2)
        self.assertAlmostEqual(result.ttft_s_by_request["r"], 0.215, places=9)
        self.assertAlmostEqual(result.max_tpot_s_by_request["r"], 0.046, places=9)

    def test_explicit_link_service_can_trigger_tpot(self):
        pipeline = build_two_stage_pipeline(link_capacity=10.0)
        result = evaluate_reference(
            pipeline=pipeline,
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=1.0, tpot_s=0.05),
            profiler=FixedStageProfiler(prefill_s=0.001, decode_s=0.001),
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.first_violation.kind, "tpot")
        self.assertGreater(result.first_violation.observed, 0.1)

    def test_decode_queue_forms_microbatch(self):
        pipeline = build_one_stage_pipeline()
        profiler = FixedStageProfiler(prefill_s=0.01, decode_s=0.01)
        result = evaluate_reference(
            pipeline=pipeline,
            workload=(
                RequestSpec("a", 0.0, 1, 1),
                RequestSpec("b", 0.0, 1, 1),
            ),
            sla=SLA(ttft_s=1.0, tpot_s=1.0),
            profiler=profiler,
        )
        self.assertTrue(result.feasible)
        self.assertIn(2, profiler.decode_batch_sizes)

    def test_reference_memory_red_line(self):
        pipeline = build_one_stage_pipeline(memory_capacity=1024)
        result = evaluate_reference(
            pipeline=pipeline,
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=10.0, tpot_s=10.0),
            profiler=FixedStageProfiler(prefill_s=0.01, decode_s=0.01),
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.first_violation.kind, "memory")

    def test_reference_capacity_is_bracketed(self):
        pipeline = build_one_stage_pipeline()
        profiler = FixedStageProfiler(prefill_s=0.1, decode_s=0.01)
        workload = (
            RequestSpec("a", 0.0, 1, 1),
            RequestSpec("b", 1.0, 1, 1),
            RequestSpec("c", 2.0, 1, 1),
        )
        result = find_reference_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=SLA(ttft_s=0.15, tpot_s=0.1),
            profiler=profiler,
            initial_intensity=1.0,
            max_intensity=32.0,
            tolerance=0.05,
            verification_grid_points=5,
        )
        self.assertFalse(result.right_censored)
        self.assertIsNotNone(result.unsafe_intensity)
        self.assertLess(result.safe_intensity, result.unsafe_intensity)
        self.assertTrue(result.representative_safe_run.feasible)
        self.assertIsNotNone(result.representative_unsafe_run)
        self.assertFalse(result.representative_unsafe_run.feasible)
        self.assertTrue(result.monotonicity_verified_on_samples)

    def test_reference_capacity_reports_right_censoring_at_search_limit(self):
        pipeline = build_one_stage_pipeline()
        profiler = FixedStageProfiler(prefill_s=0.01, decode_s=0.01)
        workload = (
            RequestSpec("a", 0.0, 1, 1),
            RequestSpec("b", 10.0, 1, 1),
        )
        result = find_reference_capacity(
            pipeline=pipeline,
            workload=workload,
            sla=SLA(ttft_s=10.0, tpot_s=10.0),
            profiler=profiler,
            initial_intensity=0.25,
            max_intensity=0.5,
            verification_grid_points=5,
        )
        self.assertTrue(result.right_censored)
        self.assertEqual(result.safe_intensity, 0.5)
        self.assertIsNone(result.unsafe_intensity)
        self.assertIsNone(result.representative_unsafe_run)
        self.assertTrue(result.representative_safe_run.feasible)
        self.assertEqual(result.search_max_intensity, 0.5)
        self.assertTrue(result.monotonicity_verified_on_samples)
        self.assertIn(0.5, result.verification_probe_intensities)


if __name__ == "__main__":
    unittest.main()
