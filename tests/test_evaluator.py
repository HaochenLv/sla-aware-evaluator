from __future__ import annotations

import unittest
from dataclasses import replace

from sla_aware_mvp.demo import build_pipelines, build_workload
from sla_aware_mvp.domain import (
    EvaluatorConfig,
    Phase,
    RequestRuntime,
    RequestSpec,
    SLA,
    ViolationKind,
)
from sla_aware_mvp.evaluator import evaluate


class FixedProfiler:
    def prefill_time(self, request, pipeline, n_prefill, n_decode):
        return 0.001

    def decode_time_per_token(
        self, request, context_tokens, pipeline, n_prefill, n_decode
    ):
        return 4.0 if n_prefill else 3.0


class FastProfiler:
    def prefill_time(self, request, pipeline, n_prefill, n_decode):
        return 0.01

    def decode_time_per_token(
        self, request, context_tokens, pipeline, n_prefill, n_decode
    ):
        return 0.01


class EvaluatorTests(unittest.TestCase):
    def test_pipeline_validation(self):
        for pipeline in build_pipelines():
            pipeline.validate()

    def test_finite_workload_drains(self):
        pipeline, _ = build_pipelines()
        result = evaluate(
            pipeline=pipeline,
            workload=build_workload()[:2],
            sla=SLA(ttft_s=2.0, tpot_s=0.1),
        )
        self.assertTrue(result.feasible)
        self.assertGreater(result.final_time_s, build_workload()[1].arrival_time_s)

    def test_continuous_progress_survives_arrival_without_network_lifetime(self):
        _, pipeline = build_pipelines()
        workload = (
            RequestSpec("decode", 0.0, 1, 48),
            # Prefill finishes at .001; 32 tokens at 96.001; after another 20s:
            RequestSpec("arrival", 116.001, 1, 1),
        )
        result = evaluate(
            pipeline=pipeline,
            workload=workload,
            sla=SLA(ttft_s=10.0, tpot_s=10.0),
            config=EvaluatorConfig(
                record_trace=True, conservative_network_lifetime=False
            ),
            profiler=FixedProfiler(),
        )
        snapshot = next(
            item for item in result.trace if abs(item.time_s - 116.001) < 1e-8
        )
        self.assertAlmostEqual(snapshot.request_progress["decode"], 38.6666667, places=5)

    def test_network_commitment_extends_active_lifetime(self):
        _, pipeline = build_pipelines()
        request = RequestSpec("r", 0.0, 1, 2)
        conservative = evaluate(
            pipeline=pipeline,
            workload=(request,),
            sla=SLA(ttft_s=1.0, tpot_s=0.5),
            config=EvaluatorConfig(conservative_network_lifetime=True),
            profiler=FastProfiler(),
        )
        compute_only = evaluate(
            pipeline=pipeline,
            workload=(request,),
            sla=SLA(ttft_s=1.0, tpot_s=0.5),
            config=EvaluatorConfig(conservative_network_lifetime=False),
            profiler=FastProfiler(),
        )
        self.assertTrue(conservative.feasible)
        self.assertTrue(compute_only.feasible)
        self.assertGreater(conservative.final_time_s, compute_only.final_time_s)

    def test_block_context_uses_same_epsilon_policy(self):
        runtime = RequestRuntime(
            RequestSpec("r", 0.0, 100, 64),
            Phase.DECODE,
            decode_progress=16.0 - 5e-10,
        )
        context = runtime.resource_context(block_size=16, epsilon=1e-9)
        self.assertEqual(runtime.block_index(16, 1e-9), 1)
        self.assertEqual(context, 132)

    def test_simultaneous_arrivals_are_order_independent(self):
        _, pipeline = build_pipelines()
        a = RequestSpec("a", 0.0, 512, 32)
        b = RequestSpec("b", 0.0, 768, 32)
        kwargs = dict(
            pipeline=pipeline,
            sla=SLA(ttft_s=2.0, tpot_s=0.1),
            config=EvaluatorConfig(record_trace=True),
        )
        first = evaluate(workload=(a, b), **kwargs)
        second = evaluate(workload=(b, a), **kwargs)
        self.assertEqual(first.feasible, second.feasible)
        self.assertEqual(first.trace, second.trace)

    def test_sla_time_red_line(self):
        _, pipeline = build_pipelines()
        result = evaluate(
            pipeline=pipeline,
            workload=(RequestSpec("r", 0.0, 1024, 16),),
            sla=SLA(ttft_s=0.001, tpot_s=0.1),
        )
        self.assertEqual(result.first_violation.kind, ViolationKind.SLA_TIME)
        self.assertGreaterEqual(len(result.first_violations), 1)

    def test_simultaneous_network_violations_are_retained(self):
        pipeline, _ = build_pipelines()
        tiny_links = {
            link_id: replace(link, capacity_bytes_per_s=1.0)
            for link_id, link in pipeline.links.items()
        }
        tiny_pipeline = replace(pipeline, links=tiny_links)
        result = evaluate(
            pipeline=tiny_pipeline,
            workload=(RequestSpec("r", 0.0, 16, 1),),
            sla=SLA(ttft_s=10.0, tpot_s=10.0),
            profiler=FastProfiler(),
        )
        network_objects = {
            violation.object_id
            for violation in result.first_violations
            if violation.kind == ViolationKind.NETWORK
        }
        self.assertIn("a-b", network_objects)
        self.assertIn("b-c", network_objects)

    def test_memory_red_line(self):
        _, pipeline = build_pipelines()
        tiny_nodes = {
            node_id: replace(node, memory_capacity_bytes=1024)
            for node_id, node in pipeline.nodes.items()
        }
        tiny_pipeline = replace(pipeline, nodes=tiny_nodes)
        result = evaluate(
            pipeline=tiny_pipeline,
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=10.0, tpot_s=10.0),
        )
        self.assertEqual(result.first_violation.kind, ViolationKind.MEMORY)


if __name__ == "__main__":
    unittest.main()
