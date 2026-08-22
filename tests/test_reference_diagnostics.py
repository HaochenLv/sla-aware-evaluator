from __future__ import annotations

import unittest

from sla_aware_mvp.domain import Boundary, GPUNode, Link, ModelSpec, Pipeline, RequestSpec, SLA, Stage
from sla_aware_mvp.reference_diagnostics import diagnose_reference_tpot


class FixedStageProfiler:
    def __init__(self, prefill_s: float = 0.001, decode_s: float = 0.001):
        self.prefill_s = prefill_s
        self.decode_s = decode_s

    def prefill_stage_time(self, request, pipeline, stage, n_prefill, n_decode):
        return self.prefill_s

    def decode_stage_time_per_token(
        self, request, context_tokens, pipeline, stage, n_prefill, n_decode
    ):
        return self.decode_s


def build_two_stage_pipeline(link_capacity: float) -> Pipeline:
    model = ModelSpec(
        name="diag-toy",
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
        "a": GPUNode("a", 1.0, 10_000_000),
        "b": GPUNode("b", 1.0, 10_000_000),
    }
    links = {"a-b": Link("a-b", "a", "b", link_capacity)}
    return Pipeline(
        "diag-two-stage",
        model,
        nodes,
        links,
        (Stage("s0", 0, 1, "a"), Stage("s1", 1, 2, "b")),
        (Boundary("s0", "s1", ("a-b",)),),
    )


class ReferenceDiagnosticTests(unittest.TestCase):
    def test_tpot_decomposition_accounts_for_explicit_service(self):
        pipeline = build_two_stage_pipeline(link_capacity=10.0)
        diagnostic, run = diagnose_reference_tpot(
            pipeline=pipeline,
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=1.0, tpot_s=0.05),
            profiler=FixedStageProfiler(),
            intensity=1.0,
        )
        self.assertFalse(run.feasible)
        self.assertEqual(diagnostic.violation_kind, "tpot")
        self.assertAlmostEqual(diagnostic.gpu_profile_service_s, 0.002, places=9)
        self.assertAlmostEqual(diagnostic.explicit_link_service_s, 0.1, places=9)
        self.assertAlmostEqual(diagnostic.residual_wait_s, 0.0, places=9)
        self.assertAlmostEqual(
            diagnostic.accounted_service_s,
            diagnostic.observed_tpot_s,
            places=9,
        )

    def test_infinite_network_counterfactual_can_remove_link_driven_failure(self):
        pipeline = build_two_stage_pipeline(link_capacity=10.0)
        diagnostic, _ = diagnose_reference_tpot(
            pipeline=pipeline,
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=1.0, tpot_s=0.05),
            profiler=FixedStageProfiler(),
            intensity=1.0,
            network_capacity_multiplier=1_000_000.0,
        )
        self.assertTrue(diagnostic.counterfactual_feasible)
        self.assertIsNone(diagnostic.counterfactual_first_violation_kind)
        self.assertLess(diagnostic.counterfactual_max_tpot_s, 0.05)


if __name__ == "__main__":
    unittest.main()
