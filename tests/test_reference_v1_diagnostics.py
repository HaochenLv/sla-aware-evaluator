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
from sla_aware_mvp.reference_v1_diagnostics import diagnose_reference_v1_tpot


class FixedProfiler:
    def __init__(self, prefill_s: float = 0.001, decode_s: float = 0.001):
        self.prefill_s = prefill_s
        self.decode_s = decode_s

    def prefill_stage_time(self, request, pipeline, stage, n_prefill, n_decode):
        return self.prefill_s

    def decode_stage_time_per_token(
        self, request, context_tokens, pipeline, stage, n_prefill, n_decode
    ):
        return self.decode_s


def build_pipeline(link_capacity: float) -> Pipeline:
    model = ModelSpec(
        name="diag-v1",
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
    return Pipeline(
        "diag-v1-two",
        model,
        {
            "a": GPUNode("a", 1.0, 10_000_000),
            "b": GPUNode("b", 1.0, 10_000_000),
        },
        {"a-b": Link("a-b", "a", "b", link_capacity)},
        (Stage("s0", 0, 1, "a"), Stage("s1", 1, 2, "b")),
        (Boundary("s0", "s1", ("a-b",)),),
    )


class ReferenceV1DiagnosticTests(unittest.TestCase):
    def test_link_driven_tpot_decomposition_closes(self):
        diagnostic, run = diagnose_reference_v1_tpot(
            pipeline=build_pipeline(10.0),
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=1.0, tpot_s=0.05),
            profiler=FixedProfiler(),
            intensity=1.0,
        )
        self.assertFalse(run.feasible)
        self.assertEqual(diagnostic.violation_kind, "tpot")
        self.assertAlmostEqual(diagnostic.gpu_service_elapsed_s, 0.002, places=9)
        self.assertAlmostEqual(diagnostic.explicit_link_service_s, 0.1, places=9)
        self.assertAlmostEqual(diagnostic.unattributed_s, 0.0, places=9)
        self.assertAlmostEqual(
            diagnostic.fully_attributed_s,
            diagnostic.observed_tpot_s,
            places=9,
        )

    def test_infinite_network_counterfactual_removes_link_only_failure(self):
        diagnostic, _ = diagnose_reference_v1_tpot(
            pipeline=build_pipeline(10.0),
            workload=(RequestSpec("r", 0.0, 1, 1),),
            sla=SLA(ttft_s=1.0, tpot_s=0.05),
            profiler=FixedProfiler(),
            intensity=1.0,
            network_capacity_multiplier=1_000_000.0,
        )
        self.assertTrue(diagnostic.counterfactual_feasible)
        self.assertIsNone(diagnostic.counterfactual_first_violation_kind)


if __name__ == "__main__":
    unittest.main()
