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
from sla_aware_mvp.reference_round_diagnostics import diagnose_reference_round_tpot


class FixedProfiler:
    def __init__(self, prefill_s=0.001, decode_s=0.001):
        self.prefill_s = prefill_s
        self.decode_s = decode_s

    def prefill_stage_time(self, request, pipeline, stage, n_prefill, n_decode):
        return self.prefill_s

    def decode_stage_time_per_token(
        self, request, context_tokens, pipeline, stage, n_prefill, n_decode
    ):
        return self.decode_s


def build_pipeline(link_capacity):
    model = ModelSpec(
        "diag-round", 2, 1000, 2, 1, 1, 1, 1, 1.0, 1
    )
    return Pipeline(
        "diag-round-two",
        model,
        {
            "a": GPUNode("a", 1.0, 10_000_000),
            "b": GPUNode("b", 1.0, 10_000_000),
        },
        {"a-b": Link("a-b", "a", "b", link_capacity)},
        (Stage("s0", 0, 1, "a"), Stage("s1", 1, 2, "b")),
        (Boundary("s0", "s1", ("a-b",)),),
    )


class ReferenceRoundDiagnosticTests(unittest.TestCase):
    def test_link_driven_tpot_decomposition_closes(self):
        diagnostic, run = diagnose_reference_round_tpot(
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
        diagnostic, _ = diagnose_reference_round_tpot(
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
