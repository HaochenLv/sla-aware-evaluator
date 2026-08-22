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
from sla_aware_mvp.reference_diagnostics import (
    ReferenceTraceRecorder,
    diagnose_reference_tpot,
)


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


def _model(name: str, num_layers: int) -> ModelSpec:
    return ModelSpec(
        name=name,
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


def build_two_stage_pipeline(link_capacity: float) -> Pipeline:
    model = _model("diag-two", 2)
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


def build_one_stage_pipeline() -> Pipeline:
    model = _model("diag-one", 1)
    nodes = {"a": GPUNode("a", 1.0, 10_000_000)}
    return Pipeline(
        "diag-one-stage",
        model,
        nodes,
        {},
        (Stage("s0", 0, 1, "a"),),
        (),
    )


class ReferenceDiagnosticTests(unittest.TestCase):
    def test_trace_hook_does_not_change_reference_result(self):
        pipeline = build_two_stage_pipeline(link_capacity=1000.0)
        workload = (RequestSpec("r", 0.0, 10, 2),)
        sla = SLA(ttft_s=1.0, tpot_s=1.0)
        profiler = FixedStageProfiler(prefill_s=0.01, decode_s=0.01)
        baseline = evaluate_reference(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=profiler,
        )
        recorder = ReferenceTraceRecorder()
        traced = evaluate_reference(
            pipeline=pipeline,
            workload=workload,
            sla=sla,
            profiler=profiler,
            trace_sink=recorder,
        )
        self.assertEqual(baseline, traced)
        self.assertGreater(len(recorder.events), 0)

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
        self.assertAlmostEqual(diagnostic.gpu_service_elapsed_s, 0.002, places=9)
        self.assertAlmostEqual(diagnostic.gpu_queue_wait_s, 0.0, places=9)
        self.assertAlmostEqual(diagnostic.explicit_link_service_s, 0.1, places=9)
        self.assertAlmostEqual(diagnostic.link_queue_wait_s, 0.0, places=9)
        self.assertAlmostEqual(diagnostic.residual_wait_s, 0.0, places=9)
        self.assertAlmostEqual(diagnostic.unattributed_s, 0.0, places=9)
        self.assertAlmostEqual(
            diagnostic.fully_attributed_s,
            diagnostic.observed_tpot_s,
            places=9,
        )

    def test_gpu_queue_wait_is_attributed_for_decode_token(self):
        pipeline = build_one_stage_pipeline()
        diagnostic, run = diagnose_reference_tpot(
            pipeline=pipeline,
            workload=(
                RequestSpec("a", 0.0, 1, 2),
                RequestSpec("b", 0.15, 1, 1),
            ),
            sla=SLA(ttft_s=1.0, tpot_s=0.15),
            profiler=FixedStageProfiler(prefill_s=0.1, decode_s=0.1),
            intensity=1.0,
        )
        self.assertFalse(run.feasible)
        self.assertEqual(diagnostic.request_id, "a")
        self.assertEqual(diagnostic.failing_output_token_index, 1)
        self.assertAlmostEqual(diagnostic.gpu_queue_wait_s, 0.1, places=9)
        self.assertAlmostEqual(diagnostic.gpu_service_elapsed_s, 0.1, places=9)
        self.assertAlmostEqual(diagnostic.link_queue_wait_s, 0.0, places=9)
        self.assertAlmostEqual(diagnostic.explicit_link_service_s, 0.0, places=9)
        self.assertAlmostEqual(diagnostic.unattributed_s, 0.0, places=9)
        self.assertAlmostEqual(
            diagnostic.fully_attributed_s,
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
