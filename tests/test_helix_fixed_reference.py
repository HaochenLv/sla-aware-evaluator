from __future__ import annotations

import unittest
from types import SimpleNamespace

from sla_aware_mvp.domain import (
    Boundary,
    GPUNode,
    Link,
    ModelSpec,
    Pipeline,
    Stage,
)
from sla_aware_mvp.helix_fixed_reference import (
    extract_helix_query_metrics,
    validate_helix_fixed_reference_pipeline,
)


def _model() -> ModelSpec:
    return ModelSpec(
        name="LLaMA-2-70B",
        num_layers=4,
        total_params=4e9,
        weight_bytes_per_param=2,
        hidden_size=8,
        num_attention_heads=2,
        num_kv_heads=2,
        kv_element_bytes=2,
        flops_per_token_per_layer=1.0,
    )


def _direct_pipeline() -> Pipeline:
    nodes = {
        "a": GPUNode("a", 1, 10_000, hardware_type="A100-40GB"),
        "b": GPUNode("b", 1, 10_000, hardware_type="A100-40GB"),
    }
    links = {"a-b": Link("a-b", "a", "b", 1000.0)}
    return Pipeline(
        "direct",
        _model(),
        nodes,
        links,
        (Stage("s0", 0, 2, "a"), Stage("s1", 2, 4, "b")),
        (Boundary("s0", "s1", ("a-b",)),),
    )


class HelixFixedReferenceTests(unittest.TestCase):
    def test_current_direct_stage_boundary_is_supported(self):
        validate_helix_fixed_reference_pipeline(_direct_pipeline())

    def test_multihop_boundary_is_explicitly_rejected_in_mvp(self):
        base = _direct_pipeline()
        nodes = {
            **base.nodes,
            "x": GPUNode("x", 1, 10_000, hardware_type="A100-40GB"),
        }
        links = {
            "a-x": Link("a-x", "a", "x", 1000.0),
            "x-b": Link("x-b", "x", "b", 1000.0),
        }
        pipeline = Pipeline(
            "multi-hop",
            base.model,
            nodes,
            links,
            base.stages,
            (Boundary("s0", "s1", ("a-x", "x-b")),),
        )
        with self.assertRaisesRegex(ValueError, "one direct physical link"):
            validate_helix_fixed_reference_pipeline(pipeline)

    def test_query_metric_extraction_keeps_aligned_and_true_ttft_separate(self):
        query = SimpleNamespace(
            creation_time=10.0,
            inference_history=[
                SimpleNamespace(start_time=10.0, end_time=10.4),
                SimpleNamespace(start_time=10.4, end_time=10.46),
                SimpleNamespace(start_time=10.46, end_time=10.53),
            ],
        )
        metric = extract_helix_query_metrics(
            request_id="r", query=query, overhead_s=0.005
        )
        self.assertAlmostEqual(metric.aligned_ttft_s, 0.405)
        self.assertAlmostEqual(metric.true_first_token_ttft_s, 0.465)
        self.assertEqual(len(metric.decode_tpot_s), 2)
        self.assertAlmostEqual(metric.decode_tpot_s[0], 0.065)
        self.assertAlmostEqual(metric.decode_tpot_s[1], 0.075)
        self.assertAlmostEqual(metric.max_tpot_s, 0.075)


if __name__ == "__main__":
    unittest.main()
