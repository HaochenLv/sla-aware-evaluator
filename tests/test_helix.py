from __future__ import annotations

import unittest

from sla_aware_mvp.helix import HelixA100Llama2Profiler, HelixLayerProfile
from sla_aware_mvp.helix_demo import (
    HELIX_COMMIT,
    artifact_root,
    build_helix_pipelines,
)
from sla_aware_mvp.domain import RequestSpec
from sla_aware_mvp.workload import build_helix_azure_conversation_workload


class HelixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = artifact_root()

    def test_ms_conversion_and_interpolation(self):
        profile = HelixLayerProfile.from_csv(
            self.root / "simulator/model_manager/llama2_70b/a100/prompt_bs2time.csv",
            x_kind="prompt_tokens",
        )
        self.assertAlmostEqual(profile.lookup_seconds_per_layer(250), 0.002)
        self.assertAlmostEqual(profile.lookup_seconds_per_layer(375), 0.003)

    def test_profile_range_is_strict(self):
        profile = HelixLayerProfile.from_csv(
            self.root / "simulator/model_manager/llama2_70b/a100/decode_bs2time.csv",
            x_kind="decode_active_tokens",
        )
        with self.assertRaises(ValueError):
            profile.lookup_seconds_per_layer(801)
        self.assertAlmostEqual(profile.lookup_seconds_per_layer(1), 0.0007)

    def test_stage_time_scales_with_layers(self):
        profiler = HelixA100Llama2Profiler.from_artifact(
            self.root, commit=HELIX_COMMIT
        )
        pipeline, _ = build_helix_pipelines()
        sample = RequestSpec("sample", 0.0, 750, 32)
        expected = (
            profiler.prompt.lookup_seconds_per_layer(sample.input_tokens)
            * pipeline.model.num_layers
        )
        pipeline_prefill = profiler.prefill_time(sample, pipeline, 1, 0)
        stage_prefill = sum(
            profiler.prefill_stage_time(sample, pipeline, stage, 1, 0)
            for stage in pipeline.stages
        )
        self.assertAlmostEqual(pipeline_prefill, expected)
        self.assertAlmostEqual(stage_prefill, pipeline_prefill)

        context = sample.input_tokens + 7
        pipeline_decode = profiler.decode_time_per_token(
            sample, context, pipeline, 0, 8
        )
        stage_decode = sum(
            profiler.decode_stage_time_per_token(
                sample, context, pipeline, stage, 0, 8
            )
            for stage in pipeline.stages
        )
        self.assertAlmostEqual(stage_decode, pipeline_decode)
        self.assertNotIn("context_len", profiler.provenance.observed_dimensions)

    def test_azure_workload_is_safe_and_deterministic(self):
        first = build_helix_azure_conversation_workload(
            self.root, commit=HELIX_COMMIT, duration_s=30, seed=9
        )
        second = build_helix_azure_conversation_workload(
            self.root, commit=HELIX_COMMIT, duration_s=30, seed=9
        )
        self.assertEqual(first, second)
        self.assertGreater(len(first.requests), 0)
        self.assertTrue(all(request.input_tokens <= 2047 for request in first.requests))
        self.assertTrue(first.provenance.generated)


if __name__ == "__main__":
    unittest.main()
