from __future__ import annotations

import unittest

from sla_aware_mvp.capacity import find_capacity, scale_workload
from sla_aware_mvp.demo import build_pipelines, build_workload
from sla_aware_mvp.domain import SLA


class CapacityTests(unittest.TestCase):
    def test_scaling_preserves_lengths_and_compresses_arrivals(self):
        workload = build_workload()[:3]
        scaled = scale_workload(workload, 2.0)
        self.assertEqual(scaled[1].input_tokens, workload[1].input_tokens)
        self.assertAlmostEqual(scaled[2].arrival_time_s, workload[2].arrival_time_s / 2)

    def test_capacity_is_bracketed_after_sampled_monotonicity_check(self):
        pipeline, _ = build_pipelines()
        result = find_capacity(
            pipeline=pipeline,
            workload=build_workload(),
            sla=SLA(ttft_s=1.0, tpot_s=0.070, fixed_overhead_s=0.005),
            tolerance=0.03,
            verification_grid_points=7,
        )
        self.assertFalse(result.right_censored)
        self.assertGreater(result.safe_intensity, 0)
        self.assertIsNotNone(result.unsafe_intensity)
        self.assertGreater(result.unsafe_intensity, result.safe_intensity)
        self.assertIsNotNone(result.representative_unsafe_run)
        self.assertFalse(result.representative_unsafe_run.feasible)
        self.assertTrue(result.monotonicity_verified_on_samples)
        self.assertGreater(len(result.verification_probe_intensities), 0)

    def test_capacity_reports_right_censoring_at_search_limit(self):
        pipeline, _ = build_pipelines()
        result = find_capacity(
            pipeline=pipeline,
            workload=build_workload()[:2],
            sla=SLA(ttft_s=100.0, tpot_s=100.0),
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
