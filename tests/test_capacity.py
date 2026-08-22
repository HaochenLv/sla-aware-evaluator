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

    def test_capacity_is_bracketed(self):
        pipeline, _ = build_pipelines()
        result = find_capacity(
            pipeline=pipeline,
            workload=build_workload(),
            sla=SLA(ttft_s=1.0, tpot_s=0.070, fixed_overhead_s=0.005),
            tolerance=0.03,
        )
        self.assertGreater(result.safe_intensity, 0)
        self.assertGreater(result.unsafe_intensity, result.safe_intensity)
        self.assertFalse(result.representative_unsafe_run.feasible)


if __name__ == "__main__":
    unittest.main()

