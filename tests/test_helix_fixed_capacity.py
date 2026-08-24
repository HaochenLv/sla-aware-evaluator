from __future__ import annotations

import unittest
from unittest.mock import patch

from sla_aware_mvp.domain import RequestSpec, SLA
from sla_aware_mvp.helix_fixed_capacity import find_helix_fixed_capacity
from sla_aware_mvp.helix_fixed_reference import HelixReferenceResult


def _run(feasible: bool) -> HelixReferenceResult:
    return HelixReferenceResult(
        feasible=feasible,
        finished_requests=1,
        total_requests=1,
        final_time_s=1.0,
        first_violation_kind=None if feasible else "tpot",
        first_violation_request_id=None if feasible else "r",
        first_violation_observed_s=None if feasible else 0.2,
        first_violation_limit_s=None if feasible else 0.15,
        query_metrics={},
    )


class HelixFixedCapacityTests(unittest.TestCase):
    def test_capacity_brackets_monotonic_frontier(self):
        def fake_evaluate(*, workload, **kwargs):
            # scale_workload maps the second arrival to 1/intensity.
            intensity = 1.0 / workload[1].arrival_time_s
            return _run(intensity <= 1.0)

        workload = (
            RequestSpec("r0", 0.0, 1, 1),
            RequestSpec("r1", 1.0, 1, 1),
        )
        with patch(
            "sla_aware_mvp.helix_fixed_capacity.evaluate_helix_fixed_reference",
            side_effect=fake_evaluate,
        ):
            result = find_helix_fixed_capacity(
                pipeline=object(),
                workload=workload,
                sla=SLA(1.0, 0.1),
                helix_root="/unused",
                initial_intensity=0.5,
                tolerance=0.01,
                max_intensity=4.0,
            )
        self.assertLessEqual(result.safe_intensity, 1.0)
        self.assertGreater(result.unsafe_intensity, 1.0)
        self.assertFalse(result.right_censored)


if __name__ == "__main__":
    unittest.main()
