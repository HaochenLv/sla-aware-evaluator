from __future__ import annotations

import unittest

from sla_aware_mvp.domain import GPUNode, ModelSpec, Pipeline, RequestSpec, SLA, Stage
from sla_aware_mvp.reference import evaluate_reference
from sla_aware_mvp.reference_queue_blocker_diagnostics import diagnose_gpu_queue_blockers


class FixedProfiler:
    def __init__(self, prefill_s=0.1, decode_s=0.1):
        self.prefill_s = prefill_s
        self.decode_s = decode_s

    def prefill_stage_time(self, request, pipeline, stage, n_prefill, n_decode):
        return self.prefill_s

    def decode_stage_time_per_token(
        self, request, context_tokens, pipeline, stage, n_prefill, n_decode
    ):
        return self.decode_s


def pipeline():
    model = ModelSpec(
        "blocker-toy", 1, 1000, 2, 1, 1, 1, 1, 1.0, 1
    )
    return Pipeline(
        "blocker-one",
        model,
        {"a": GPUNode("a", 1.0, 10_000_000)},
        {},
        (Stage("s0", 0, 1, "a"),),
        (),
    )


class QueueBlockerDiagnosticTests(unittest.TestCase):
    def test_constructed_queue_wait_is_prefill_blocked(self):
        diagnostic, run = diagnose_gpu_queue_blockers(
            evaluator=evaluate_reference,
            pipeline=pipeline(),
            workload=(
                RequestSpec("a", 0.0, 1, 2),
                RequestSpec("b", 0.15, 1, 1),
            ),
            sla=SLA(ttft_s=1.0, tpot_s=0.15),
            profiler=FixedProfiler(0.1, 0.1),
            intensity=1.0,
        )
        self.assertFalse(run.feasible)
        self.assertEqual(diagnostic.request_id, "a")
        self.assertEqual(diagnostic.token_index, 1)
        self.assertAlmostEqual(diagnostic.gpu_queue_wait_s, 0.1, places=9)
        self.assertAlmostEqual(diagnostic.blocked_by_prefill_s, 0.1, places=9)
        self.assertAlmostEqual(diagnostic.blocked_by_decode_s, 0.0, places=9)
        self.assertAlmostEqual(diagnostic.idle_or_sync_s, 0.0, places=9)

    def test_queue_breakdown_closes(self):
        diagnostic, _ = diagnose_gpu_queue_blockers(
            evaluator=evaluate_reference,
            pipeline=pipeline(),
            workload=(
                RequestSpec("a", 0.0, 1, 2),
                RequestSpec("b", 0.15, 1, 1),
            ),
            sla=SLA(ttft_s=1.0, tpot_s=0.15),
            profiler=FixedProfiler(0.1, 0.1),
            intensity=1.0,
        )
        self.assertAlmostEqual(
            diagnostic.gpu_queue_wait_s,
            diagnostic.blocked_by_prefill_s
            + diagnostic.blocked_by_decode_s
            + diagnostic.idle_or_sync_s,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
