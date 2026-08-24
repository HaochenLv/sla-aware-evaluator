from __future__ import annotations

import json

from .domain import SLA
from .helix import HelixA100Llama2Profiler
from .helix_demo import HELIX_COMMIT, artifact_root, build_helix_pipelines
from .reference import evaluate_reference
from .reference_queue_blocker_diagnostics import diagnose_gpu_queue_blockers
from .reference_round import evaluate_reference_round
from .workload import build_helix_azure_conversation_workload


# Exact first-unsafe intensity reproduced by E4 for both Slow/Fast and both v0/
# Decode-round under this fixed 30 s workload/SLA. Reusing it avoids repeating
# capacity search inside an observational diagnostic.
E4_UNSAFE_INTENSITY = 0.026041666666666664


def _diag_dict(diag) -> dict:
    return {
        "request_id": diag.request_id,
        "token_index": diag.token_index,
        "observed_tpot_s": diag.observed_tpot_s,
        "gpu_queue_wait_s": diag.gpu_queue_wait_s,
        "blocked_by_prefill_s": diag.blocked_by_prefill_s,
        "blocked_by_decode_s": diag.blocked_by_decode_s,
        "idle_or_sync_s": diag.idle_or_sync_s,
        "fractions_of_gpu_queue": {
            "prefill": (
                diag.blocked_by_prefill_s / diag.gpu_queue_wait_s
                if diag.gpu_queue_wait_s > 0
                else 0.0
            ),
            "decode": (
                diag.blocked_by_decode_s / diag.gpu_queue_wait_s
                if diag.gpu_queue_wait_s > 0
                else 0.0
            ),
            "idle_or_sync": (
                diag.idle_or_sync_s / diag.gpu_queue_wait_s
                if diag.gpu_queue_wait_s > 0
                else 0.0
            ),
        },
        "per_stage": [
            {
                "stage_id": item.stage_id,
                "gpu_queue_wait_s": item.gpu_queue_wait_s,
                "blocked_by_prefill_s": item.blocked_by_prefill_s,
                "blocked_by_decode_s": item.blocked_by_decode_s,
                "idle_or_sync_s": item.idle_or_sync_s,
            }
            for item in diag.per_stage
        ],
    }


def main() -> None:
    root = artifact_root()
    profiler = HelixA100Llama2Profiler.from_artifact(root, commit=HELIX_COMMIT)
    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
    )
    workload = sample.requests
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    output = {
        "experiment": "GPU queue blocker phase attribution",
        "helix_commit": HELIX_COMMIT,
        "workload_requests": len(workload),
        "intensity_source": "E4 exact first-unsafe bracket upper bound",
        "unsafe_intensity": E4_UNSAFE_INTENSITY,
        "pipelines": {},
    }

    for pipeline in build_helix_pipelines():
        per_pipeline = {}
        for name, evaluator in (
            ("reference_v0", evaluate_reference),
            ("reference_decode_round", evaluate_reference_round),
        ):
            diagnostic, _ = diagnose_gpu_queue_blockers(
                evaluator=evaluator,
                pipeline=pipeline,
                workload=workload,
                sla=sla,
                profiler=profiler,
                intensity=E4_UNSAFE_INTENSITY,
            )
            per_pipeline[name] = _diag_dict(diagnostic)
        output["pipelines"][pipeline.id] = per_pipeline

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
