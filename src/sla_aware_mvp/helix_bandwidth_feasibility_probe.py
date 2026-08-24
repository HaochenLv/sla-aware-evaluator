from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import RequestSpec, SLA
from .helix_bandwidth_sweep import build_bandwidth_pipeline
from .helix_demo import HELIX_COMMIT
from .helix_fixed_reference import evaluate_helix_fixed_reference
from .workload import build_helix_azure_conversation_workload


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    root = Path(root_env)
    offset = int(os.environ.get("VALIDATION_OFFSET", "0"))
    bandwidth_gbps = float(os.environ["BANDWIDTH_GBPS"])

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
        interval_offset=offset,
    )
    longest = max(sample.requests, key=lambda request: request.input_tokens)
    isolated = RequestSpec(
        id=f"isolated-{longest.id}",
        arrival_time_s=0.0,
        input_tokens=longest.input_tokens,
        output_tokens=1,
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    pipeline = build_bandwidth_pipeline(bandwidth_gbps)
    run = evaluate_helix_fixed_reference(
        pipeline=pipeline,
        workload=(isolated,),
        sla=sla,
        helix_root=root,
    )
    metric = run.query_metrics[isolated.id]
    result = {
        "design": {
            "purpose": "isolate the per-request network feasibility threshold without queueing or request overlap",
            "controlled_variables": "same LLaMA-70B 8x10-layer A100 pipeline, one longest-prompt request, SLA, and pinned HELIX runtime",
            "changed_variable": "all seven inter-stage links use one selected bandwidth",
            "reference_ttft_semantics": "aligned Prefill completion is the current evaluator-aligned feasibility metric; true first-token TTFT remains diagnostic",
            "helix_commit": HELIX_COMMIT,
            "interval_offset": offset,
            "bandwidth_gbps": bandwidth_gbps,
            "sla": {
                "ttft_s": sla.ttft_s,
                "tpot_s": sla.tpot_s,
                "fixed_overhead_s": sla.fixed_overhead_s,
            },
        },
        "request": {
            "source_request_id": longest.id,
            "input_tokens": longest.input_tokens,
            "source_output_tokens": longest.output_tokens,
            "probe_output_tokens": isolated.output_tokens,
        },
        "result": {
            "feasible": run.feasible,
            "first_violation_kind": run.first_violation_kind,
            "aligned_ttft_s": metric.aligned_ttft_s,
            "true_first_token_ttft_s": metric.true_first_token_ttft_s,
            "max_tpot_s": metric.max_tpot_s,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
