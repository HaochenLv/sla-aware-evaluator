from __future__ import annotations

import json
import os
from pathlib import Path

from .domain import SLA
from .helix_bandwidth_ordering_validation import _capacity_summary
from .helix_bandwidth_sweep import build_bandwidth_pipeline
from .helix_demo import HELIX_COMMIT
from .workload import build_helix_azure_conversation_workload


def main() -> None:
    root_env = os.environ.get("HELIX_ROOT")
    if not root_env:
        raise RuntimeError("set HELIX_ROOT")
    bandwidth_env = os.environ.get("BANDWIDTH_GBPS")
    if not bandwidth_env:
        raise RuntimeError("set BANDWIDTH_GBPS")

    root = Path(root_env)
    gbps = float(bandwidth_env)
    offset = int(os.environ.get("VALIDATION_OFFSET", "0"))

    sample = build_helix_azure_conversation_workload(
        root,
        commit=HELIX_COMMIT,
        duration_s=30,
        target_request_rate=1.5,
        seed=7,
        interval_offset=offset,
    )
    workload = sample.requests
    base_rate = (len(workload) - 1) / (
        workload[-1].arrival_time_s - workload[0].arrival_time_s
    )
    sla = SLA(ttft_s=2.0, tpot_s=0.150, fixed_overhead_s=0.005)
    pipeline = build_bandwidth_pipeline(gbps)
    summary = _capacity_summary(
        pipeline=pipeline,
        workload=workload,
        sla=sla,
        root=root,
        base_rate=base_rate,
    )

    result = {
        "design": {
            "purpose": "parallel shard of E18 HELIX bandwidth sweep",
            "interval_offset": offset,
            "bandwidth_gbps": gbps,
            "helix_commit": HELIX_COMMIT,
            "seed": 7,
            "sla": {
                "ttft_s": sla.ttft_s,
                "tpot_s": sla.tpot_s,
                "fixed_overhead_s": sla.fixed_overhead_s,
            },
        },
        "workload": {
            "request_count": len(workload),
            "base_arrival_rate_rps": base_rate,
        },
        "pipeline": summary,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
