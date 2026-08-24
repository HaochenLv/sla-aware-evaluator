from __future__ import annotations

import json

from .prefill_interference_phase_sensitivity import OFFSETS_S, QUEUE_WAIT_S
from .prefill_service_debt_validation import debt


def main() -> None:
    rows = []
    for pipeline, prompt_map in QUEUE_WAIT_S.items():
        for prompt_tokens, waits in prompt_map.items():
            if max(waits) <= 1e-12:
                continue
            service_debt_s = debt((prompt_tokens,), include_overhead=True)
            fractions = [wait / service_debt_s for wait in waits]
            worst_index = max(range(len(waits)), key=lambda index: waits[index])
            rows.append(
                {
                    "observable_macro_state": {
                        "pipeline": pipeline,
                        "num_prefill": 1,
                        "num_decode": 1,
                        "prefill_prompt_tokens": prompt_tokens,
                        "prefill_blocking_service_debt_s": service_debt_s,
                    },
                    "hidden_timing_offsets_s": list(OFFSETS_S),
                    "observed_decode_queue_wait_s": list(waits),
                    "observed_exposure_fraction_of_service_debt": fractions,
                    "minimum_state_only_multiplier_covering_observed_offsets": max(fractions),
                    "headroom_at_unit_multiplier": 1.0 - max(fractions),
                    "worst_observed_offset_s": OFFSETS_S[worst_index],
                }
            )

    global_worst = max(
        rows,
        key=lambda row: row["minimum_state_only_multiplier_covering_observed_offsets"],
    )
    global_multiplier = global_worst[
        "minimum_state_only_multiplier_covering_observed_offsets"
    ]

    result = {
        "design": {
            "purpose": "quantify how much a scheduler-free macro-state-only Prefill exposure bound can be tightened before it contradicts observed hidden timing phases",
            "source": "E10 archived one-Decode/one-Prefill HELIX timing sweep + E22/E26 independently profiled Prefill blocking-service debt",
            "evaluator_semantics_changed": False,
            "hidden_variable": "relative execution/stage phase represented by the E10 arrival offset sweep",
            "observable_state_kept_fixed_within_each_row": "pipeline, N_P=1, N_D=1, Prefill prompt length, and full blocking-service debt magnitude",
        },
        "rows": rows,
        "summary": {
            "nonzero_macro_states": len(rows),
            "global_minimum_empirical_state_only_multiplier": global_multiplier,
            "global_unit_multiplier_headroom": 1.0 - global_multiplier,
            "tightest_macro_state": global_worst["observable_macro_state"],
            "tightest_hidden_offset_s": global_worst["worst_observed_offset_s"],
        },
        "interpretation_guardrail": "This is an observability audit, not a proof-calibrated multiplier. Under the coarse macro-state used by the scheduler-free Evaluator, different hidden execution phases produce different actual Decode interference while the visible Prefill magnitude is unchanged. Across the observed E10 phases, a state-only multiplier below the global empirical maximum would already fail a controlled case. The tightest observed service-debt fraction is about 0.953143, leaving only about 4.69% empirical headroom at multiplier 1.0. Do not set alpha=0.953143: unseen phases/workloads may be worse. Materially tighter exposure would require additional phase/stage/scheduler state or a different independently validated upper bound. If that additional state recreates serving scheduling, keep the full blocking-service debt as the conservative envelope instead.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
