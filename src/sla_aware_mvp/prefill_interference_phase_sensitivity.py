from __future__ import annotations

import json


# Offline E28 audit from successful E10 HELIX artifacts (workflow 32688956807).
# No Evaluator or HELIX runtime semantics are changed.
OFFSETS_S = (0.01, 0.04, 0.08, 0.12)
ISOLATED_RAW_TPOT_S = {
    "fast": 0.11215074400322234,
    "slow": 0.11242602880402064,
}

# Measured full HELIX Prefill compute-node batch service. This is constant across
# the four timing offsets for a fixed prompt length.
MEASURED_PREFILL_SERVICE_S = {
    64: 0.04473487359999993,
    256: 0.17893949439999973,
    512: 0.3540389887999998,
    1024: 0.7080779775999996,
}

# Measured Decode GPU queue wait for one Decode victim + one later Prefill.
QUEUE_WAIT_S = {
    "fast": {
        64: (0.0, 0.0, 0.0, 0.0),
        256: (
            0.006143721596798601,
            0.03213634920479852,
            0.07213634920479856,
            0.006143721596799989,
        ),
        512: (
            0.20473418559680062,
            0.23072681361440095,
            0.27072681361440154,
            0.20473418559680256,
        ),
        1024: (
            0.6057551135968006,
            0.6317477424336008,
            0.6717477424336012,
            0.6057551135968015,
        ),
    },
    "slow": {
        64: (0.0, 0.0, 0.0, 0.0),
        256: (
            0.07606606079600042,
            0.10205868850600097,
            0.142058688506002,
            0.07606606079600431,
        ),
        512: (
            0.26809273599600186,
            0.29408536421800174,
            0.32124591461800234,
            0.26809273599600303,
        ),
        1024: (
            0.6192011840000027,
            0.6352849044420021,
            0.6752849044420022,
            0.6192011840000035,
        ),
    },
}


def monotone_non_decreasing(values: tuple[float, ...]) -> bool:
    return all(right >= left - 1e-12 for left, right in zip(values, values[1:]))


def monotone_non_increasing(values: tuple[float, ...]) -> bool:
    return all(right <= left + 1e-12 for left, right in zip(values, values[1:]))


def main() -> None:
    rows = []
    for pipeline, prompt_map in QUEUE_WAIT_S.items():
        for prompt_tokens, waits in prompt_map.items():
            service = MEASURED_PREFILL_SERVICE_S[prompt_tokens]
            peak_index = max(range(len(waits)), key=lambda index: waits[index])
            rows.append(
                {
                    "pipeline": pipeline,
                    "prompt_tokens": prompt_tokens,
                    "offsets_s": list(OFFSETS_S),
                    "decode_queue_wait_s": list(waits),
                    "measured_prefill_service_s": service,
                    "queue_to_full_service_fraction": [value / service for value in waits],
                    "monotone_non_decreasing_with_offset": monotone_non_decreasing(waits),
                    "monotone_non_increasing_with_offset": monotone_non_increasing(waits),
                    "peak_offset_s": OFFSETS_S[peak_index],
                    "wait_at_0p01_minus_0p12_abs_s": abs(waits[0] - waits[-1]),
                }
            )

    nonzero = [row for row in rows if max(row["decode_queue_wait_s"]) > 1e-12]
    reset_gap_s = OFFSETS_S[-1] - OFFSETS_S[0]
    reset_vs_decode = {
        pipeline: {
            "offset_gap_s": reset_gap_s,
            "isolated_raw_tpot_s": raw_tpot,
            "relative_difference": abs(raw_tpot - reset_gap_s) / raw_tpot,
        }
        for pipeline, raw_tpot in ISOLATED_RAW_TPOT_S.items()
    }

    result = {
        "design": {
            "purpose": "test whether Prefill-induced Decode exposure is a simple monotone function of one timing offset when full Prefill service magnitude is fixed",
            "source_workflow": 32688956807,
            "evaluator_semantics_changed": False,
            "helix_runtime_changed": False,
            "one_victim_decode_one_prefill": True,
            "offset_definition": "Prefill arrival = isolated target Prefill finish + offset",
        },
        "rows": rows,
        "summary": {
            "pipeline_prompt_groups": len(rows),
            "nonzero_interference_groups": len(nonzero),
            "nonzero_groups_not_monotone_increasing": sum(
                not row["monotone_non_decreasing_with_offset"] for row in nonzero
            ),
            "nonzero_groups_not_monotone_decreasing": sum(
                not row["monotone_non_increasing_with_offset"] for row in nonzero
            ),
            "nonzero_groups_peak_at_0p08": sum(
                abs(row["peak_offset_s"] - 0.08) <= 1e-12 for row in nonzero
            ),
            "nonzero_groups_0p01_equals_0p12_within_1e12": sum(
                row["wait_at_0p01_minus_0p12_abs_s"] <= 1e-12 for row in nonzero
            ),
            "maximum_0p01_0p12_absolute_difference_s": max(
                row["wait_at_0p01_minus_0p12_abs_s"] for row in nonzero
            ),
            "near_one_decode_cycle_reset": reset_vs_decode,
        },
        "interpretation_guardrail": "For each fixed pipeline and prompt length, full Prefill service magnitude is unchanged across offsets, yet observed Decode queue exposure varies non-monotonically. All six nonzero pipeline-prompt groups rise toward offset 0.08 and return at offset 0.12 to the same queue wait as offset 0.01 (within floating-point precision); the 0.11-s offset separation is close to one isolated Decode token interval (~0.112 s). This is evidence of execution/stage-phase sensitivity, not proof of an exact universal period. Therefore a naive monotone remaining-time or time-since-overlap heuristic is not justified by E10. A scheduler-free Conservative Evaluator may deliberately use full blocking-service debt as a worst-case envelope, accepting a coarse frontier, unless a different cheap exposure bound is independently validated.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
