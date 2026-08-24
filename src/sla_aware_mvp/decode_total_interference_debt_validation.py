from __future__ import annotations

import json
from dataclasses import dataclass


# Corrected E23 audit.
# E11 observations are explicit below and use the candidate's profiled Prefill
# compute debt. E10 originally copied ratios from the E10 artifact field
# `interference_to_prefill_service_ratio`, whose denominator is HELIX *measured
# blocker batch service*, not the profiler-only compute debt. Rechecking the E10
# raw observations against the actual HELIX prompt profile shows two compute-only
# violations (1024-token blocker, offset 0.08, Slow/Fast).
BASELINE_DECODE_LAYER_SERVICE_S = 0.11205898239999991
MATERIAL_SERVICE_INFLATION_S = 0.001
E10_CASES = 32
E10_COMPUTE_ONLY_BOUND_VIOLATIONS = 2
E10_MAX_COMPUTE_ONLY_RATIO_FAST = 1.0371599284115631
E10_MAX_COMPUTE_ONLY_RATIO_SLOW = 1.0426212086863917
# Retain the original artifact ratios only under their correct denominator name.
E10_MAX_MEASURED_PREFILL_SERVICE_RATIO_FAST = 0.9486917595014914
E10_MAX_MEASURED_PREFILL_SERVICE_RATIO_SLOW = 0.9536872008515953


@dataclass(frozen=True)
class Case:
    pipeline: str
    label: str
    queue_wait_s: float
    decode_layer_service_s: float
    full_profiled_prefill_compute_debt_s: float


CASES = (
    Case("slow", "2x256", 0.10442612530600115, 0.11205898239999991, 0.32767999999999997),
    Case("slow", "2x512", 0.3094170688000029, 0.11205898239999991, 0.6476800000000001),
    Case("slow", "256+512", 0.29645280050600176, 0.11208847359999992, 0.48768000000000006),
    Case("slow", "512+1024", 0.659539777018002, 0.11208847359999992, 0.9715200000000002),
    Case("slow", "3x256", 0.12080093439600204, 0.11205898239999991, 0.49151999999999996),
    Case("slow", "3x512", 0.31691621119600233, 0.1563212288000002, 0.9715200000000002),
    Case("fast", "2x256", 0.03450378600479842, 0.11205898239999991, 0.32767999999999997),
    Case("fast", "2x512", 0.2549816872144014, 0.11205898239999991, 0.6476800000000001),
    Case("fast", "256+512", 0.23307213160480084, 0.11208847359999992, 0.48768000000000006),
    Case("fast", "512+1024", 0.6559804968144012, 0.11208847359999992, 0.9715200000000002),
    Case("fast", "3x256", 0.05087859519679855, 0.11205898239999991, 0.49151999999999996),
    Case("fast", "3x512", 0.3005322271968032, 0.1563286016000002, 0.9715200000000002),
)


def analyze(case: Case) -> dict:
    service_inflation_s = max(
        0.0, case.decode_layer_service_s - BASELINE_DECODE_LAYER_SERVICE_S
    )
    total_compute_excess_s = case.queue_wait_s + service_inflation_s
    ratio = total_compute_excess_s / case.full_profiled_prefill_compute_debt_s
    return {
        "pipeline": case.pipeline,
        "case": case.label,
        "queue_wait_s": case.queue_wait_s,
        "decode_layer_service_s": case.decode_layer_service_s,
        "baseline_decode_layer_service_s": BASELINE_DECODE_LAYER_SERVICE_S,
        "decode_service_inflation_s": service_inflation_s,
        "material_decode_service_inflation": service_inflation_s > MATERIAL_SERVICE_INFLATION_S,
        "total_decode_compute_excess_s": total_compute_excess_s,
        "full_profiled_prefill_compute_debt_s": case.full_profiled_prefill_compute_debt_s,
        "excess_to_compute_debt_ratio": ratio,
        "compute_only_bound_holds": total_compute_excess_s <= case.full_profiled_prefill_compute_debt_s + 1e-12,
    }


def main() -> None:
    rows = [analyze(case) for case in CASES]
    worst_e11 = max(rows, key=lambda row: row["excess_to_compute_debt_ratio"])
    material_inflation = [row for row in rows if row["material_decode_service_inflation"]]
    e10_worst_compute = max(
        E10_MAX_COMPUTE_ONLY_RATIO_FAST, E10_MAX_COMPUTE_ONLY_RATIO_SLOW
    )
    combined_worst_compute = max(
        e10_worst_compute, worst_e11["excess_to_compute_debt_ratio"]
    )
    e11_violations = sum(not row["compute_only_bound_holds"] for row in rows)
    result = {
        "design": {
            "purpose": "correct the E23 denominator audit and test profiler-only Prefill compute debt against total Decode compute-side excess",
            "observation_sources": [
                "E10 successful HELIX one-Prefill blocking sweep run 32688956807",
                "E11 successful HELIX multi-Prefill additivity run 32689148345",
            ],
            "evaluator_semantics_changed": False,
            "new_simulator_or_scheduler": False,
            "total_excess_definition": "GPU queue wait + max(0, Decode layer service - isolated Decode layer service)",
            "candidate_debt": "sum(full HELIX-profiler Prefill compute time)",
            "network_excluded": True,
        },
        "correction": {
            "previous_issue": "E10 ratios 0.948692/0.953687 used HELIX measured Prefill batch service as denominator but were mislabeled as profiler-only compute debt",
            "consequence": "the previous 44/44 compute-only coverage claim is invalid",
        },
        "e10_one_prefill": {
            "cases": E10_CASES,
            "compute_only_bound_violations": E10_COMPUTE_ONLY_BOUND_VIOLATIONS,
            "max_excess_to_compute_debt_ratio_fast": E10_MAX_COMPUTE_ONLY_RATIO_FAST,
            "max_excess_to_compute_debt_ratio_slow": E10_MAX_COMPUTE_ONLY_RATIO_SLOW,
            "max_excess_to_measured_prefill_service_ratio_fast": E10_MAX_MEASURED_PREFILL_SERVICE_RATIO_FAST,
            "max_excess_to_measured_prefill_service_ratio_slow": E10_MAX_MEASURED_PREFILL_SERVICE_RATIO_SLOW,
            "violating_mechanism": "1024-token blocker at decode-relative offset 0.08 on each pipeline",
        },
        "e11_multi_prefill": {
            "cases": len(rows),
            "compute_only_bound_violations": e11_violations,
            "material_service_inflation_cases": len(material_inflation),
            "max_excess_to_compute_debt_ratio": worst_e11["excess_to_compute_debt_ratio"],
            "worst_case": {
                "pipeline": worst_e11["pipeline"],
                "case": worst_e11["case"],
            },
            "all_compute_only_bounds_hold": e11_violations == 0,
        },
        "combined_compute_only": {
            "cases": E10_CASES + len(rows),
            "bound_violations": E10_COMPUTE_ONLY_BOUND_VIOLATIONS + e11_violations,
            "max_observed_excess_to_compute_debt_ratio": combined_worst_compute,
            "unit_compute_debt_covers_all_observed_cases": combined_worst_compute <= 1.0,
        },
        "material_service_inflation_cases": material_inflation,
        "e11_cases": rows,
        "next_question": "E20/E22 identified a profiled Prefill runtime-overhead term. Test service debt = Prefill compute profile + profiled overhead before rejecting the simple debt abstraction.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
