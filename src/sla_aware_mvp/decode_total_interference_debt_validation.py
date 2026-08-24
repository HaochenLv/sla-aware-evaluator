from __future__ import annotations

import json
from dataclasses import dataclass


# E11 measured HELIX observations recovered from successful workflow run 32689148345.
# E10 aggregate extrema below are recovered from successful run 32688956807.
# This is an offline analysis only: no HELIX runtime or Conservative Evaluator
# semantics are changed.
BASELINE_DECODE_LAYER_SERVICE_S = 0.11205898239999991
MATERIAL_SERVICE_INFLATION_S = 0.001
E10_CASES = 32
E10_BOUND_VIOLATIONS = 0
E10_MAX_RATIO_FAST = 0.9486917595014914
E10_MAX_RATIO_SLOW = 0.9536872008515953


@dataclass(frozen=True)
class Case:
    pipeline: str
    label: str
    queue_wait_s: float
    decode_layer_service_s: float
    full_prefill_debt_ub_s: float


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
    ratio = total_compute_excess_s / case.full_prefill_debt_ub_s
    return {
        "pipeline": case.pipeline,
        "case": case.label,
        "queue_wait_s": case.queue_wait_s,
        "decode_layer_service_s": case.decode_layer_service_s,
        "baseline_decode_layer_service_s": BASELINE_DECODE_LAYER_SERVICE_S,
        "decode_service_inflation_s": service_inflation_s,
        "material_decode_service_inflation": service_inflation_s > MATERIAL_SERVICE_INFLATION_S,
        "total_decode_compute_excess_s": total_compute_excess_s,
        "full_active_prefill_debt_ub_s": case.full_prefill_debt_ub_s,
        "excess_to_debt_ratio": ratio,
        "bound_holds": total_compute_excess_s <= case.full_prefill_debt_ub_s + 1e-12,
    }


def main() -> None:
    rows = [analyze(case) for case in CASES]
    worst_e11 = max(rows, key=lambda row: row["excess_to_debt_ratio"])
    material_inflation = [row for row in rows if row["material_decode_service_inflation"]]
    e10_worst = max(E10_MAX_RATIO_FAST, E10_MAX_RATIO_SLOW)
    combined_worst = max(e10_worst, worst_e11["excess_to_debt_ratio"])
    result = {
        "design": {
            "purpose": "test whether full active-Prefill compute debt upper-bounds total Decode compute-side excess, not only queue wait",
            "observation_sources": [
                "E10 successful HELIX one-Prefill blocking sweep run 32688956807",
                "E11 successful HELIX multi-Prefill additivity run 32689148345",
            ],
            "evaluator_semantics_changed": False,
            "new_simulator_or_scheduler": False,
            "total_excess_definition": "GPU queue wait + max(0, Decode layer service - isolated Decode layer service)",
            "candidate_debt": "sum(full profiled pipeline compute time of active Prefills)",
            "material_service_inflation_threshold_s": MATERIAL_SERVICE_INFLATION_S,
            "network_excluded": True,
        },
        "e10_one_prefill": {
            "cases": E10_CASES,
            "bound_violations": E10_BOUND_VIOLATIONS,
            "decode_service_inflation_observed": False,
            "max_excess_to_debt_ratio_fast": E10_MAX_RATIO_FAST,
            "max_excess_to_debt_ratio_slow": E10_MAX_RATIO_SLOW,
        },
        "e11_multi_prefill": {
            "cases": len(rows),
            "bound_violations": sum(not row["bound_holds"] for row in rows),
            "material_service_inflation_cases": len(material_inflation),
            "max_excess_to_debt_ratio": worst_e11["excess_to_debt_ratio"],
            "worst_case": {
                "pipeline": worst_e11["pipeline"],
                "case": worst_e11["case"],
            },
            "all_bounds_hold": all(row["bound_holds"] for row in rows),
        },
        "combined": {
            "cases": E10_CASES + len(rows),
            "bound_violations": E10_BOUND_VIOLATIONS + sum(not row["bound_holds"] for row in rows),
            "max_observed_excess_to_debt_ratio": combined_worst,
            "headroom_of_unit_debt_at_worst_case": 1.0 - combined_worst,
            "minimum_empirical_multiplier_to_cover_observed_cases": combined_worst,
            "unit_multiplier_covers_all_observed_cases": combined_worst <= 1.0,
        },
        "material_service_inflation_cases": material_inflation,
        "e11_cases": rows,
        "interpretation_guardrail": "The observed minimum multiplier is not a proof-valid calibration target. The exact E10 worst case already reaches 0.953687 of full Prefill debt, leaving only 4.63% empirical headroom at multiplier 1.0. Reducing the debt coefficient below 1 would weaken the intended conservative interpretation and is not supported by these experiments.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
