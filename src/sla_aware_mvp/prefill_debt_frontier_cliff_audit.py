from __future__ import annotations

import json


TPOT_S = 0.150
FIXED_OVERHEAD_S = 0.005
DECODE_COMPUTE_S = 0.112

# Archived successful observations from E15/E16.
# E15 source: workflow run 32689929342, artifact exact-singleton-prefill-guard.
# E16 source: workflow run 32690886932, four multiworkload artifacts.
# Slow/Fast have identical candidate first-unsafe states for every workload, so
# each row below is a workload-level observation replicated across both pipelines.
# NOTE: E15/E16 used the earlier compute-only Prefill debt candidate. E25 audits
# the frontier mechanism of that candidate only; it is not a magnitude-validation
# experiment. E23/E26 later corrected the magnitude interpretation.
OBSERVATIONS = (
    {
        "workload": "seed7-offset0",
        "safe_intensity": 0.0131,
        "unsafe_intensity": 0.0132,
        "prefill_debt_s": 0.68192,
    },
    {
        "workload": "seed3-offset0",
        "safe_intensity": 0.0152,
        "unsafe_intensity": 0.0153,
        "prefill_debt_s": 0.66336,
    },
    {
        "workload": "seed11-offset0",
        "safe_intensity": 0.0155,
        "unsafe_intensity": 0.0156,
        "prefill_debt_s": 0.12992,
    },
    {
        "workload": "seed19-offset0",
        "safe_intensity": 0.0152,
        "unsafe_intensity": 0.0153,
        "prefill_debt_s": 0.11584,
    },
    {
        "workload": "seed7-offset200",
        "safe_intensity": 0.0087,
        "unsafe_intensity": 0.0088,
        "prefill_debt_s": 0.68192,
    },
)


def main() -> None:
    no_debt_budget_s = TPOT_S - DECODE_COMPUTE_S - FIXED_OVERHEAD_S
    rows = []
    for item in OBSERVATIONS:
        required_s = DECODE_COMPUTE_S + item["prefill_debt_s"] + FIXED_OVERHEAD_S
        rows.append(
            {
                **item,
                "num_prefill_at_first_unsafe": 1,
                "num_decode_at_first_unsafe": 1,
                "decode_compute_s": DECODE_COMPUTE_S,
                "fixed_overhead_s": FIXED_OVERHEAD_S,
                "compute_side_required_s": required_s,
                "excess_over_tpot_s": required_s - TPOT_S,
                "required_to_tpot_ratio": required_s / TPOT_S,
                "prefill_debt_to_no_debt_budget_ratio": item["prefill_debt_s"] / no_debt_budget_s,
                "replicated_pipeline_cases": 2,
            }
        )

    result = {
        "design": {
            "purpose": "audit whether the E15/E16 Prefill-debt capacity frontier is a marginal budget crossing or a discrete overlap-onset cliff",
            "evaluator_semantics_changed": False,
            "source_runs": [32689929342, 32690886932],
            "pipelines": ["helix-slow-link-placement", "helix-fast-link-placement"],
            "candidate": "E15/E16 historical candidate: exact singleton Decode + full active-Prefill compute debt",
        },
        "base_decode_budget": {
            "tpot_s": TPOT_S,
            "decode_compute_s": DECODE_COMPUTE_S,
            "fixed_overhead_s": FIXED_OVERHEAD_S,
            "remaining_before_prefill_debt_s": no_debt_budget_s,
        },
        "workload_rows": rows,
        "summary": {
            "workloads": len(rows),
            "pipeline_x_workload_first_unsafe_cases": 2 * len(rows),
            "all_first_unsafe_states_are_1prefill_1decode": True,
            "all_slow_fast_first_unsafe_states_identical": True,
            "minimum_prefill_debt_s": min(row["prefill_debt_s"] for row in rows),
            "maximum_prefill_debt_s": max(row["prefill_debt_s"] for row in rows),
            "minimum_excess_over_tpot_s": min(row["excess_over_tpot_s"] for row in rows),
            "maximum_excess_over_tpot_s": max(row["excess_over_tpot_s"] for row in rows),
            "minimum_prefill_debt_to_no_debt_budget_ratio": min(
                row["prefill_debt_to_no_debt_budget_ratio"] for row in rows
            ),
            "frontier_mechanism": "discrete Prefill+Decode overlap onset, not a near-zero residual-budget crossing",
        },
        "interpretation_guardrail": "E25 is a timing/frontier-mechanism audit, not a magnitude-validity result. Corrected E23/E26 show that profiler-only Prefill compute covers 42/44 controlled interference cases, while independently profiled overhead-inclusive Prefill blocking-service debt covers 44/44. E25 still shows that even the smaller historical compute-only debt creates a coarse overlap-onset cliff when charged immediately; charging the larger blocking-service debt immediately would not solve that timing problem. Do not fit a global discount coefficient to conflate magnitude and exposure timing.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
