from __future__ import annotations

import json


# E26 is an offline cross-check only. No Evaluator/HELIX semantics are changed.
# E10/E11 observations come from successful HELIX runs 32688956807 and
# 32689148345. E22 overhead calibration comes from the independent E20 threshold
# experiment: h_P midpoint = 59.37720874470879 us/token. That coefficient was
# inferred from measured threshold brackets, not from the E10/E11 interference
# observations below.
BASELINE_DECODE_LAYER_SERVICE_S = 0.11205898239999991
E22_OVERHEAD_S_PER_TOKEN = 59.37720874470879e-6

# HELIX public prompt profile summed over the fixed 80-layer pipeline.
# These values follow prompt_bs2time.csv at pinned HELIX commit 8639497... .
PROFILED_PREFILL_COMPUTE_S = {
    64: 0.04096,
    256: 0.16384,
    512: 0.32384,
    1024: 0.64768,
}

# Independent E10 tracer observation: total HELIX compute-node batch service
# containing the one Prefill blocker. This was not an input to E22 calibration.
E10_MEASURED_PREFILL_SERVICE_S = {
    64: 0.04473487359999993,
    256: 0.17893949439999973,
    512: 0.3540389887999998,
    1024: 0.7080779775999996,
}

# pipeline, prompt_tokens, decode-relative arrival offset, measured Decode queue.
# E10 has no Decode service inflation in these controlled one-Prefill cases.
E10 = (
    ("fast", 64, 0.01, 0.0),
    ("fast", 64, 0.04, 0.0),
    ("fast", 64, 0.08, 0.0),
    ("fast", 64, 0.12, 0.0),
    ("fast", 256, 0.01, 0.006143721596798601),
    ("fast", 256, 0.04, 0.03213634920479852),
    ("fast", 256, 0.08, 0.07213634920479856),
    ("fast", 256, 0.12, 0.006143721596799989),
    ("fast", 512, 0.01, 0.20473418559680062),
    ("fast", 512, 0.04, 0.23072681361440095),
    ("fast", 512, 0.08, 0.27072681361440154),
    ("fast", 512, 0.12, 0.20473418559680256),
    ("fast", 1024, 0.01, 0.6057551135968006),
    ("fast", 1024, 0.04, 0.6317477424336008),
    ("fast", 1024, 0.08, 0.6717477424336012),
    ("fast", 1024, 0.12, 0.6057551135968015),
    ("slow", 64, 0.01, 0.0),
    ("slow", 64, 0.04, 0.0),
    ("slow", 64, 0.08, 0.0),
    ("slow", 64, 0.12, 0.0),
    ("slow", 256, 0.01, 0.07606606079600042),
    ("slow", 256, 0.04, 0.10205868850600097),
    ("slow", 256, 0.08, 0.142058688506002),
    ("slow", 256, 0.12, 0.07606606079600431),
    ("slow", 512, 0.01, 0.26809273599600186),
    ("slow", 512, 0.04, 0.29408536421800174),
    ("slow", 512, 0.08, 0.32124591461800234),
    ("slow", 512, 0.12, 0.26809273599600303),
    ("slow", 1024, 0.01, 0.6192011840000027),
    ("slow", 1024, 0.04, 0.6352849044420021),
    ("slow", 1024, 0.08, 0.6752849044420022),
    ("slow", 1024, 0.12, 0.6192011840000035),
)

# pipeline, active Prefill token tuple, queue wait, Decode layer service.
E11 = (
    ("fast", (256, 256), 0.03450378600479842, 0.11205898239999991),
    ("fast", (512, 512), 0.2549816872144014, 0.11205898239999991),
    ("fast", (256, 512), 0.23307213160480084, 0.11208847359999992),
    ("fast", (512, 1024), 0.6559804968144012, 0.11208847359999992),
    ("fast", (256, 256, 256), 0.05087859519679855, 0.11205898239999991),
    ("fast", (512, 512, 512), 0.3005322271968032, 0.1563286016000002),
    ("slow", (256, 256), 0.10442612530600115, 0.11205898239999991),
    ("slow", (512, 512), 0.3094170688000029, 0.11205898239999991),
    ("slow", (256, 512), 0.29645280050600176, 0.11208847359999992),
    ("slow", (512, 1024), 0.659539777018002, 0.11208847359999992),
    ("slow", (256, 256, 256), 0.12080093439600204, 0.11205898239999991),
    ("slow", (512, 512, 512), 0.31691621119600233, 0.1563212288000002),
)


def debt(tokens: tuple[int, ...], *, include_overhead: bool) -> float:
    compute = sum(PROFILED_PREFILL_COMPUTE_S[item] for item in tokens)
    if not include_overhead:
        return compute
    return compute + E22_OVERHEAD_S_PER_TOKEN * sum(tokens)


def analyze_row(*, pipeline: str, label: str, tokens: tuple[int, ...], excess_s: float) -> dict:
    compute_debt = debt(tokens, include_overhead=False)
    service_debt = debt(tokens, include_overhead=True)
    return {
        "pipeline": pipeline,
        "case": label,
        "prefill_tokens": list(tokens),
        "total_decode_compute_side_excess_s": excess_s,
        "profiled_prefill_compute_debt_s": compute_debt,
        "profiled_prefill_overhead_s": service_debt - compute_debt,
        "profiled_prefill_service_debt_s": service_debt,
        "excess_to_compute_debt_ratio": excess_s / compute_debt,
        "excess_to_service_debt_ratio": excess_s / service_debt,
        "compute_only_bound_holds": excess_s <= compute_debt + 1e-12,
        "service_debt_bound_holds": excess_s <= service_debt + 1e-12,
    }


def main() -> None:
    rows = []
    for pipeline, tokens, offset, queue_wait in E10:
        rows.append(
            analyze_row(
                pipeline=pipeline,
                label=f"E10-{tokens}@{offset:.2f}",
                tokens=(tokens,),
                excess_s=queue_wait,
            )
        )
    for pipeline, tokens, queue_wait, decode_service in E11:
        inflation = max(0.0, decode_service - BASELINE_DECODE_LAYER_SERVICE_S)
        rows.append(
            analyze_row(
                pipeline=pipeline,
                label="E11-" + "+".join(str(item) for item in tokens),
                tokens=tokens,
                excess_s=queue_wait + inflation,
            )
        )

    compute_violations = [row for row in rows if not row["compute_only_bound_holds"]]
    service_violations = [row for row in rows if not row["service_debt_bound_holds"]]
    worst_compute = max(rows, key=lambda row: row["excess_to_compute_debt_ratio"])
    worst_service = max(rows, key=lambda row: row["excess_to_service_debt_ratio"])
    e10_rows = [row for row in rows if row["case"].startswith("E10-")]
    e11_rows = [row for row in rows if row["case"].startswith("E11-")]

    service_transfer = []
    for tokens, measured in E10_MEASURED_PREFILL_SERVICE_S.items():
        predicted = debt((tokens,), include_overhead=True)
        service_transfer.append(
            {
                "prompt_tokens": tokens,
                "profiled_compute_s": PROFILED_PREFILL_COMPUTE_S[tokens],
                "e22_predicted_overhead_s": E22_OVERHEAD_S_PER_TOKEN * tokens,
                "predicted_prefill_service_s": predicted,
                "e10_measured_prefill_service_s": measured,
                "absolute_error_s": predicted - measured,
                "relative_error": (predicted - measured) / measured,
            }
        )

    result = {
        "design": {
            "purpose": "test whether a profiled Prefill service debt (compute + independently calibrated runtime overhead) covers HELIX Prefill-induced Decode compute-side excess",
            "evaluator_semantics_changed": False,
            "network_redline_changed": False,
            "observation_runs": [32688956807, 32689148345],
            "overhead_calibration_source": "E22 / E20 isolated aligned-TTFT thresholds",
            "overhead_midpoint_us_per_token": E22_OVERHEAD_S_PER_TOKEN * 1e6,
            "overhead_fitted_to_interference_cases": False,
        },
        "independent_prefill_service_transfer": {
            "cases": service_transfer,
            "max_abs_relative_error": max(abs(row["relative_error"]) for row in service_transfer),
        },
        "compute_only": {
            "cases": len(rows),
            "bound_violations": len(compute_violations),
            "worst_ratio": worst_compute["excess_to_compute_debt_ratio"],
            "worst_case": {"pipeline": worst_compute["pipeline"], "case": worst_compute["case"]},
            "violations": compute_violations,
        },
        "compute_plus_profiled_overhead": {
            "cases": len(rows),
            "bound_violations": len(service_violations),
            "worst_ratio": worst_service["excess_to_service_debt_ratio"],
            "worst_case": {"pipeline": worst_service["pipeline"], "case": worst_service["case"]},
            "unit_service_debt_covers_all_observed_cases": len(service_violations) == 0,
        },
        "by_experiment": {
            "e10_cases": len(e10_rows),
            "e10_compute_only_violations": sum(not row["compute_only_bound_holds"] for row in e10_rows),
            "e10_service_debt_violations": sum(not row["service_debt_bound_holds"] for row in e10_rows),
            "e10_max_service_ratio": max(row["excess_to_service_debt_ratio"] for row in e10_rows),
            "e11_cases": len(e11_rows),
            "e11_compute_only_violations": sum(not row["compute_only_bound_holds"] for row in e11_rows),
            "e11_service_debt_violations": sum(not row["service_debt_bound_holds"] for row in e11_rows),
            "e11_max_service_ratio": max(row["excess_to_service_debt_ratio"] for row in e11_rows),
        },
        "interpretation_guardrail": "This is cross-experiment consistency evidence, not a universal proof. The E22 overhead coefficient was inferred independently from isolated Prefill threshold measurements and was not fitted to E10/E11 queue/interference observations. It also predicts the independently observed E10 Prefill compute-node service to within about 0.06%, strengthening the profiling/service-time interpretation. The result supports moving Prefill runtime overhead into the profiling/service-debt interface rather than modifying the network red-line or fitting a HELIX-specific interference coefficient.",
        "rows": rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
