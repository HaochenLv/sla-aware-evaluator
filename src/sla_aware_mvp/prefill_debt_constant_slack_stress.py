from __future__ import annotations

import json


E22_H_S_PER_TOKEN = 59.37720874470879e-6
PREFILL_COMPUTE_S = {
    64: 0.04096,
    256: 0.16384,
    512: 0.32384,
    1024: 0.64768,
}

# Archived E10 one-Prefill observations. Each tuple is
# (prompt_tokens, relative_offset_s, observed Decode queue + positive service inflation_s).
E10 = {
    "helix-fast-link-placement": (
        (64, 0.01, 0.0),
        (64, 0.04, 0.0),
        (64, 0.08, 0.0),
        (64, 0.12, 0.0),
        (256, 0.01, 0.006143721596798601),
        (256, 0.04, 0.03213634920479852),
        (256, 0.08, 0.07213634920479856),
        (256, 0.12, 0.006143721596799989),
        (512, 0.01, 0.20473418559680062),
        (512, 0.04, 0.23072681361440095),
        (512, 0.08, 0.27072681361440154),
        (512, 0.12, 0.20473418559680256),
        (1024, 0.01, 0.6057551135968006),
        (1024, 0.04, 0.6317477424336008),
        (1024, 0.08, 0.6717477424336012),
        (1024, 0.12, 0.6057551135968015),
    ),
    "helix-slow-link-placement": (
        (64, 0.01, 0.0),
        (64, 0.04, 0.0),
        (64, 0.08, 0.0),
        (64, 0.12, 0.0),
        (256, 0.01, 0.07606606079600042),
        (256, 0.04, 0.10205868850600097),
        (256, 0.08, 0.142058688506002),
        (256, 0.12, 0.07606606079600431),
        (512, 0.01, 0.26809273599600186),
        (512, 0.04, 0.29408536421800174),
        (512, 0.08, 0.32124591461800234),
        (512, 0.12, 0.26809273599600303),
        (1024, 0.01, 0.6192011840000027),
        (1024, 0.04, 0.6352849044420021),
        (1024, 0.08, 0.6752849044420022),
        (1024, 0.12, 0.6192011840000035),
    ),
}

# Archived E11 multi-Prefill observations. Each tuple is
# ((prompt_tokens, ...), observed Decode queue + positive service inflation_s).
E11 = {
    "helix-fast-link-placement": (
        ((256, 256), 0.03450378600479842),
        ((512, 512), 0.2549816872144014),
        ((256, 512), 0.23310162280480085),
        ((512, 1024), 0.6560099880144012),
        ((256, 256, 256), 0.05087859519679855),
        ((512, 512, 512), 0.3448018463968035),
    ),
    "helix-slow-link-placement": (
        ((256, 256), 0.10442612530600115),
        ((512, 512), 0.3094170688000029),
        ((256, 512), 0.29648229170600177),
        ((512, 1024), 0.6595692682180021),
        ((256, 256, 256), 0.12080093439600204),
        ((512, 512, 512), 0.36117845759600264),
    ),
}

# Historical E25 first-unsafe compute-only Prefill debts. Using compute-only here is
# intentionally optimistic: blocking-service debt is at least this large before any
# candidate slack is subtracted.
E25_FIRST_UNSAFE_COMPUTE_DEBT_S = (
    0.68192,
    0.66336,
    0.12992,
    0.11584,
    0.68192,
)
E25_NO_DEBT_TPOT_RESIDUAL_S = 0.033


def blocking_service_s(tokens: int) -> float:
    return PREFILL_COMPUTE_S[tokens] + E22_H_S_PER_TOKEN * tokens


def candidate_debt_s(tokens: tuple[int, ...], sigma_s: float) -> float:
    return sum(max(0.0, blocking_service_s(token) - sigma_s) for token in tokens)


def main() -> None:
    cases = []
    for pipeline, rows in E10.items():
        for tokens, offset_s, excess_s in rows:
            service_s = blocking_service_s(tokens)
            sigma_limit_s = service_s - excess_s
            cases.append(
                {
                    "source": "E10",
                    "pipeline": pipeline,
                    "tokens": [tokens],
                    "offset_s": offset_s,
                    "observed_excess_s": excess_s,
                    "full_blocking_service_debt_s": service_s,
                    "sigma_limit_s": sigma_limit_s,
                }
            )

    for pipeline, rows in E11.items():
        for tokens, excess_s in rows:
            full_s = sum(blocking_service_s(token) for token in tokens)
            sigma_limit_s = (full_s - excess_s) / len(tokens)
            cases.append(
                {
                    "source": "E11",
                    "pipeline": pipeline,
                    "tokens": list(tokens),
                    "offset_s": None,
                    "observed_excess_s": excess_s,
                    "full_blocking_service_debt_s": full_s,
                    "sigma_limit_s": sigma_limit_s,
                }
            )

    limiting = min(cases, key=lambda item: item["sigma_limit_s"])
    sigma_star_s = limiting["sigma_limit_s"]

    checked = []
    for item in cases:
        tokens = tuple(item["tokens"])
        debt_s = candidate_debt_s(tokens, sigma_star_s)
        ratio = item["observed_excess_s"] / debt_s if debt_s > 0 else 0.0
        checked.append({**item, "candidate_debt_at_sigma_star_s": debt_s, "excess_to_candidate_ratio": ratio})

    e25 = []
    for compute_debt_s in E25_FIRST_UNSAFE_COMPUTE_DEBT_S:
        optimistic_after_slack_s = max(0.0, compute_debt_s - sigma_star_s)
        e25.append(
            {
                "historical_compute_debt_s": compute_debt_s,
                "optimistic_compute_minus_sigma_star_s": optimistic_after_slack_s,
                "ratio_to_33ms_residual": optimistic_after_slack_s / E25_NO_DEBT_TPOT_RESIDUAL_S,
                "still_exceeds_33ms_residual": optimistic_after_slack_s > E25_NO_DEBT_TPOT_RESIDUAL_S,
            }
        )

    result = {
        "design": {
            "purpose": "stress-test whether a single scheduler-free constant slack can materially reduce Prefill blocking-service debt while preserving all archived E10/E11 bounds",
            "candidate_family": "I_sigma=sum(max(0, T_prefill_blocking_service-sigma))",
            "blocking_service_input": "E22 independently calibrated overhead + HELIX profiler compute",
            "evaluator_semantics_changed": False,
            "network_redline_changed": False,
            "fit_warning": "sigma_star is an empirical upper limit from archived cases, not a validated model parameter",
        },
        "summary": {
            "controlled_cases": len(cases),
            "largest_archived_safe_constant_slack_s": sigma_star_s,
            "largest_archived_safe_constant_slack_ms": 1000.0 * sigma_star_s,
            "limiting_case": limiting,
            "violations_at_sigma_star": sum(
                item["observed_excess_s"] > item["candidate_debt_at_sigma_star_s"] + 1e-12
                for item in checked
            ),
            "max_excess_to_candidate_ratio_at_sigma_star": max(
                item["excess_to_candidate_ratio"] for item in checked
            ),
            "e25_first_unsafe_cases": len(e25) * 2,
            "e25_all_still_cliffs_even_under_optimistic_compute_minus_slack": all(
                item["still_exceeds_33ms_residual"] for item in e25
            ),
            "e25_min_ratio_to_33ms_residual": min(item["ratio_to_33ms_residual"] for item in e25),
        },
        "e25_workload_rows": e25,
        "interpretation_guardrail": "Do not adopt sigma_star. It is fitted to archived HELIX observations and unseen hidden phases may require less or zero slack. The useful result is negative: even granting the largest constant slack compatible with all archived E10/E11 cases does not remove any historical E25 overlap-onset cliff. Therefore a global constant slack is neither well-validated nor sufficient to solve the coarse frontier.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
