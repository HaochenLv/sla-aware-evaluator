from __future__ import annotations

import json

from .prefill_debt_constant_slack_stress import E10, E11, E22_H_S_PER_TOKEN


# Exact public HELIX prompt profile anchors needed for the E25 debt range.
PROMPT_PROFILE_MS = (
    (0, 0.0),
    (250, 2.0),
    (500, 4.0),
    (750, 5.0),
    (1000, 8.0),
    (1250, 9.0),
)
NUM_LAYERS = 80
E25_COMPUTE_DEBT_S = (0.68192, 0.66336, 0.12992, 0.11584, 0.68192)
TPOT_RESIDUAL_S = 0.033


def per_layer_prompt_ms(tokens: int) -> float:
    for index, (right_x, right_y) in enumerate(PROMPT_PROFILE_MS):
        if tokens == right_x:
            return right_y
        if tokens < right_x:
            left_x, left_y = PROMPT_PROFILE_MS[index - 1]
            return left_y + (right_y - left_y) * (tokens - left_x) / (right_x - left_x)
    raise ValueError(f"tokens outside local profile range: {tokens}")


def compute_service_s(tokens: int) -> float:
    return per_layer_prompt_ms(tokens) * 0.001 * NUM_LAYERS


def blocking_service_s(tokens: int) -> float:
    return compute_service_s(tokens) + E22_H_S_PER_TOKEN * tokens


def infer_tokens_from_compute_debt(debt_s: float) -> int:
    matches = []
    for tokens in range(1, 1251):
        if abs(compute_service_s(tokens) - debt_s) <= 1e-12:
            matches.append(tokens)
    if len(matches) != 1:
        raise ValueError(f"compute debt {debt_s} has {len(matches)} token matches: {matches}")
    return matches[0]


def interpolate_alpha(tokens: int, anchors: dict[int, float]) -> float:
    xs = sorted(anchors)
    if tokens <= xs[0]:
        return anchors[xs[0]]
    if tokens >= xs[-1]:
        # Outside the E10 training range, do not extrapolate a discount.
        return 1.0
    for left, right in zip(xs, xs[1:]):
        if left <= tokens <= right:
            weight = (tokens - left) / (right - left)
            return anchors[left] + weight * (anchors[right] - anchors[left])
    raise AssertionError("unreachable")


def main() -> None:
    anchors = {64: 0.0, 256: 0.0, 512: 0.0, 1024: 0.0}
    for rows in E10.values():
        for tokens, _offset_s, excess_s in rows:
            anchors[tokens] = max(anchors[tokens], excess_s / blocking_service_s(tokens))

    heldout = []
    for pipeline, rows in E11.items():
        for tokens_tuple, excess_s in rows:
            candidate_s = sum(anchors[tokens] * blocking_service_s(tokens) for tokens in tokens_tuple)
            heldout.append(
                {
                    "pipeline": pipeline,
                    "tokens": list(tokens_tuple),
                    "observed_excess_s": excess_s,
                    "candidate_prompt_envelope_s": candidate_s,
                    "excess_to_candidate_ratio": excess_s / candidate_s,
                    "bound_holds": excess_s <= candidate_s + 1e-12,
                }
            )

    e25_rows = []
    for compute_debt_s_value in E25_COMPUTE_DEBT_S:
        tokens = infer_tokens_from_compute_debt(compute_debt_s_value)
        alpha = interpolate_alpha(tokens, anchors)
        full_service_s = blocking_service_s(tokens)
        candidate_s = alpha * full_service_s
        e25_rows.append(
            {
                "historical_compute_debt_s": compute_debt_s_value,
                "inferred_prompt_tokens": tokens,
                "compute_debt_reconstructed_s": compute_service_s(tokens),
                "blocking_service_s": full_service_s,
                "interpolated_alpha": alpha,
                "optimistic_prompt_envelope_debt_s": candidate_s,
                "ratio_to_33ms_residual": candidate_s / TPOT_RESIDUAL_S,
                "still_exceeds_33ms_residual": candidate_s > TPOT_RESIDUAL_S,
            }
        )

    result = {
        "design": {
            "purpose": "test whether a more flexible prompt-length-dependent state-only exposure envelope can transfer from one-Prefill E10 to held-out multi-Prefill E11 and materially reduce the E25 overlap cliff",
            "train": "E10 one-Prefill cases: alpha(L)=max observed excess / independently profiled blocking service at L",
            "heldout": "E11 multi-Prefill cases: additive sum alpha(L_i)*T_blocking(L_i)",
            "e25_stress": "piecewise-linear interpolation between E10 alpha anchors; alpha=1 above 1024 tokens",
            "fit_warning": "This is an intentionally optimistic fitted diagnostic, not a validated Evaluator rule.",
            "evaluator_semantics_changed": False,
            "network_redline_changed": False,
        },
        "alpha_anchors": anchors,
        "heldout_e11": {
            "cases": len(heldout),
            "violations": sum(not row["bound_holds"] for row in heldout),
            "max_excess_to_candidate_ratio": max(row["excess_to_candidate_ratio"] for row in heldout),
            "rows": heldout,
        },
        "e25": {
            "pipeline_x_workload_cases": 2 * len(e25_rows),
            "all_still_exceed_33ms": all(row["still_exceeds_33ms_residual"] for row in e25_rows),
            "minimum_ratio_to_33ms": min(row["ratio_to_33ms_residual"] for row in e25_rows),
            "rows": e25_rows,
        },
        "interpretation_guardrail": "Do not adopt the fitted alpha curve. Its value is diagnostic: even a prompt-dependent envelope trained on hidden-phase maxima and successfully transferred to the archived multi-Prefill cases does not eliminate any historical E25 overlap-onset cliff. Therefore adding fitted prompt-length discounts would increase model complexity without solving the principal coarseness problem.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
