from __future__ import annotations

import json


TTFT_S = 2.0
FIXED_OVERHEAD_S = 0.005
ACTIVATION_BYTES_PER_TOKEN = 16384
HELIX_TOKEN_BYTES = 2
INTER_STAGE_LINKS = 7
STAGES = 8
HELIX_CPU_CONCAT_BYTES_PER_S = 4e9
HELIX_CPU_TRANSFER_BYTES_PER_S = 5e9

# E20 direct HELIX aligned-TTFT threshold brackets. These are measured observations,
# not model inputs. Each tuple is (prompt_tokens, prefill_compute_s,
# unsafe_lower_gbps, safe_upper_gbps).
OBSERVATIONS = (
    (512, 0.3238400000000001, 0.2828125, 0.28662109375),
    (1024, 0.6476800000000001, 0.7284179687500001, 0.7322265625000002),
    (1536, 0.89152, 1.39111328125, 1.394921875),
    (1821, 1.00544, 1.8938476562499997, 1.8976562499999998),
)


def serial_network_time_s(prompt_tokens: int, bandwidth_gbps: float) -> float:
    payload_bytes = (ACTIVATION_BYTES_PER_TOKEN + HELIX_TOKEN_BYTES) * prompt_tokens
    bytes_per_s = bandwidth_gbps * 1e9 / 8.0
    return INTER_STAGE_LINKS * payload_bytes / bytes_per_s


def overhead_interval_s_per_token(
    prompt_tokens: int,
    prefill_compute_s: float,
    unsafe_lower_gbps: float,
    safe_upper_gbps: float,
) -> tuple[float, float]:
    lower = (
        TTFT_S
        - FIXED_OVERHEAD_S
        - prefill_compute_s
        - serial_network_time_s(prompt_tokens, unsafe_lower_gbps)
    ) / prompt_tokens
    upper = (
        TTFT_S
        - FIXED_OVERHEAD_S
        - prefill_compute_s
        - serial_network_time_s(prompt_tokens, safe_upper_gbps)
    ) / prompt_tokens
    return lower, upper


def threshold_from_linear_overhead(
    prompt_tokens: int,
    prefill_compute_s: float,
    overhead_s_per_token: float,
) -> float:
    remaining_s = (
        TTFT_S
        - FIXED_OVERHEAD_S
        - prefill_compute_s
        - overhead_s_per_token * prompt_tokens
    )
    payload_bytes = (ACTIVATION_BYTES_PER_TOKEN + HELIX_TOKEN_BYTES) * prompt_tokens
    coefficient_s_gbps = INTER_STAGE_LINKS * payload_bytes / (1e9 / 8.0)
    return coefficient_s_gbps / remaining_s


def main() -> None:
    rows = []
    intervals = []
    for prompt, compute_s, unsafe, safe in OBSERVATIONS:
        lower, upper = overhead_interval_s_per_token(prompt, compute_s, unsafe, safe)
        intervals.append((lower, upper))
        rows.append(
            {
                "prompt_tokens": prompt,
                "prefill_compute_s": compute_s,
                "helix_threshold_bracket_gbps": [unsafe, safe],
                "overhead_interval_us_per_token": [lower * 1e6, upper * 1e6],
            }
        )

    intersection_lower = max(lower for lower, _ in intervals)
    intersection_upper = min(upper for _, upper in intervals)
    intersection_mid = (intersection_lower + intersection_upper) / 2.0

    source_coefficient = (
        STAGES
        * ACTIVATION_BYTES_PER_TOKEN
        * (1.0 / HELIX_CPU_CONCAT_BYTES_PER_S + 1.0 / HELIX_CPU_TRANSFER_BYTES_PER_S)
    )

    all_prompt_predictions = []
    for prompt, compute_s, unsafe, safe in OBSERVATIONS:
        prediction = threshold_from_linear_overhead(prompt, compute_s, intersection_mid)
        all_prompt_predictions.append(
            {
                "prompt_tokens": prompt,
                "predicted_threshold_gbps": prediction,
                "inside_observed_bracket": unsafe < prediction <= safe,
            }
        )

    loo = []
    for held_out_index, (prompt, compute_s, unsafe, safe) in enumerate(OBSERVATIONS):
        train_intervals = [
            interval for index, interval in enumerate(intervals) if index != held_out_index
        ]
        train_lower = max(lower for lower, _ in train_intervals)
        train_upper = min(upper for _, upper in train_intervals)
        train_mid = (train_lower + train_upper) / 2.0
        prediction = threshold_from_linear_overhead(prompt, compute_s, train_mid)
        loo.append(
            {
                "held_out_prompt_tokens": prompt,
                "train_intersection_us_per_token": [train_lower * 1e6, train_upper * 1e6],
                "predicted_threshold_gbps": prediction,
                "observed_bracket_gbps": [unsafe, safe],
                "inside_observed_bracket": unsafe < prediction <= safe,
            }
        )

    result = {
        "design": {
            "purpose": "infer a generic linear Prefill runtime-overhead term from E20 isolated HELIX thresholds without changing the network red-line equation",
            "evaluator_semantics_changed": False,
            "reference_metric": "HELIX aligned Prefill-completion TTFT",
            "assumed_overhead_form": "T_ovhd^P(L)=h_P*L",
            "network_term": "seven serial stage-boundary transfers using HELIX activation+token payload",
        },
        "per_prompt_constraints": rows,
        "joint_overhead_interval_us_per_token": [intersection_lower * 1e6, intersection_upper * 1e6],
        "joint_midpoint_us_per_token": intersection_mid * 1e6,
        "helix_source_cpu_buffer_coefficient_us_per_token": source_coefficient * 1e6,
        "source_coefficient_inside_joint_interval": intersection_lower <= source_coefficient <= intersection_upper,
        "joint_midpoint_predictions": all_prompt_predictions,
        "leave_one_out": loo,
        "all_joint_predictions_inside": all(item["inside_observed_bracket"] for item in all_prompt_predictions),
        "all_leave_one_out_predictions_inside": all(item["inside_observed_bracket"] for item in loo),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
