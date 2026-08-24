# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#18 are research-only; do not merge.
- Best compute-side candidate remains exact HELIX singleton Decode profile alignment + full active-Prefill compute debt. Network red-line equations are still unchanged.

## Experiment log
- **E15–E16 / Prefill guard**: exact singleton Decode + active-Prefill debt stays conservative across all tested workload variants; no candidate optimism was observed, but compute-driven progress ties bandwidth-only Slow/Fast pipelines.
- **E17 / bandwidth-only ordering**: across six Azure trace offsets, 10 Gbps never has a lower HELIX capacity frontier than 2.5 Gbps. The resolved capacity gain is small (~0.6–1.5%), so fine ranking is a secondary phase-timing effect.
- **E18 / 1.25–20 Gbps sweep**: on both representative 17-request and 50-request windows, 2.5→20 Gbps changes HELIX safe request-rate capacity by only ~1.3–1.5%; 5 and 10 Gbps share the same capacity bracket. Bandwidth strongly reduces TTFT, while the multi-request capacity frontier quickly becomes TPOT/interference dominated. Offset-0 has no positive safe capacity at 1.25 Gbps.
- **E19 / isolated network-feasibility threshold**: replaying exactly one longest-prompt request removes queueing/overlap. For the 1709-token prompt, 1.6 Gbps is aligned-TTFT unsafe (2.0328 s) and 1.7 Gbps is safe (1.9752 s), giving a threshold in (1.6, 1.7] Gbps. For the 1821-token prompt, 1.8 Gbps is unsafe (2.0462 s) and 2.0 Gbps is safe (1.9533 s), giving a threshold in (1.8, 2.0] Gbps. In all these probes TPOT stays near 117.5–117.6 ms; the below-threshold failure is TTFT-only.

## Interpretation
With one isolated request, HELIX aligned TTFT follows `T(B)=a+c/B` to numerical precision: the fitted 2 s thresholds are about 1.655 Gbps for 1709 prompt tokens and 1.894 Gbps for 1821 prompt tokens. This supports treating network as a simple per-request SLA feasibility/red-line term rather than recreating HELIX network event timing inside the cheap Evaluator. Once this network feasibility threshold is crossed, E18 shows that request-rate capacity is mainly controlled by Decode/Prefill interference in the tested regime. Standard true first-token TTFT remains a separate semantic issue and gives a stricter threshold than the current aligned metric.

## Next
Do not integrate E15 yet and do not add a scheduler. Directly validate the Evaluator's existing network-red-line formula against HELIX isolated feasibility thresholds across several prompt lengths (e.g. 512/1024/1536/1821 tokens) on the same fixed Pipeline. Compare predicted minimum bandwidth versus HELIX threshold before changing any network equation.
