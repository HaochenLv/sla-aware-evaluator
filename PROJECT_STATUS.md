# Project Status

## Current state
- Conservative Evaluator remains unchanged; explicit queue overhead is still zero in the current model.
- HELIX fixed-pipeline Reference stays on draft PR #6. Diagnostic PRs #8/#9/#10 are research-only; do not merge.
- HELIX is a relative execution reference, not absolute real-system ground truth.

## Experiment log
- **E7 / capacity comparison**: Conservative and HELIX both rank Fast > Slow, but Conservative estimates much higher safe request rates under the current strict SLA criterion.
- **E8 / metric semantics**: HELIX Decode Increment latency matches strong per-token TPOT; aligned TTFT remains Prefill completion rather than true first-token TTFT.
- **E9 / frontier diagnostic**: the safe→unsafe TPOT cliff is caused by a later Prefill blocking an active Decode; first unsafe points add ~0.67–0.71 s Prefill-attributed GPU queue wait.
- **E10 / one-Prefill sweep**: 32 controlled cases. Full blocker Prefill compute upper-bounds measured Decode queue in all cases; worst queue/bound ratio ~0.95.
- **E11 / multi-Prefill additivity**: 12 controlled cases with 2–3 simultaneous Prefills across Slow/Fast. Sum of full profiled Prefill compute times upper-bounds measured Decode queue in all 12 cases. Worst queue/bound ratio is 0.6752 (Fast) and 0.6789 (Slow). However, with 3×512-token Prefills, Decode layer service itself rises from ~0.1121 s isolated to ~0.1563 s. Thus queueing is not the entire compute-side interference under heavier overlap.

## Candidate model (not yet adopted)
A cleaner candidate is a single deterministic **Prefill interference debt** for active Decode:
`I_D^UB(t) = sum_{p in active Prefill} T_prefill,p^UB`.
Instead of treating this only as queue time, test whether it upper-bounds the total extra Decode compute-side delay caused by Prefill overlap. Then use
`Delta_D = tau_D - T_decode,base^UB - I_D^UB(t) - T_fix`
and leave the existing SLA-to-network red-line formulation unchanged. This is experimentally motivated, not yet mathematically proven.

## Next
Validate the interference-debt bound against total Decode excess (queue + service inflation, excluding network where separable) across the one- and multi-Prefill controlled cases. Do not modify Conservative until that bound survives.
