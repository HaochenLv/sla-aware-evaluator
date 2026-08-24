# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- E23 is a research-only total Decode compute-side excess audit. Network red-line equations are unchanged.
- Important correction: profiler-only Prefill **compute** debt and HELIX measured Prefill **service** time must not be conflated.

## Experiment log
- **E10 / one-Prefill sweep**: 32 controlled Slow/Fast cases. The original E10 artifact reports `interference_to_prefill_service_ratio`, where the denominator is HELIX measured blocker batch service. The old E23 status incorrectly relabeled those ratios as profiler-only Prefill compute debt.
- **E23 corrected E10 audit**: against the actual HELIX prompt compute profile, compute-only debt fails in 2/32 cases: the 1024-token blocker at relative offset 0.08 on Fast and Slow. Total Decode compute-side excess / profiler-only debt reaches `1.037160` Fast and `1.042621` Slow. Thus the old `44/44` compute-only claim is invalid.
- **E11 / multi-Prefill total excess**: for the 12 explicit 2–3 Prefill cases, profiler-only summed Prefill compute still upper-bounds `queue wait + positive Decode service inflation` in all 12/12. Worst ratio is `0.678904` (Slow, 512+1024); the 3x512 material-service-inflation ratios remain `0.371766` Slow and `0.354910` Fast.
- **Combined corrected compute-only result**: 42/44 controlled cases are covered; 2/44 violate the profiler-only debt, with worst ratio `1.042621`.

## Interpretation
The simple Prefill-interference abstraction is not disproved yet, but its correct unit cannot be called `full Prefill compute` if the profiling interface omits runtime overhead that also occupies/blockades the HELIX compute-node service path. This correction connects directly to E20/E22: E20 identified CPU-buffer/runtime overhead missing from the Prefill budget, and E22 showed that a generic profiled linear Prefill overhead can recover the isolated HELIX thresholds without hard-coding HELIX constants. The next test should therefore use **Prefill service debt = profiled compute + profiled/bounded runtime overhead**, while keeping network payload/red-line accounting separate.

## Next
Run a clean offline cross-check on the archived E10/E11 observations using the E22 generic overhead calibration as an external profiling input. Report compute-only versus compute+overhead debt separately. Do not hide the corrected 2/44 compute-only violations, do not fit a discount multiplier, and do not change the production Evaluator yet.
