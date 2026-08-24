# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current E31 candidate keeps the original event trajectory and network red-line and charges profiled Prefill blocking-service debt inside Decode's SLA remaining-time budget.
- E38 is observational only: no production Evaluator semantic change, scheduler reconstruction, or network red-line change.

## Experiment log
- **E31/E33/E34/E35 safety/tightness**: candidate-safe remained HELIX-safe in all tested workload×pipeline checks; exact frontier conservatism ranges from about `3.65%` to `20.91%`, with seed19 Slow the worst observed case.
- **E36/E37 exposure result**: seed19 Slow over-conservatism includes phase-dependent Prefill exposure, and cheap event/request observables do not resolve it. E37 found identical visible Arrival state with exposure `3.4334%` versus `40.3133%`, and another collision at `87.4482%` versus `95.3687%`.
- **E37 trajectory clue**: at seed19 Slow intensity `0.0153`, Conservative still has `azure-00009` active at progress `413.267/415` when `azure-00010` arrives, while pinned HELIX has already completed `415/415`.
- **E38 setup**: compare the victim timeline at intensity `0.0153` and the exact Slow HELIX safe/unsafe edges in three views: current Conservative progression (`profiled compute + fixed overhead`), an observational `fixed=0` progression counterfactual, and raw pinned HELIX runtime. HELIX still adds the 5 ms fixed overhead only when extracting SLA metrics; it does not slow runtime progression.
- **E38 fixed-overhead accumulation**: current Conservative target Decode span is `48.555 s`; the `fixed=0` counterfactual is `46.480 s`. The difference is exactly `2.075 s = 415 × 5 ms`. Including the one Prefill fixed charge, removing fixed overhead from progression advances target Finish by `2.080 s` in all three tested intensities.
- **E38 candidate point `0.0153`**: current Conservative finishes `azure-00009` at `1230.192988 s`, `1.447347 s` later than raw HELIX (`1228.745641 s`) and therefore still has it active at the new Arrival. The `fixed=0` trajectory finishes at `1228.112988 s`, `0.632653 s` earlier than HELIX and, like HELIX, has already finished before `azure-00010` arrives at `1229.990196 s`. Thus the specific false overlap that triggers E31 at `0.0153` disappears when SLA fixed overhead is not accumulated as physical per-token progress time.
- **E38 safe/unsafe edges**: at `0.0183779296875`, current Conservative progress at interferer Arrival is `343.10`, fixed=0 is `358.46`, and HELIX has completed `353`; at `0.01838091796875`, the corresponding values are `343.04`, `358.40`, and `352`. Removing fixed therefore over-corrects relative to raw HELIX by roughly 5–6 tokens at these points, but stays much closer in Finish time: current Finish is `+1.447/+1.354 s` late, fixed=0 is `-0.633/-0.726 s` early.
- **E38 Prefill timing**: both Conservative trajectories finish the victim Prefill about `0.451–0.456 s` earlier than raw HELIX (`0.6674/0.6624 s` versus `1.11825 s` Prefill latency). This is an opposite-direction trajectory mismatch because the Evaluator progression is profiling-driven and does not replay HELIX's end-to-end Prefill runtime/network path.
- **E38 conclusion**: seed19 false overlap is materially caused by a semantic alignment issue: the current Evaluator accumulates `T_fix=5 ms` as physical progress on every Decode token, while the HELIX reference uses that 5 ms only as SLA accounting overhead. However, simply deleting fixed overhead from physical progression is not yet validated as a safe model change because other runtime components make the counterfactual trajectory earlier than HELIX.

## Interpretation
E38 separates two trajectory effects. The 5 ms fixed SLA overhead contributes a deterministic `2.075 s` delay over 415 Decode tokens and is large enough to create the candidate-point overlap that does not exist in HELIX. At the same time, the Conservative trajectory omits other end-to-end runtime components and therefore starts/finishes portions of the request earlier than HELIX when that fixed term is removed. The correct modeling question is therefore not a fitted exposure multiplier; it is whether **SLA accounting overhead and physical progression time should be separated** while keeping the full blocking-service debt and network red-line safety checks.

## Next
1. Run a seed19-only split-semantics ablation: progress with profiling time only, but retain the original 5 ms fixed overhead inside TTFT/TPOT remaining-budget and network red-line calculations.
2. Test both Slow and Fast candidate frontiers against pinned HELIX before broadening to other workloads; reject the split if any candidate-safe point is HELIX-unsafe.
3. Do not modify production Evaluator/default semantics or merge any diagnostic PR yet.
