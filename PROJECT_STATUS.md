# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current candidate keeps the original event trajectory and network red-line and charges profiled Prefill blocking-service debt inside Decode's SLA remaining-time budget.
- E37 is diagnostic only: no scheduler reconstruction, red-line change, or production Evaluator semantic change.

## Experiment log
- **E27 transfer**: stage-additive Prefill blocking overhead matches direct HELIX service across 7/8/10 stages × 256/512/1024 prompts; max absolute relative error `0.0698798%`.
- **E31/E33/E34/E35**: candidate remains conservative in all tested workload×pipeline safe-edge checks, but exact frontier gap ranges from about `3.65%` to `20.91%`; seed19 Slow is the worst observed case.
- **E36 localization**: seed19 Slow first becomes candidate-unsafe at intensity `0.0153` on the Arrival of `azure-00010` while Evaluator state is `N_P=1,N_D=1`; E31 immediately charges the full `126.587 ms` Prefill blocking-service debt. HELIX queue tracing shows actual exposure is phase-dependent (`0%`, `13%`, `29%`, `74%` in the diagnosed points), with stable Decode service and sub-ms network/other residual.
- **E37 controlled cheap-observable audit**: re-ran focused E10-style one-Decode/one-Prefill cases while recording only request/event-level state available to the scheduler-free Evaluator: event type, Prefill age, completed Decode tokens, 16-token Decode block, context, and remaining tokens. Fast + 256-token Prefill at offsets `0.01 s` and `0.08 s` has an *identical* visible state at Prefill Arrival: event=`Arrival`, Prefill age=`0`, Decode completed=`0`, block=`0`, context=`256`, remaining=`64`, and an in-flight Decode iteration in both cases. Yet actual Prefill exposure changes from `3.4334%` to `40.3133%` of full Prefill service, a `36.8799 percentage-point` spread. Slow + 1024-token Prefill shows the same visible-state collision at offsets `0.01/0.08`, while exposure changes from `87.4482%` to `95.3687%`.
- **E37 seed19 progress audit**: at Slow intensity `0.0153`, HELIX has already completed all `415/415` Decode tokens for `azure-00009` by the Arrival of `azure-00010`, so there is no HELIX active Decode overlap and observed exposure is `0%`. This differs from the Conservative trajectory at the same workload point, where E36 still records `azure-00009` active at progress `413.267/415`. At the exact HELIX safe edge `0.0183779296875`, the target has completed `353/415` tokens (block 22) and exposure is `0%`; at the exact unsafe edge `0.01838091796875`, it has completed `352/415` tokens (also block 22) and exposure jumps to `73.5709%`.
- **E37 falsification result**: event type + Prefill age + request-level Decode progress/context are not sufficient to determine exposure. The audit found two groups with identical visible state but materially different exposure; the worst identical-state spread is `36.8799 percentage points`. A simple rule based only on these observables cannot safely remove the full-debt envelope.

## Interpretation
E37 narrows the problem further. First, exposure depends on hidden intra-iteration/stage timing even when event type, Prefill age, exact completed Decode tokens, block index, context, and remaining output are identical. This independently rejects the simplest request/event-level tightening rules. Second, seed19 reveals an additional source of conservatism: **trajectory alignment**. At intensity `0.0153`, the Conservative Evaluator still has the victim Decode active when the new Prefill arrives, whereas pinned HELIX has already finished that request. Therefore the ~20.9% seed19 gap is not purely an exposure-multiplier problem; part of it can arise because the coarse profiling-driven event trajectory creates an overlap that the reference execution does not have.

The result does **not** justify a global discount or a seed19-specific coefficient. Prior E29/E37 controlled cases still require up to about `95%` of full blocking-service debt. The next smallest experiment should isolate Conservative-vs-HELIX transition/finish timing for the seed19 victim request and decompose the trajectory mismatch into Prefill completion timing, per-token Decode timing, block-event discretization, and fixed/network residuals before changing the model.

## Next
1. Audit seed19 `azure-00009` Conservative vs HELIX timeline at the same intensities: Prefill completion, Decode start, selected Decode block boundaries, and Finish.
2. Quantify how much of the candidate/HELIX capacity gap is caused by trajectory mismatch versus exposure timing.
3. Keep full blocking-service debt as the safety baseline; do not add a discount or scheduler-phase reconstruction.
4. Do not merge and do not change default Evaluator semantics yet.
