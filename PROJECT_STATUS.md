# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- Current candidate keeps the original event trajectory and network red-line and charges profiled Prefill blocking-service debt inside Decode's SLA remaining-time budget.
- E36 is diagnostic only: no scheduler reconstruction, red-line change, or production Evaluator semantic change.

## Experiment log
- **E27 transfer**: stage-additive Prefill blocking overhead matches direct HELIX service across 7/8/10 stages × 256/512/1024 prompts; max absolute relative error `0.0698798%`.
- **E31/E33/E34/E35**: candidate remains conservative in all tested workload×pipeline safe-edge checks, but exact frontier gap ranges from about `3.65%` to `20.91%`; seed19 Slow is the worst observed case.
- **E36 Evaluator first-unsafe state (seed19 Slow, intensity 0.0153)**: the first violation is triggered exactly by an `Arrival` event. Previous state is `N_P=0,N_D=1`; arrival of `azure-00010` makes `N_P=1,N_D=1` while `azure-00009` is decoding. The new Prefill has age `0.0 s`, prompt `181` tokens, and full profiled blocking-service debt `0.126587275 s`. The Decode request has prompt `1070`, output `415`, progress `413.267/415`, context `1485`, and singleton Decode compute `0.112 s`. E31 immediately charges the full Prefill debt, so `0.112 + 0.126587 + 0.005 = 0.243587 s > 0.150 s` and declares `sla_time` unsafe.
- **E36 HELIX at the same candidate point**: for Slow at intensity `0.0153`, `azure-00009` has `0.000000 s` queue wait and TPOT+fixed `0.117426 s`. The Evaluator therefore charges `126.587 ms` of possible Prefill blocking at a coarse overlap state where HELIX observes zero actual blocking for the target Decode iteration.
- **E36 +5% phase exposure**: at Slow intensity `0.016065`, the same Prefill `azure-00010` blocks the same Decode request for only `16.655 ms` (`13.16%` of full debt), keeping TPOT+fixed at `0.134081 s`. At Fast with the same intensity, the same blocker causes `36.972 ms` queue wait (`29.21%` of full debt), producing TPOT+fixed `0.154122 s` and violation. Thus Slow/Fast can have identical coarse `(N_P,N_D)` but different actual blocking exposure because execution phase differs.
- **E36 near the exact Slow frontier**: at HELIX safe edge `0.0183779296875`, the target Decode again sees `0 ms` queue wait. Increasing intensity by only `2.99e-6` to `0.01838091796875` produces `93.079 ms` Prefill queue blocking (`73.53%` of full debt), entirely from `azure-00010`, and TPOT+fixed jumps to `0.210505 s`. This confirms a discrete phase-alignment cliff rather than a smooth utilization crossing.
- **E36 runtime decomposition**: the target Decode layer service remains essentially fixed at `0.112059 s`; network/other residual is only `0.367 ms` on Slow and `0.092 ms` on Fast. The changing term is Prefill-induced queue wait, and the blocker is consistently `azure-00010` when blocking occurs.

## Interpretation
E36 localizes the seed19 Slow tightness problem. The magnitude model for full Prefill blocking-service debt is not contradicted: the same Prefill can create substantial queue blocking near the HELIX cliff. The over-conservatism comes from **exposure timing**: the scheduler-free Evaluator treats `Prefill active + Decode active` as if the Decode could immediately experience the Prefill's full blocking service, beginning at the Prefill Arrival event. In HELIX, actual exposure can be `0%`, `13%`, `29%`, or `74%` of that same full debt depending on pipeline/execution phase. This also explains why E31 ties Slow/Fast at the coarse time-budget cliff while HELIX distinguishes them.

The result does **not** justify a global discount on blocking debt. Prior E29 evidence observed exposure up to `95.3143%`, while E36 shows exposure is highly phase-dependent and discontinuous. The next modeling question is whether one cheap event-level observable can bound exposure more tightly than binary active-state overlap without reconstructing the serving scheduler.

## Next
1. Test cheap observables already available to the Evaluator, especially Prefill age since Arrival, Decode progress/context, and event type (`Arrival` versus later block updates), against archived E10/E11/E28 cases plus E36.
2. Reject any rule that loses the existing no-optimism safety evidence; do not fit a seed19-specific coefficient.
3. Keep full blocking-service debt as the safety baseline until a richer state bound is independently validated.
4. Do not merge and do not change default Evaluator semantics yet.
