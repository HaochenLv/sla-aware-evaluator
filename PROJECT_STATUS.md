# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 is a research-only successor to historical E24. It preserves the existing Conservative event trajectory and normalized-cost network red-line equation.
- Candidate under test is **Prefill blocking-service debt**, not obsolete compute-only debt.

## Experiment log
- **E23 correction / E26**: compute-only Prefill debt covers 42/44 controlled E10/E11 cases; blocking-service debt = profiler compute + independently calibrated E22 blocking overhead covers 44/44. E22 reconstructs independent E10 measured Prefill service within `0.05710%` maximum relative error.
- **E25/E28–E30**: immediate full-debt charging creates a coarse overlap-onset cliff, but simple global tightening is not supported. Hidden execution phase can consume `95.3143%` of full blocking-service debt, and even the largest archived-compatible constant slack (`32.995 ms`) does not remove any E25 cliff.
- **E31 setup**: reuse the budget-consistent structural placement from E24 but replace its magnitude with `I_blocking = sum(T_P^compute,profile + h_P * L_in)` using the E22 midpoint `h_P = 59.377209 us/token` on the same validated 8-stage HELIX configuration. Decode budget becomes `Delta_D = tau_D - T_decode - I_blocking - T_queue - T_fix`. Existing Prefill network commitments, event progression, and network red-line formula are unchanged.
- **E31 guardrail**: the E22 coefficient is experiment-local provenance for this 8-stage configuration. It is not hard-coded as a universal runtime constant; E27 must still validate transfer across stage count before any generic interface claim.
- **E31 execution status**: repeated PR-triggered runs still fail before any workflow step starts. Runs `32703589628` and `32703653672` on `ubuntu-24.04` both had `steps=null` / no job logs. As an infrastructure-only diagnostic, the workflow runner image was changed to `ubuntu-22.04` without changing any experiment semantics; run `32704649314` also failed pre-step with job `97363061844` and `steps=null`. This makes an `ubuntu-24.04` image-specific problem unlikely. No E31 frontier or HELIX probe result is claimed.

## Interpretation
E31 is the first budget-consistent ablation that combines the currently supported interference magnitude with the original SLA-to-resource logic, but its direct frontier remains unmeasured because the hosted runner never starts the job. The failure is infrastructure-side, not an experimental-model result. If E31 remains conservative against HELIX once executable, it supports integrating a blocking-service interference term into Decode's remaining time budget without changing the network red line or adding a scheduler. If it is excessively coarse, E28–E30 already indicate that global discounts/slacks are not a defensible fix; the remaining choice is to accept robust screening coarseness or justify one additional cheap state variable.

## Next
Keep E31 pending rather than changing model semantics to work around infrastructure. When Actions execution is available, run the existing E31 workflow unchanged and inspect: candidate safe/unsafe frontier, first-violation cause/state, and HELIX probes at the candidate edges. Keep E27 pending in parallel. No merge and no default Evaluator semantic change.
