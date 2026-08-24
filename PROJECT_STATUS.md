# Project Status

## Current state
- Conservative Evaluator default remains unchanged; HELIX fixed-pipeline Reference remains draft PR #6 and is a relative execution reference, not ground truth.
- E31 is a research-only successor to historical E24. It preserves the existing Conservative event trajectory and normalized-cost network red-line equation.
- Candidate under test is **Prefill blocking-service debt**, not obsolete compute-only debt.

## Experiment log
- **E23 correction / E26**: compute-only Prefill debt covers 42/44 controlled E10/E11 cases; blocking-service debt = profiler compute + independently calibrated E22 blocking overhead covers 44/44. E22 reconstructs independent E10 measured Prefill service within `0.05710%` maximum relative error.
- **E25/E28–E30**: immediate full-debt charging creates a coarse overlap-onset cliff, but simple global tightening is not supported. Hidden execution phase can consume `95.3143%` of full blocking-service debt, and even the largest archived-compatible constant slack (`32.995 ms`) does not remove any E25 cliff.
- **E31 setup**: reuse the budget-consistent structural placement from E24 but replace its magnitude with `I_blocking = sum(T_P^compute,profile + h_P * L_in)` using the E22 midpoint `h_P = 59.377209 us/token` on the same validated 8-stage HELIX configuration. Decode budget becomes `Delta_D = tau_D - T_decode - I_blocking - T_queue - T_fix`. Existing Prefill network commitments, event progression, and network red-line formula are unchanged.
- **E31 execution**: after the repository was made public, rerun `32704649314` completed successfully and uploaded artifact `9512219265`.
- **E31 frontier**: for both Slow and Fast fixed pipelines, candidate safe intensity is `0.0131` and first unsafe intensity is `0.0132` on the `0.0001` grid. With base arrival rate `0.6464646465 rps`, this is a safe lower bound `0.0084686869 rps` and unsafe upper bound `0.0085333333 rps`.
- **E31 first unsafe mechanism**: both pipelines first fail at the same `N_P=1, N_D=1` state. Decode compute is `0.112 s`, profiled blocking-service debt is `0.7490756231 s`, and fixed overhead is `0.005 s`, so required compute-side time is `0.8660756231 s > 0.150 s` TPOT. The old no-debt network residual was `0.033 s`; after debt it becomes negative (`-0.7160756231 s`). The first violation is therefore an interpretable `sla_time` exhaustion, not a mysterious network-only failure.
- **E31 HELIX probes**: at candidate safe `0.0131`, HELIX is feasible for both pipelines. At candidate first-unsafe `0.0132`, HELIX is still feasible for both pipelines (`max TPOT` Slow `0.117426 s`, Fast `0.117151 s`). At `0.01386` (5% above candidate unsafe), HELIX violates TPOT for both pipelines (Slow `1.084347 s`, Fast `0.825945 s`). Thus the candidate is conservative on this workload, but the exact HELIX frontier remains only bracketed above the candidate; E31 does not yet quantify the tightest gap.
- **E31 guardrail**: the E22 coefficient is experiment-local provenance for this 8-stage configuration. It is not a universal runtime constant; E27 must still validate transfer across stage count before any generic interface claim.

## Interpretation
E31 closes the first end-to-end loop for the current candidate: independently supported Prefill blocking-service debt can be inserted into Decode's remaining SLA budget while preserving the original event trajectory and network red-line logic. The resulting candidate is conservative against the pinned HELIX reference on the tested seed-7 workload: its `0.0131` safe point is HELIX-safe, while HELIX remains safe at the candidate's `0.0132` first-unsafe point and becomes unsafe by `0.01386`. This is evidence for safety, not yet evidence for a tight capacity estimate. The remaining important questions are frontier tightness across multiple workloads and transfer of the blocking-overhead profile across stage counts.

## Next
1. Refine the HELIX frontier between `0.0132` and `0.01386` for Slow/Fast to quantify E31 conservatism on this workload.
2. Repeat E31 on multiple workload seeds/windows to test for optimism or pathological over-conservatism.
3. Run E27 stage-count transfer (7/8/10 stages) now that hosted Actions is working again.
Do not merge and do not change the default Evaluator semantics yet.
