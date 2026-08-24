# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- Current magnitude candidate remains **Prefill blocking-service debt = profiled Prefill compute + profiled/bounded blocking runtime overhead**. Network red-line equations remain unchanged.
- E27 stage-count transfer remains infrastructure-blocked: repeated Actions attempts fail before any workflow step, so there is no out-of-configuration HELIX result yet.

## Experiment log
- **E23 correction / E26**: profiler-only Prefill compute covers 42/44 archived E10/E11 interference cases. Adding independently calibrated E22 blocking/runtime overhead gives 44/44 coverage; tightest service-debt ratio is `0.953143`. E22 also predicts independent E10 measured Prefill compute-node service within `0.05710%` maximum relative error.
- **E25 / frontier cliff**: charging the full historical debt immediately at the first `N_P=1, N_D=1` overlap creates a coarse finite-trace unsafe cliff. Magnitude and exposure timing are separate questions.
- **E28–E29 / hidden phase**: actual Prefill exposure is timing/stage-phase sensitive inside the same scheduler-free macro-state. The tightest observed hidden phase already consumes `95.3143%` of independently profiled blocking-service debt, so a large global multiplier discount is unsupported.
- **E30 / constant-slack stress test**: tested the scheduler-free family `I_sigma = sum(max(0, T_P^blocking - sigma))` on all 44 archived E10/E11 controlled cases. The largest constant per-Prefill slack that preserves every archived bound is only `32.995216 ms`, limited by Slow + 512-token Prefill at offset `0.08 s` (`0.321246 s` observed excess vs `0.354241 s` profiled blocking service). At this fitted limit there are 0/44 violations, but this is an empirical upper limit, not a validated parameter.
- **E30 / does this solve the E25 cliff?** No. Even granting that maximal observed slack and using the more optimistic historical **compute-only** E25 debts, all 10 first-unsafe pipeline×workload cases still exceed the `33 ms` no-debt TPOT residual. The smallest optimistic post-slack debt is `82.845 ms`, still `2.510x` the residual budget; the largest is `648.925 ms`.

## Interpretation
E30 is a negative but useful result: the coarse overlap-onset frontier cannot be fixed by subtracting one small scheduler-free constant from every Prefill debt. A larger constant would already contradict archived controlled HELIX cases, while the largest archived-compatible constant is still far too small to remove any E25 cliff. Therefore do not tune a global multiplier or constant slack merely to smooth capacity. The observed coarseness comes from intentionally hiding fine execution phase while TPOT leaves only a very small residual budget. Full blocking-service debt remains the clean leading worst-case envelope until a genuinely justified cheap state variable is found.

## Next
Do not integrate E30's fitted `sigma`; it is diagnostic only. Keep E27 pending until runners recover. The next implementation experiment should be an opt-in budget-consistent ablation using **blocking-service debt** inside `Delta_D = tau_D - T_decode - I_blocking - T_queue - T_fix`, while preserving the existing event trajectory and network red-line equation. Then validate its safe frontier against HELIX across workload variants when Actions is available. No merge and no default Evaluator semantic change yet.
