# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- E26 is a research-only offline cross-experiment consistency check. Network red-line equations and runtime/state-machine semantics remain unchanged.
- Correct candidate terminology for this branch: **Prefill service debt = profiled Prefill compute + profiled/bounded Prefill runtime overhead**.

## Experiment log
- **E23 correction**: profiler-only Prefill compute debt covers 42/44 controlled E10/E11 cases, not 44/44. The two violations are the 1024-token one-Prefill cases at relative offset 0.08: excess/compute-debt is `1.037160` Fast and `1.042621` Slow.
- **Independent overhead input from E22**: E22 inferred `T_ovhd^P(L)=h_P L` from isolated E20 HELIX aligned-TTFT threshold brackets, with joint midpoint `h_P = 59.377209 us/token`. E10/E11 queue/interference observations were not used to fit this coefficient.
- **E26 / service-debt cross-check**: adding the independently calibrated E22 Prefill overhead to the HELIX prompt compute profile gives 44/44 coverage of `Decode queue wait + positive Decode service inflation` across the archived E10/E11 controlled cases. Compute-only has 2/44 violations; compute+profiled-overhead has 0/44.
- **E26 tightest case**: Slow, one 1024-token Prefill at offset 0.08. Observed Decode compute-side excess is `0.675285 s`; profiler-only Prefill compute is `0.647680 s` (ratio `1.042621`, violation); E22-overhead adds `0.060802 s`, giving service debt `0.708482 s` and ratio `0.953143` (bound holds).
- **E26 multi-Prefill check**: E11 remains comfortably covered. The largest service-debt ratio is `0.620641` for Slow 512+1024; heavy 3x512 service-inflation cases also remain below the service debt.

## Interpretation
E26 resolves the apparent E23 contradiction without changing the network red line and without fitting an interference-specific safety coefficient. The missing quantity is the same upstream Prefill runtime-overhead term already exposed by E20/E22. In the pinned HELIX runtime, that overhead lengthens Prefill compute-node service and therefore can also extend Prefill-induced Decode blocking. A compute-only interference debt is therefore incomplete under the current profiling interface; a **profiled service-time debt** is the cleaner abstraction. This is stronger evidence for a unified profiling/budget interface: `T_P^service,UB = T_P^compute,profile + T_P^overhead,profile/bound`.

## Next
Do not integrate yet. First test whether the same overhead-inclusive service-debt abstraction remains conservative when the independently calibrated overhead is transferred across a different stage count or another prompt region. Separately, E25 still shows that charging the full service debt immediately at first Prefill+Decode overlap can create a coarse finite-trace frontier; magnitude correctness and exposure timing are now two distinct questions. Keep them separate rather than tuning one coefficient to solve both.
