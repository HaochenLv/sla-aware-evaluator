# Project Status

## Current state
- Conservative Evaluator remains unchanged; HELIX fixed-pipeline Reference stays draft PR #6 and is a relative execution reference, not ground truth.
- E26 is a research-only offline cross-experiment consistency check. Network red-line equations and runtime/state-machine semantics remain unchanged.
- Correct candidate terminology: **Prefill blocking-service debt**, not generic total overhead debt.

## Experiment log
- **E23 correction**: profiler-only Prefill compute debt covers 42/44 controlled E10/E11 cases, not 44/44. The two violations are the 1024-token one-Prefill cases at relative offset 0.08: excess/compute-debt is `1.037160` Fast and `1.042621` Slow.
- **Independent overhead input from E22**: E22 inferred `T_ovhd^P(L)=h_P L` from isolated E20 HELIX aligned-TTFT threshold brackets, with joint midpoint `h_P = 59.377209 us/token`. E10/E11 queue/interference observations were not used to fit this coefficient.
- **E26 / independent Prefill-service transfer**: applying that E22 midpoint to the HELIX prompt compute profile predicts the E10 tracer's independently measured Prefill compute-node service at 64/256/512/1024 tokens with maximum relative error only `0.05710%`. Examples: 1024-token predicted service `0.708482 s` vs measured `0.708078 s`; 512-token predicted `0.354241 s` vs measured `0.354039 s`. This is a cross-experiment check, not a fit to E10.
- **E26 / service-debt interference cross-check**: adding the independently calibrated E22 Prefill overhead to the HELIX prompt compute profile gives 44/44 coverage of `Decode queue wait + positive Decode service inflation` across archived E10/E11 cases. Compute-only has 2/44 violations; compute+profiled-overhead has 0/44.
- **E26 tightest case**: Slow, one 1024-token Prefill at offset 0.08. Observed Decode compute-side excess is `0.675285 s`; profiler-only Prefill compute is `0.647680 s` (ratio `1.042621`, violation); E22-overhead adds `0.060802 s`, giving service debt `0.708482 s` and ratio `0.953143` (bound holds).
- **E26 multi-Prefill check**: E11 remains comfortably covered. Largest service-debt ratio is `0.620641` for Slow 512+1024; heavy 3x512 service-inflation cases also remain below the service debt.

## Interpretation
E26 resolves the E23 contradiction without changing the network red line or fitting an interference-specific coefficient. The E22 coefficient, calibrated only from isolated TTFT bandwidth thresholds, independently reconstructs E10 Prefill compute-node service to ~0.06%. In pinned HELIX, this buffer/runtime overhead is part of compute-node batch service, so it can extend Prefill-induced Decode blocking. Therefore the useful generic quantity is not automatically `all Prefill latency overhead`; it is the portion of Prefill service that occupies the resource and can block Decode. A cleaner interface is `T_P^blocking,UB = T_P^compute,profile + T_P^blocking-overhead,profile/bound`. Non-blocking/asynchronous overhead may still belong in Prefill TTFT budget but must not automatically be charged as Decode interference.

## Next
Do not integrate yet. E27 tests whether the blocking-overhead component transfers across stage count. Separately, E25 shows that charging the full blocking-service debt immediately at first Prefill+Decode overlap can create a coarse finite-trace frontier; magnitude correctness and exposure timing remain distinct questions.
