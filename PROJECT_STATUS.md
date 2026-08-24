# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Hand-written Reference schedulers: frozen; no longer the validation path.
- Current branch: `feat/helix-fixed-pipeline-reference`; draft PR #6 is open and must not be merged yet.
- Reference reuses pinned HELIX public runtime with an externally fixed Layer-Level pipeline; HELIX placement/max-flow optimization is not used.
- HELIX simulator is a validated execution reference, not ground truth: the paper reports <1% online decode-throughput difference versus its real prototype, but also a systematic ~150 ms latency gap from unmodeled CPU-GPU transfer overhead.

## Experiment log
- **E6 / HELIX fixed-pipeline integration**: smoke + unit/regression CI PASS; no new serving simulator/scheduler was implemented.
- **E7 / 30 s, 17-request capacity**: Conservative Slow = 1.2031 rps safe (1.2211 unsafe); Fast >= 10.3434 rps (right-censored). HELIX Reference Slow capacity is in [0.008663, 0.008792) rps; Fast is in [0.008792, 0.008921) rps. Fast > Slow is resolved pairwise.
- **E8 / Reference validity check**: HELIX authors validate simulator fidelity against their vLLM/ZeroMQ prototype. Relative throughput is strong, but absolute latency has ~150 ms systematic error. Therefore E7 ranking evidence is usable; E7 absolute every-token TPOT capacity is provisional and must not be treated as ground truth yet.

## Next
Verify that the latency quantity extracted from HELIX corresponds exactly to our per-token TPOT definition, and determine how the documented ~150 ms simulator/prototype latency gap should be handled. Do not modify Conservative or add simulator complexity before this semantic check.
