# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Hand-written Reference schedulers: frozen; no longer the validation path.
- Current branch: `feat/helix-fixed-pipeline-reference`; draft PR #6 is open and must not be merged yet.
- Reference now reuses HELIX public simulator with an externally fixed Layer-Level pipeline; HELIX placement/max-flow optimization is not used.

## Experiment log
- **E0-E4 / old Reference**: rebuilding serving scheduling materially changed capacity, so that direction was rejected.
- **E5 / HELIX interface study**: fixed pipeline injection is feasible; HELIX already provides execution, network service, profiling, and latency history.
- **E6 / fixed-pipeline smoke**: integration CI PASS. Two requests complete on both Slow/Fast; Fast has lower TTFT/TPOT than Slow. The adapter is mechanically connected to pinned HELIX. Added a small arrival-rate capacity wrapper; unit/regression CI also passes.
- **E7 / 30 s capacity comparison**: currently executing the same 17-request Azure-derived workload under Conservative and HELIX Reference; this is intentionally slower because HELIX explicitly simulates layer-level serving execution.

## Next
Inspect E7 capacity bounds. If HELIX capacity is stable and Slow/Fast ordering is sensible, use this Reference for small-scale validation of Conservative request-rate estimates; do not build another simulator.
