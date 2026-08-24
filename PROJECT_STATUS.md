# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Hand-written Reference schedulers: frozen; no longer the validation path.
- Current branch: `feat/helix-fixed-pipeline-reference`; draft PR #6 is open and must not be merged yet.
- New Reference path: thin adapter around HELIX public simulator with externally fixed Layer-Level pipeline; no HELIX placement/max-flow optimization.

## Experiment log
- **E0-E4 / old Reference**: scheduler details materially changed capacity; rebuilding a serving simulator was rejected.
- **E5 / HELIX interface study**: fixed pipeline injection is feasible; HELIX already provides runtime execution, network service, profiling, and latency history.
- **E6 / adapter implementation**: added fixed-route wrapper, fixed layer loading, finite workload replay, aligned/true TTFT extraction, TPOT extraction, and a two-request Slow/Fast smoke. Unit tests cover direct-route validation and metric semantics. GPU execution still calls HELIX `execution_policy`; no new serving simulator is implemented.

## Next
Run pinned HELIX integration CI. If the two-request fixed-pipeline smoke passes, add only a small request-rate feasibility/capacity-search wrapper and compare HELIX Reference capacity with Conservative capacity on the same 30 s Slow/Fast workload.
