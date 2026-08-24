# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Hand-written Reference schedulers: frozen; they became too complex and are no longer the target validation path.
- New branch: `feat/helix-fixed-pipeline-reference`.
- Validation target: reuse HELIX public event simulator/runtime with an externally fixed Layer-Level pipeline; HELIX placement/max-flow optimization must not participate.

## Experiment log
- **E0-E4 / old Reference**: explicit simulation worked, but scheduler details materially changed capacity, showing that rebuilding a serving simulator is the wrong direction.
- **E5 / HELIX interface study**: feasible with a thin adapter. HELIX requests already support a preset `PipelineStage` list; later Decode iterations copy the same pipeline. HELIX also exposes its FIFO mixed Prefill/Decode `execution_policy` and request/query latency histories. Fixed layer placement can be loaded directly; no placement optimization is required.

## Next
Build only a thin fixed-pipeline replay adapter around HELIX: fixed route injection + fixed layer loading + finite workload replay + TTFT/TPOT extraction + request-rate feasibility search. Do not implement a new scheduler or simulator.
