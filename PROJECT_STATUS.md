# Project Status

## Current state
- Conservative Evaluator: stabilized; fixed-Pipeline capacity is reproducible and separates Slow/Fast.
- Reference v0: explicit GPU/link service works, but Decode batching phase-shifts requests; failing tokens often run batch=1, inflating GPU queue wait and depressing capacity.
- Current branch: `feat/reference-evaluator-v1-decode-batching`.

## Experiment log
- **E0 / Reference v0 diagnosis**: exact TPOT trace closed to GPU queue + GPU service + link queue + link service + overhead. Slow/Fast first failing tokens were batch=1 at every stage; GPU queue wait dominated, so Reference v0 is not yet a credible ranking target.
- **E1 / Reference v1 plan**: replace opportunistic per-GPU Decode batching with deterministic Decode cohorts that persist across stages; keep Prefill FIFO, explicit FCFS links, HELIX profiling, SLA, workload, memory model, and Conservative Evaluator unchanged.

## Next
Implement Reference v1 cohort batching, run unit/regression tests, then rerun the same Slow/Fast 30 s diagnostic before any multi-Pipeline ranking experiment.
