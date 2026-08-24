# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Reference v0: frozen baseline; Decode batching phase-shift causes excessive GPU queue wait.
- Reference v1: deterministic Decode cohorts now persist across stages; Prefill FIFO, FCFS D/B links, profiling, SLA, workload, memory model remain unchanged.
- Current branch: `feat/reference-evaluator-v1-decode-batching`.

## Experiment log
- **E0 / v0 diagnosis**: exact TPOT attribution closed; Slow/Fast failing tokens were batch=1 at every stage and GPU queue wait dominated.
- **E1 / v1 implementation smoke**: isolated compile/core tests PASS. Same-ready requests batch together; a late request joins the next cohort; cohort identity persists; explicit link-driven TPOT failure remains; v1 TPOT decomposition closes with zero unattributed time in the toy test.

## Next
Run full repository regression, then the same HELIX 30 s Slow/Fast v0-v1 experiment. Do not start multi-Pipeline ranking until v1 is validated.
