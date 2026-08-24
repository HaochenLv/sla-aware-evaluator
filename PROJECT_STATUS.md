# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Reference v0: frozen baseline; Decode phase-shift causes excessive GPU queue wait.
- Reference v1 cohort attempt: implementation is mechanically consistent but does not solve multi-stage phase-shift batching, so it is not accepted as the ranking reference.
- Current branch: `feat/reference-evaluator-v1-decode-batching`.

## Experiment log
- **E0 / v0 diagnosis**: exact TPOT attribution closed; Slow/Fast failing tokens were batch=1 at every stage and GPU queue wait dominated.
- **E1 / v1 smoke**: cohort identity persists across stages; trace/decomposition/link controls pass.
- **E2 / regression + phase-shift probe**: 35/35 tests PASS. A two-stage late-arrival probe still produced only batch=1 at stage 0 because a new ready request launches while the previous Decode cohort is downstream. Therefore v1 does not fix the original failure mode.

## Next
Freeze this v1 attempt. Implement a deterministic Decode-round admission policy: at most one Decode cohort is in flight end-to-end; requests becoming ready during a round join the next round. Keep Prefill FIFO, FCFS D/B links, profiling, SLA, workload, memory model, and Conservative Evaluator unchanged. Validate on toy/regression tests before rerunning HELIX Slow/Fast.
