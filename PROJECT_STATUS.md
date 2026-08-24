# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Reference v0: frozen baseline; Decode phase-shift causes excessive GPU queue wait.
- Reference v1 cohort attempt: rejected; multi-stage phase shift remained.
- Reference Decode-round candidate: at most one Decode cohort is in flight end-to-end; requests becoming ready during a round join the next round. Prefill FIFO, FCFS D/B links, profiling, SLA, workload, memory model remain unchanged.
- Current branch: `feat/reference-evaluator-v1-decode-round`.

## Experiment log
- **E0 / v0 diagnosis**: exact TPOT attribution closed; failing tokens were batch=1 and GPU queue wait dominated.
- **E1-E2 / cohort v1**: core tests passed, but a two-stage late-arrival probe still stayed batch=1, so the policy was rejected.
- **E3 / Decode-round candidate**: phase-shift probe now forms a later batch=2 and preserves that batch across stages; single-request drain, explicit link failure, capacity bracketing, TPOT decomposition, and infinite-network control pass. Current mirrored regression: 42/42 PASS.

## Next
Rerun the official HELIX-derived 30 s Slow/Fast experiment with Decode-round and compare capacity, batch-size distribution, GPU/link queue attribution, and wall time against Reference v0. Do not start multi-Pipeline ranking until this check passes.
