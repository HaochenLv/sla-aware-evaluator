# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Reference v0: frozen baseline; first unsafe HELIX run has large GPU queue wait.
- Cohort v1: rejected; multi-stage phase shift remained.
- Decode-round candidate: fixes the phase-shift toy, but the official HELIX 30 s result is numerically identical to v0; it is not accepted as the ranking reference.
- Current branch: `feat/reference-evaluator-v1-decode-round`.

## Experiment log
- **E0 / v0 diagnosis**: exact TPOT attribution closed; failing tokens were batch=1 and GPU queue wait dominated.
- **E1-E3 / batching repairs**: cohort v1 failed the multi-stage toy; Decode-round repaired the toy and passed 42/42 regression tests.
- **E4 / official HELIX 30 s v0 vs round**: CI PASS. Slow and Fast both keep the exact v0 capacity bracket (`0.0243056` safe, `0.0260417` unsafe) and the same failing TPOT/batch=1 trace. Round gating therefore does not address the real HELIX failure mode.

## Next
Do not change scheduling again yet. Attribute the failing token's GPU queue wait to the phase occupying each GPU (Prefill vs Decode vs idle/synchronization). This decides whether the next reference correction should target Prefill/Decode scheduling rather than Decode batching.
