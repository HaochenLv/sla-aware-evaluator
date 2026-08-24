# Project Status

## Current state
- Conservative Evaluator: stabilized; unchanged.
- Hand-written Reference schedulers: frozen; no longer the validation path.
- Current branch: `feat/helix-fixed-pipeline-reference`; draft PR #6 is open and must not be merged yet.
- Reference reuses pinned HELIX public runtime with an externally fixed Layer-Level pipeline; HELIX placement/max-flow optimization is not used.

## Experiment log
- **E6 / HELIX fixed-pipeline integration**: smoke + unit/regression CI PASS; no new serving simulator/scheduler was implemented.
- **E7 / 30 s, 17-request capacity**: Conservative Slow = 1.2031 rps safe (1.2211 unsafe); Fast >= 10.3434 rps (right-censored). HELIX Reference Slow capacity is in [0.008663, 0.008792) rps; Fast is in [0.008792, 0.008921) rps. Fast > Slow is therefore resolved pairwise.
- Under this strict every-token TPOT criterion, Conservative is not conservative relative to HELIX Reference: Slow overestimates by about 137-139x; Fast by at least 1159x. HELIX transitions abruptly from max TPOT ~0.117 s to large TPOT violations when one request overlap appears.

## Next
Do not add simulator complexity. Inspect the single overlap/TPOT jump and verify whether the current strict all-token SLA capacity criterion is the intended validation target. Then decide whether Conservative needs a simple queue/interference guard or should be positioned as a ranking/screening proxy rather than a hard capacity bound.
