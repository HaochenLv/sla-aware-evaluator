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
- **E8 / Reference metric semantics**: HELIX represents each Decode token as one `RequestPhase.Increment` request. The next Decode iteration is issued exactly when the previous iteration reaches the sink, so an Increment's source-to-sink latency equals that query's token-to-token interval. Thus the extracted Decode iteration latency is semantically valid for strong per-token TPOT. However, current `aligned_ttft` is Prefill completion, while true TTFT is first Decode completion. HELIX's documented ~150 ms simulator/prototype latency gap also prevents treating absolute 150 ms TPOT capacity as ground truth.

## Next
Keep HELIX as a relative execution reference. Before modifying Conservative, separate two questions: (1) use true first-token TTFT for external validation; (2) test ranking/relative-capacity behavior under HELIX rather than interpreting E7 absolute capacity as real-system ground truth. Do not add simulator complexity.
