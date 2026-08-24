# Project Status

## Current state
- Conservative Evaluator remains unchanged on the validated base; HELIX fixed-pipeline Reference remains draft PR #6 and is not ground truth.
- Diagnostic PRs #8–#12 are research-only; do not merge.

## Experiment log
- **E7**: Conservative and HELIX both rank Fast > Slow, but the original Conservative capacity was far higher.
- **E9–E11**: HELIX diagnostics confirm real Prefill→Decode queue/interference; full active Prefill compute is a viable conservative interference upper-bound candidate in controlled cases.
- **E12**: adding that Prefill guard to the unchanged Conservative trace still rejected too late, exposing a more basic profile/runtime mismatch.
- **E13 / singleton Decode alignment diagnostic**: pinned HELIX runtime doubles interpolated Decode time when a batch contains exactly one Decode token. Our profiler adapter omitted that rule, so the original Conservative used ~0.056 s per Decode token while HELIX isolated service is ~0.112 s. A diagnostic 2× Decode wrapper alone moves both Slow/Fast Conservative frontiers to safe 0.0132 / unsafe 0.0134 intensity (0.008533 / 0.008663 rps), very close to HELIX Slow [0.0134,0.0136) and Fast [0.0136,0.0138). The diagnostic is now slightly more conservative than HELIX. Adding the Prefill guard on top moves the frontier further left to [0.0130,0.0132), so it would over-correct E7 at this stage.

## Revised diagnosis
The giant E7 discrepancy was primarily caused by an underestimated Decode compute term, not by the missing queue model alone. The network red-line formulation itself is not contradicted: it was being fed an overly optimistic `T_comp,D`, leaving ~89 ms TPOT slack instead of roughly 33 ms before network/fixed costs.

## Next
Test the exact HELIX runtime rule (double only when active Decode batch size is 1; keep n_decode>=2 profile values unchanged) and inspect the first `sla_time` violation near the frontier. Only after that decide whether any Prefill interference guard is still required.
