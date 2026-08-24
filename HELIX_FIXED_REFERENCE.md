# HELIX Fixed-Pipeline Reference

Purpose: validate the Conservative Evaluator without building another serving simulator.

The adapter reuses HELIX's public event simulator, profiling, batching/execution policy, network service, and query history. The evaluated Layer-Level pipeline is supplied externally and remains fixed; HELIX placement/max-flow optimization is not invoked.

Current MVP scope: one direct physical link between adjacent pipeline stages. This matches the present Slow/Fast validation pipelines. Multi-hop physical routes are intentionally deferred rather than modeled with invented pass-through compute stages.

TTFT reporting keeps two values separate: `aligned_ttft_s` = HELIX Prefill completion + configured overhead, matching the current Conservative evaluator semantics; `true_first_token_ttft_s` = completion of the first Decode iteration from query arrival, retained as a sensitivity metric. TPOT uses each HELIX Decode iteration's end-to-end duration.
