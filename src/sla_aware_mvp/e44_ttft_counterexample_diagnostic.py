from __future__ import annotations

# E45 diagnostic source retained for reproducibility. See PROJECT_STATUS.md for the
# completed interpretation: the E44 TTFT optimism is caused by intrinsic Prefill
# per-stage service overhead omitted from Prefill's own TTFT budget, not by
# cross-request queueing/interference or material network mismatch.

# The executable diagnostic implementation is preserved in
# e44_ttft_counterexample_diagnostic_runner.py; this module is kept only as the
# original experiment record after the successful run.
