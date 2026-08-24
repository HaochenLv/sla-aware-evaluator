from __future__ import annotations

from .exact_singleton_decode_demo import ExactHelixDecodeRuntimeProfiler


def _prefill_stage_time(self, *args, **kwargs):
    return self.base.prefill_stage_time(*args, **kwargs)


# Experiment-local delegation only: the production profiler class remains unchanged.
ExactHelixDecodeRuntimeProfiler.prefill_stage_time = _prefill_stage_time

from .e44_ttft_counterexample_diagnostic import main  # noqa: E402


if __name__ == "__main__":
    main()
