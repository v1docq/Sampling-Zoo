from __future__ import annotations

from run_260226 import run_bench_pipeline


if __name__ == "__main__":
    run_bench_pipeline(
        full_benchmark=False,
        include_amlb=True,
        amlb_categories=("amlb_top20_mix",),
        show_progress=True,
    )
