from __future__ import annotations

from run_260226 import main


if __name__ == "__main__":
    main(
        full_benchmark=False,
        include_amlb=True,
        amlb_categories=("amlb_top20_mix",),
        show_progress=True,
    )
