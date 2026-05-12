"""Run the ONIRO Colab demo end-to-end. Entry point for `python -m oniro`."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    runpy.run_path(str(here / "demo" / "oniro_colab_demo.py"), run_name="__main__")


if __name__ == "__main__":
    main()
