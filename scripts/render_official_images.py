"""Rerender every indexed official PNG under another directory.

Example:
    .nav/bin/python scripts/render_official_images.py --output outputs/official-rerender

This executes plotting code and uses saved numerical inputs. It never copies
PNG bytes from imgs/oficiais. The final check requires the same 138 relative
PNG paths as the official archive.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "imgs/oficiais"
GENERATORS = (
    "render_official_fixed.py",
    "render_official_sargolini_bert.py",
    "render_official_sargolini_gpt.py",
    "render_official_sargolini_viz.py",
    "render_official_ffw_atlases.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    archive = ARCHIVE.resolve()
    if destination in (ROOT.resolve(), archive) or archive in destination.parents:
        raise ValueError("Output must be a separate directory outside imgs/oficiais")
    destination.mkdir(parents=True, exist_ok=True)
    for generator in GENERATORS:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / generator),
             "--output", str(destination)],
            cwd=ROOT, check=True,
        )
    expected = {p.relative_to(ARCHIVE).as_posix() for p in ARCHIVE.rglob("*.png")}
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*.png")}
    missing = expected - actual
    extra = actual - expected
    if missing or extra or len(expected) != 138:
        raise ValueError(
            f"Image paths differ: expected={len(expected)}, actual={len(actual)}, "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    print(f"Rendered all {len(actual)} official image paths in {destination}")


if __name__ == "__main__":
    main()
