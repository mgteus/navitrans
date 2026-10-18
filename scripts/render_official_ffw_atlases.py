"""Render the three official FFW activation atlases into a separate folder.

These atlas images are produced by the HardPos single-token activation table,
despite their current placement in the Sargolini section of the official index.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/hardpos_ps_gridness_exploration"
NAMES = (
    "atlas_ffw1_linear_gauss_smooth.png",
    "atlas_ffw1_activation_gauss_smooth.png",
    "atlas_ffw2_linear_gauss_smooth.png",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    archive = (ROOT / "imgs/oficiais").resolve()
    if destination == archive or archive in destination.parents:
        raise ValueError("Output must be separate from imgs/oficiais")
    with tempfile.TemporaryDirectory(prefix="ffw-atlas-") as scratch:
        env = os.environ.copy()
        env["HARDPOS_ANALYSIS_ROOT"] = scratch
        env["HARDPOS_PS_GRID"] = str(ANALYSIS / "ps_grid_1001.npz")
        env["HARDPOS_ACTIVATIONS"] = str(ROOT / "legacy/analysis/HardPos_single_token_hidden_activations.parquet")
        subprocess.run(
            [sys.executable, str(ANALYSIS / "plot_ffw_atlases.py")],
            cwd=ROOT, env=env, check=True,
        )
        target_dir = destination / "sargolini/gpt_span/transformacoes"
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in NAMES:
            source = Path(scratch) / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, target_dir / name)
    print(f"Rendered {len(NAMES)} FFW atlases in {target_dir}")


if __name__ == "__main__":
    main()
