"""Rerender 34 official Sargolini HardPos and exported figures in a separate folder.

Run from the repository root:
    .nav/bin/python scripts/render_official_sargolini_viz.py --output outputs/official-rerender

This executes the visualization notebook after clearing its saved outputs.
Names and cell numbers for the 30 HardPos figures come from the official index.
Four exported progression figures use their original plotting cells.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "imgs/oficiais/sargolini/INDICE.md"
ARCHIVE = INDEX.parent
NOTEBOOK = ROOT / "apd_vizz_gpt_squared_25 HardPos_Sargolini.ipynb"
PATH_IN_ROW = re.compile(r"^\| \[[^]]+\]\(([^)]+)\)")


def mapping() -> dict[int, Path]:
    result = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ["):
            continue
        parts = line.split("|")
        if len(parts) < 5 or parts[2].strip().strip("`") != NOTEBOOK.name:
            continue
        match = PATH_IN_ROW.match(line)
        if match is None:
            raise ValueError(f"Invalid index row: {line}")
        cell_number = int(parts[3].strip())
        if cell_number in result:
            raise ValueError(f"Duplicate notebook cell {cell_number}")
        relative = Path(match.group(1))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid image path: {relative}")
        result[cell_number] = relative
    if len(result) != 30:
        raise ValueError(f"Expected 30 Sargolini HardPos figures, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination == ARCHIVE.resolve() or ARCHIVE.resolve() in destination.parents:
        raise ValueError("Output must be separate from imgs/oficiais")
    images = mapping()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    ).execute()
    for cell_number, relative in images.items():
        pngs = []
        for output in notebook.cells[cell_number].get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                pngs.append(base64.b64decode("".join(encoded)))
        if len(pngs) != 1:
            raise ValueError(f"Cell {cell_number}: expected one PNG, found {len(pngs)}")
        target = destination / "sargolini" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pngs[0])
    exported = destination / "sargolini/exportadas"
    exported.mkdir(parents=True, exist_ok=True)

    def plot_bytes(cell_number):
        pngs = []
        for output in notebook.cells[cell_number].get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                pngs.append(base64.b64decode("".join(encoded)))
        if len(pngs) != 1:
            raise ValueError(f"Cell {cell_number}: expected one PNG, found {len(pngs)}")
        return pngs[0]

    (exported / "sargolini_prob_pre_treino.png").write_bytes(plot_bytes(110))
    (exported / "sargolini_prob_pos_treino_epoca_2000.png").write_bytes(plot_bytes(119))
    for name, cells in (
        ("sargolini_probs_training_progression.png", (110, 112, 115, 117, 119)),
        ("sargolini_logits_training_progression.png", (139, 141, 143, 145, 147)),
    ):
        panels = [Image.open(io.BytesIO(plot_bytes(cell))).convert("RGB") for cell in cells]
        width = max(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width * len(panels), height), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * width, 0))
        canvas.save(exported / name)
    print(f"Rendered {len(images) + 4} Sargolini visualization images in {destination / 'sargolini'}")


if __name__ == "__main__":
    main()
