"""Rerender the 12 official Sargolini environment figures in a separate folder.

The BERT data-preparation notebook is executed from code. Its dataset-writing
switch is disabled in the in-memory execution copy, so existing input files
are not overwritten. Output names come from the official image index.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "imgs/oficiais/sargolini/INDICE.md"
NOTEBOOK = ROOT / "start_v0_bert_squared_sargolini.ipynb"
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
        relative = Path(match.group(1))
        if relative.is_absolute() or ".." in relative.parts or cell_number in result:
            raise ValueError(f"Invalid index entry: {line}")
        result[cell_number] = relative
    if len(result) != 12:
        raise ValueError(f"Expected 12 BERT images, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    archive = INDEX.parent.resolve()
    if destination == archive or archive in destination.parents:
        raise ValueError("Output must be separate from imgs/oficiais")
    images = mapping()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    if "save_dfc = True" not in notebook.cells[147].source:
        raise ValueError("The dataset-writing switch moved from cell 147")
    notebook.cells[147].source = notebook.cells[147].source.replace(
        "save_dfc = True", "save_dfc = False"
    )
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
    print(f"Rendered {len(images)} BERT Sargolini images in {destination / 'sargolini'}")


if __name__ == "__main__":
    main()
