"""Rerender 22 official GPT Sargolini figures from saved numerical results.

The plotting cells are taken from the original training notebook. Training
metrics, cosine similarities, and full forward states are loaded from files,
so the 2,000-epoch model training is not repeated to redraw the figures.
Output names and source cell numbers come from the official image index.
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
NOTEBOOK = ROOT / "navitran_vizz_reboot_gpt_squared_25_span_sargolini.ipynb"
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
    if len(result) != 22:
        raise ValueError(f"Expected 22 GPT images, found {len(result)}")
    return result


def build_plot_notebook(source):
    cells = []
    positions = {}

    def add_source(number):
        positions[number] = len(cells)
        code = source.cells[number].source
        if number == 8:
            code = code.replace(
                "/home/mgteus/workspace/navitrans/runs/datasets/sargolini/sargolini_trainset_squared_25.gzip",
                "runs/datasets/sargolini/sargolini_trainset_squared_25.gzip",
            )
        cells.append(nbformat.v4.new_code_cell(code))

    def add_bootstrap(code):
        cells.append(nbformat.v4.new_code_cell(code))

    for number in (1, 8, 11, 15):
        add_source(number)
    add_bootstrap("""
from project_support.csv_runs import active_model_tag, load_csv_path, load_full_fwd_path
model_tag = active_model_tag('sargolini')
metrics = pd.read_csv(load_csv_path(model_tag, f'{model_tag}_train_metrics.csv'))
num_epochs = len(metrics)
train_losses = metrics['train_loss'].to_list()
val_losses = metrics['val_loss'].to_list()
""")
    for number in (55, 56):
        add_source(number)
    add_bootstrap("""
sim_matrix = torch.tensor(
    pd.read_csv(load_csv_path(model_tag, f'cosine_similarity_matrix_gpt_{model_tag}_post_train_.csv')).to_numpy()
)
""")
    add_source(73)
    add_bootstrap("""
with open(load_full_fwd_path(model_tag, f'full_fwd_{model_tag}'), 'rb') as source_file:
    full_fwd = pickle.load(source_file)
""")
    for number in range(106, 143):
        if source.cells[number].cell_type == "code" and source.cells[number].source.strip():
            add_source(number)
    return nbformat.v4.new_notebook(cells=cells), positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    archive = INDEX.parent.resolve()
    if destination == archive or archive in destination.parents:
        raise ValueError("Output must be separate from imgs/oficiais")
    images = mapping()
    source = nbformat.read(NOTEBOOK, as_version=4)
    notebook, positions = build_plot_notebook(source)
    if not set(images).issubset(positions):
        raise ValueError(f"Missing plotting cells: {set(images) - set(positions)}")
    NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    ).execute()
    for cell_number, relative in images.items():
        pngs = []
        for output in notebook.cells[positions[cell_number]].get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                pngs.append(base64.b64decode("".join(encoded)))
        if len(pngs) != 1:
            raise ValueError(f"Cell {cell_number}: expected one PNG, found {len(pngs)}")
        target = destination / "sargolini" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pngs[0])
    print(f"Rendered {len(images)} GPT Sargolini images in {destination / 'sargolini'}")


if __name__ == "__main__":
    main()
