"""Render the 67 official HardPos/fixed figures into a separate folder.

Run from the repository root with the project environment:
    .nav/bin/python scripts/render_official_fixed.py --output /tmp/official-images

The source visualization notebook is executed from code after clearing saved
outputs. Its recomputed figures are named according to the official index.
Two other figures are rendered from the training dataset and metrics CSV.
The archive in imgs/oficiais is never read as an image source or overwritten.
"""

from __future__ import annotations

import argparse
import base64
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from nbclient import NotebookClient



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_support.csv_runs import active_model_tag, load_csv_path
SOURCE = ROOT / "apd_vizz_gpt_squared_25 HardPos.ipynb"
ARCHIVE = ROOT / "imgs/oficiais/fixed"


def image_outputs(cell) -> list[bytes]:
    images = []
    for output in cell.get("outputs", []):
        encoded = output.get("data", {}).get("image/png")
        if encoded:
            images.append(base64.b64decode("".join(encoded)))
    return images


def map_notebook_images() -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {
        22: ["matrizes_similaridade_ps/similaridade_ps_pre_treino.png"],
        29: ["matrizes_similaridade_ps/similaridade_ps_treino_250.png"],
        32: ["matrizes_similaridade_ps/similaridade_ps_pos_treino.png"],
    }
    for cell, stage in [(78, "pre_treino"), (83, "treino_250"), (90, "pos_treino")]:
        for offset, ps in enumerate((200, 705, 80, 930)):
            mapping[cell + offset] = [
                f"mapas_similaridade_ps/similaridade_ps{ps}_{stage}.png"
            ]
    stages = [
        (115, "pre_treino"), (117, "treino_100"), (120, "treino_200"),
        (122, "treino_500"), (124, "pos_treino"),
    ]
    for cell, stage in stages:
        mapping[cell] = [
            f"evolucao_ps576_passo01/probabilidade_ps576_passo01_{stage}.png"
        ]
    stages = [
        (158, "pre_treino"), (160, "treino_100"), (162, "treino_200"),
        (164, "treino_500"), (166, "pos_treino"),
    ]
    for cell, stage in stages:
        mapping[cell] = [f"evolucao_ps576_passo01/logits_ps576_passo01_{stage}.png"]
    sequences = {
        "681": (681, 685, 684, 892, 884, 418, 417, 884, 885, 420),
        "507": (507, 576, 505, 575, 576, 568, 570, 569, 526, 528),
    }
    for offset, ps in enumerate(sequences["681"]):
        step = offset + 1
        mapping[127 + offset] = [
            f"probabilidades_por_passo/probabilidade_passo_{step:02d}_ps{ps}.png"
        ]
        mapping[168 + offset] = [
            f"logits_por_passo/logits_passo_{step:02d}_ps{ps}.png"
        ]
    mapping[140] = [
        f"probabilidades_por_passo/probabilidade_passo_{step:02d}_ps{ps}.png"
        for step, ps in enumerate(sequences["507"], 1)
    ]
    mapping[179] = [
        f"logits_por_passo/logits_passo_{step:02d}_ps{ps}.png"
        for step, ps in enumerate(sequences["507"], 1)
    ]
    return mapping


def render_visits(output: Path) -> None:
    df = pd.read_parquet(ROOT / "bert_trainset_squared_25.gzip")
    with (ROOT / "ps_to_psid.pkl").open("rb") as source:
        ps_to_psid = pickle.load(source)
    state_ids = df["position_state"].map(ps_to_psid)
    if state_ids.isna().any():
        raise ValueError("Unmapped position states in the HardPos training dataset")
    counts = state_ids.value_counts()
    weights = state_ids.map(counts).to_numpy(float)
    xy = np.stack(df["pos"].to_numpy()).astype(float)
    grid_size = 250
    hist = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=grid_size, range=((0, 1), (0, 1)),
        weights=weights,
    )[0]
    support = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=grid_size, range=((0, 1), (0, 1)),
    )[0]
    sigma = 0.015 * (grid_size - 1)
    smooth_sum = gaussian_filter(hist, sigma=sigma, mode="constant")
    smooth_support = gaussian_filter(support, sigma=sigma, mode="constant")
    field = np.divide(
        smooth_sum, smooth_support, out=np.zeros_like(smooth_sum),
        where=smooth_support > 1e-6,
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        field, extent=(0, 1, 0, 1), origin="lower", cmap="Blues",
        vmin=1, vmax=float(counts.max()), interpolation="bilinear",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="X", ylabel="Y",
           title="Mapa de Frequência de Visitas", aspect="equal")
    ax.grid(True, alpha=0.7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# de visitas")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)

def render_metrics(output: Path) -> None:
    model_tag = active_model_tag("fixed")
    metrics = pd.read_csv(load_csv_path(model_tag, f"{model_tag}_train_metrics.csv"))
    required = {"epoch", "train_loss", "val_loss"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Training metrics missing columns: {required - set(metrics)}")
    epoch = metrics["epoch"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(epoch, train_loss, label="Train Loss")
    ax.plot(epoch, val_loss, label="Validation Loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training and Validation Loss")
    ax.legend()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(epoch, np.exp(train_loss), label="Train Perplexity")
    ax.plot(epoch, np.exp(val_loss), label="Validation Perplexity")
    ax.set(xlabel="Epoch", ylabel="Perplexity", title="Training and Validation Perplexity")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination == ARCHIVE.resolve() or destination == ARCHIVE.parent.resolve() or ARCHIVE.resolve() in destination.parents:
        raise ValueError("Output must be separate from imgs/oficiais")
    mapping = map_notebook_images()
    expected = {p.relative_to(ARCHIVE).as_posix() for p in ARCHIVE.rglob("*.png")}
    assigned = {name for names in mapping.values() for name in names}
    assigned.update({
        "frequencia_visitas/mapa_frequencia_visitas.png",
        "metricas_treinamento/loss_perplexidade_treino_validacao.png",
    })
    if assigned != expected:
        raise ValueError(f"Filename mismatch: missing={expected-assigned}, extra={assigned-expected}")
    notebook = nbformat.read(SOURCE, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    for cell_number, names in mapping.items():
        images = image_outputs(notebook.cells[cell_number])
        if len(images) != len(names):
            raise ValueError(f"Cell {cell_number}: {len(images)} images for {len(names)} names")
        for name, data in zip(names, images):
            target = destination / "fixed" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    render_visits(destination / "fixed/frequencia_visitas/mapa_frequencia_visitas.png")
    render_metrics(destination / "fixed/metricas_treinamento/loss_perplexidade_treino_validacao.png")
    generated = list((destination / "fixed").rglob("*.png"))
    if len(generated) != 67:
        raise ValueError(f"Expected 67 fixed figures, found {len(generated)}")
    print(f"Rendered 67 fixed images in {destination / 'fixed'}")


if __name__ == "__main__":
    main()
