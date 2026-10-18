"""Mapas discretos de PS e sua suavização gaussiana para FFW1 e FFW2.

Executar: .nav/bin/python results/hardpos_ps_gridness_exploration/plot_ffw_atlases.py

A malha APD fornece o PS em cada coordenada. Cada PS treinado tem uma única
ativação da inferência com um token; visitas da trajetória não entram no cálculo.
O suavizamento segue o notebook HardPos: grade 250 x 250, sigma espacial 0.015,
kernel gaussiano normalizado pelo suporte e suporte mínimo de 1e-6.
"""

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('HARDPOS_ANALYSIS_ROOT', Path(__file__).resolve().parent))
GRID_SIZE = 250
SMOOTH_SIGMA = 0.015
MIN_SUPPORT = 1e-6
SIGMA_CELLS = SMOOTH_SIGMA * (GRID_SIZE - 1)

REPRESENTATIONS = (
    ("mlp_linear1", "Mapa da Ativação - Transformação Linear 1 [pré-ReLU]", 32, "ffw1_linear"),
    ("mlp_activation", "Mapa da Ativação - Transformação Linear 1 [pós-ReLU]", 32, "ffw1_activation"),
    ("mlp_linear2", "Mapa da Ativação - Transformação Linear 2", 16, "ffw2_linear"),
)


def render_atlas(maps, title, output, interpolation):
    n_dims = maps.shape[0]
    ncols = 8 if n_dims == 32 else 4
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.45 * ncols, 2.65 * nrows),
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0d0d0")
    for dim, ax in enumerate(np.ravel(axes)):
        if dim >= n_dims:
            ax.axis("off")
            continue
        arr = maps[dim]
        valid = arr[np.isfinite(arr)]
        if not len(valid):
            raise ValueError(f"Dimensão vazia {dim + 1}: {title}")
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1e-7
        ax.imshow(
            arr, origin="lower", extent=(0, 1, 0, 1),
            cmap=cmap, vmin=lo, vmax=hi, interpolation=interpolation,
        )
        ax.set_title(f"Dimensão {dim + 1:02d}", fontsize=9)
        ax.set(xticks=[], yticks=[], aspect="equal")
        for position in (0.2, 0.4, 0.6, 0.8):
            ax.axvline(position, color="#808080", linewidth=0.5, alpha=0.28, zorder=2)
            ax.axhline(position, color="#808080", linewidth=0.5, alpha=0.28, zorder=2)
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(output.name, maps.shape, "células válidas",
          round(float(np.isfinite(maps).mean()), 4))


def smooth_discrete_map(discrete, support):
    signal = gaussian_filter(
        np.nan_to_num(discrete, nan=0.0), SIGMA_CELLS,
        mode="constant", cval=0.0,
    )
    smoothed = np.full(discrete.shape, np.nan, dtype=np.float32)
    np.divide(signal, support, out=smoothed, where=support > MIN_SUPPORT)
    return smoothed


def main():
    with np.load(Path(os.environ.get("HARDPOS_PS_GRID", Path(__file__).resolve().parent / "ps_grid_1001.npz")), allow_pickle=False) as data:
        ps_grid = data["ps_id"]
    if ps_grid.shape != (1001, 1001):
        raise ValueError(f"Malha APD inesperada: {ps_grid.shape}")

    # A grade 250 x 250 amostra o PS exato da malha APD, sem média por área.
    fine_indices = np.rint(np.linspace(0, 1000, GRID_SIZE)).astype(int)
    sampled_ps = ps_grid[np.ix_(fine_indices, fine_indices)]

    table = pd.read_parquet(Path(os.environ.get('HARDPOS_ACTIVATIONS', ROOT / 'legacy/analysis/HardPos_single_token_hidden_activations.parquet')))
    table = table.sort_values("input_ps_id")
    ids = table["input_ps_id"].to_numpy(dtype=int)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("IDs de PS treinados não são consecutivos")

    valid_ps = sampled_ps < len(ids)
    support = gaussian_filter(
        valid_ps.astype(np.float32), SIGMA_CELLS,
        mode="constant", cval=0.0,
    )
    all_names, all_discrete, all_smoothed = [], [], []

    for name, label, n_dims, stage in REPRESENTATIONS:
        discrete_maps, smoothed_maps = [], []
        for dim in range(n_dims):
            column = f"{name}__d{dim:02d}"
            if column not in table:
                raise KeyError(column)
            values = table[column].to_numpy(dtype=np.float32)
            lookup = np.full(int(ps_grid.max()) + 1, np.nan, dtype=np.float32)
            lookup[ids] = values
            discrete = lookup[sampled_ps]
            discrete_maps.append(discrete)
            smoothed_maps.append(smooth_discrete_map(discrete, support))
            all_names.append(column)
        discrete_maps = np.stack(discrete_maps)
        smoothed_maps = np.stack(smoothed_maps)
        all_discrete.append(discrete_maps)
        all_smoothed.append(smoothed_maps)

        render_atlas(
            discrete_maps, label,
            OUT / f"atlas_{stage}_ps_discrete.png",
            "nearest",
        )
        render_atlas(
            smoothed_maps, label,
            OUT / f"atlas_{stage}_gauss_smooth.png",
            "bilinear",
        )

    np.savez_compressed(
        OUT / "ffw_maps_discrete_and_gaussian.npz",
        feature_names=np.asarray(all_names),
        discrete=np.concatenate(all_discrete),
        smoothed=np.concatenate(all_smoothed),
        support=support,
        grid_size=GRID_SIZE,
        smooth_sigma=SMOOTH_SIGMA,
        min_support=MIN_SUPPORT,
    )


if __name__ == "__main__":
    main()
