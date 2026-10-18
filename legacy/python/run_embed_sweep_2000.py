from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np
import torch
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "embed_sweep_context10_epochs2000"
TRAIN_SOURCE = ROOT / "navitran_vizz_reboot_gpt_squared_25_span.ipynb"
VIZ_SOURCE = ROOT / "apd_vizz_gpt_squared_25 HardPos_regioes_discretas.ipynb"
INPUTS = ("bert_trainset_squared_25.gzip", "ps_to_psid.pkl", "legacy/datasets/apd_full.gzip")
SEED = 42


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(embed_size: int) -> tuple[Path, str, dict]:
    run_dir = RUN_ROOT / f"embed_{embed_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        source = ROOT / name
        if not source.exists():
            raise FileNotFoundError(source)
        link = run_dir / name
        if not link.exists():
            link.symlink_to(source)
    tag = f"HardPos_embed{embed_size}_context10_epochs2000_"
    params_path = run_dir / "parameters.json"
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        assert params["embed_size"] == embed_size and params["model_tag"] == tag
        return run_dir, tag, params
    params = {
        "embed_size": embed_size,
        "d_model": embed_size,
        "hidden_dim": embed_size * 2,
        "ffw1_out": embed_size * 2,
        "ffw2_out": embed_size,
        "context_len": 10,
        "max_len": 10,
        "dataset_window_len": 11,
        "num_epochs": 2000,
        "num_layers": 1,
        "num_heads": 1,
        "dropout": 0.2,
        "batch_size": 256,
        "learning_rate": 0.0001,
        "seed": SEED,
        "model_tag": tag,
        "train_notebook": TRAIN_SOURCE.name,
        "visualization_notebook": VIZ_SOURCE.name,
        "source_sha256": {
            TRAIN_SOURCE.name: sha256(TRAIN_SOURCE),
            VIZ_SOURCE.name: sha256(VIZ_SOURCE),
        },
        "input_sha256": {name: sha256(ROOT / name) for name in INPUTS},
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(run_dir / "parameters.json", params)
    return run_dir, tag, params


def export_images(notebook: nbformat.NotebookNode, run_dir: Path, stage: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    index_rows = []
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output_index, output in enumerate(cell.get("outputs", [])):
            raw = output.get("data", {}).get("image/png")
            if not raw:
                continue
            if isinstance(raw, list):
                raw = "".join(raw)
            if stage == "train":
                category = "treino"
            else:
                category = "discreto" if cell_index >= 179 else "gaussiano"
            folder = run_dir / "images" / category
            folder.mkdir(parents=True, exist_ok=True)
            filename = f"cell_{cell_index:03d}_image_{output_index:02d}.png"
            target = folder / filename
            target.write_bytes(base64.b64decode(raw))
            counts[category] = counts.get(category, 0) + 1
            index_rows.append({
                "category": category,
                "cell": cell_index,
                "output": output_index,
                "file": str(target.relative_to(run_dir)),
                "source_preview": cell.source.strip().splitlines()[0][:160] if cell.source.strip() else "",
            })
    if index_rows:
        index_path = run_dir / f"{stage}_image_index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=index_rows[0])
            writer.writeheader()
            writer.writerows(index_rows)
    return counts


def run_notebook(source: Path, output: Path, run_dir: Path, stage: str) -> dict[str, int]:
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    if stage == "train":
        seed_cell = nbformat.v4.new_code_cell(
            "import os, random, numpy as np, torch\n"
            f"random.seed({SEED})\nnp.random.seed({SEED})\n"
            f"torch.manual_seed({SEED})\ntorch.cuda.manual_seed_all({SEED})"
        )
        notebook.cells.insert(0, seed_cell)
        for cell in notebook.cells:
            if cell.cell_type == "code" and "for epoch in range(int(num_epochs)):" in cell.source:
                cell.source = cell.source.replace(
                    "for epoch in range(int(num_epochs)):\n",
                    "for epoch in range(int(num_epochs)):\n"
                    "    if epoch % 100 == 0:\n"
                    "        with open('epoch_progress.txt', 'w') as _progress:\n"
                    "            _progress.write(str(epoch + 1))\n",
                    1,
                )
                break
        else:
            raise RuntimeError("Training loop not found")
    started = time.monotonic()
    print(f"{now_utc()} START {stage} embed_size={os.environ['EMBED_SIZE']}", flush=True)
    try:
        client = NotebookClient(
            notebook,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(run_dir)}},
            allow_errors=False,
        )
        client.execute()
    except Exception as exc:
        nbformat.write(notebook, output.with_name(output.stem + "_failed.ipynb"))
        write_json(run_dir / f"{stage}_error.json", {
            "stage": stage, "error": repr(exc), "at": now_utc(),
        })
        raise
    nbformat.write(notebook, output)
    counts = export_images(notebook, run_dir, stage)
    print(f"{now_utc()} END {stage} embed_size={os.environ['EMBED_SIZE']} "
          f"elapsed_s={time.monotonic()-started:.1f} images={counts}", flush=True)
    return counts


def validate_training(run_dir: Path, tag: str, embed_size: int) -> dict:
    model_path = run_dir / f"gpt_model_{tag}.pt"
    metrics_path = run_dir / f"{tag}_train_metrics.csv"
    if not model_path.exists() or not metrics_path.exists():
        raise RuntimeError("Model or training metrics missing")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    embedding_shape = list(state["embedding.weight"].shape)
    ffw1_shape = list(state["encoder.layers.0.linear1.weight"].shape)
    ffw2_shape = list(state["encoder.layers.0.linear2.weight"].shape)
    assert embedding_shape[1] == embed_size, embedding_shape
    assert ffw1_shape == [2 * embed_size, embed_size], ffw1_shape
    assert ffw2_shape == [embed_size, 2 * embed_size], ffw2_shape
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 2000, len(metrics)
    return {
        "model": model_path.name,
        "model_sha256": sha256(model_path),
        "embedding_shape": embedding_shape,
        "ffw1_shape": ffw1_shape,
        "ffw2_shape": ffw2_shape,
        "epochs_completed": len(metrics),
        "final_metrics": metrics[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-size", type=int, required=True, choices=(8, 4, 2))
    args = parser.parse_args()
    embed_size = args.embed_size
    previous_manifest = RUN_ROOT / f"embed_{embed_size}" / "manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            print(f"embed_size={embed_size} already complete: {previous_manifest}", flush=True)
            return
    run_dir, tag, params = prepare(embed_size)
    os.environ["NUM_EPOCHS"] = "2000"
    os.environ["EMBED_SIZE"] = str(embed_size)
    os.environ["MODEL_TAG"] = tag
    os.environ["PATH"] = str(ROOT / ".nav" / "bin") + os.pathsep + os.environ["PATH"]
    manifest_path = run_dir / "manifest.json"
    manifest = {"parameters": params, "started_at": now_utc(), "status": "training"}
    write_json(manifest_path, manifest)
    train_output = run_dir / "train_executed.ipynb"
    viz_output = run_dir / "visualization_executed.ipynb"
    if not train_output.exists():
        manifest["training_images"] = run_notebook(TRAIN_SOURCE, train_output, run_dir, "train")
    manifest["training"] = validate_training(run_dir, tag, embed_size)
    manifest["status"] = "visualizing"
    write_json(manifest_path, manifest)
    if not viz_output.exists():
        manifest["visualization_images"] = run_notebook(VIZ_SOURCE, viz_output, run_dir, "visualization")
    else:
        manifest["visualization_images"] = {
            p.name: len(list(p.glob("*.png"))) for p in (run_dir / "images").iterdir()
            if p.is_dir()
        }
    manifest["status"] = "complete"
    manifest["completed_at"] = now_utc()
    write_json(manifest_path, manifest)
    print(json.dumps({"embed_size": embed_size, "status": "complete",
                      "training": manifest["training"],
                      "visualization_images": manifest["visualization_images"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
