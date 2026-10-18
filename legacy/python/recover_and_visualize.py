from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nbformat

from run_embed_sweep import (
    ROOT, RUN_ROOT, VIZ_SOURCE, export_images, now_utc,
    run_notebook, validate_training, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-size", type=int, required=True, choices=(8, 4, 2))
    args = parser.parse_args()
    embed_size = args.embed_size
    run_dir = RUN_ROOT / f"embed_{embed_size}"
    params_path = run_dir / "parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    tag = params["model_tag"]
    os.environ["EMBED_SIZE"] = str(embed_size)
    os.environ["MODEL_TAG"] = tag
    os.environ["PATH"] = str(ROOT / ".nav" / "bin") + os.pathsep + os.environ["PATH"]

    failed_path = run_dir / "train_executed_failed.ipynb"
    train_output = run_dir / "train_executed.ipynb"
    if not failed_path.exists():
        raise FileNotFoundError(failed_path)
    training = validate_training(run_dir, tag, embed_size)
    required = (
        f"full_fwd_{tag}",
        f"full_fwd_pre_train_{tag}",
        f"embedded_repr_gpt_{tag}_post_train_.csv",
        f"cosine_similarity_matrix_gpt_{tag}_post_train_.csv",
        f"{tag}_gpt_probs_distribs_681.csv",
        f"{tag}_gpt_probs_distribs_507.csv",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Training artifacts missing: {missing}")

    notebook = nbformat.read(failed_path, as_version=4)
    errors = [(i, cell.source.strip()) for i, cell in enumerate(notebook.cells)
              if cell.cell_type == "code" and any(
                  output.output_type == "error" for output in cell.get("outputs", []))]
    if len(errors) != 1 or errors[0][1] != "x[7]":
        raise RuntimeError(f"Unexpected training notebook error: {errors}")
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.strip() in ("x[7]", "y[7]"):
            cell.source = "x.shape" if cell.source.strip() == "x[7]" else "y.shape"
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata["recovery_note"] = (
        "1000 training epochs and all persisted artifacts completed. "
        "Two trailing exploratory index probes were replaced by shape probes "
        "after x[7] raised IndexError."
    )
    nbformat.write(notebook, train_output)
    training_images = export_images(notebook, run_dir, "train")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"] = training
    manifest["training_images"] = training_images
    manifest["training_notebook_note"] = notebook.metadata["recovery_note"]
    manifest["status"] = "visualizing"
    write_json(manifest_path, manifest)

    viz_output = run_dir / "visualization_executed.ipynb"
    if not viz_output.exists():
        manifest["visualization_images"] = run_notebook(
            VIZ_SOURCE, viz_output, run_dir, "visualization"
        )
    manifest["status"] = "complete"
    manifest["completed_at"] = now_utc()
    write_json(manifest_path, manifest)
    print(json.dumps({
        "embed_size": embed_size,
        "status": manifest["status"],
        "training": training,
        "visualization_images": manifest["visualization_images"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
