"""Paths and run identity for artifacts produced by model notebooks."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from numbers import Integral
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "model_runs"
ACTIVE = RUNS / "active.json"
VALID_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*\Z")


def _tag(model_tag: str) -> str:
    if not isinstance(model_tag, str) or not VALID_TAG.fullmatch(model_tag):
        raise ValueError(f"Invalid model_tag: {model_tag!r}")
    return model_tag


def _filename(filename: str) -> str:
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".csv"):
        raise ValueError(f"Invalid CSV filename: {filename!r}")
    return filename


def run_csv_dir(model_tag: str) -> Path:
    return RUNS / _tag(model_tag) / "csv"


def run_full_fwd_dir(model_tag: str) -> Path:
    return RUNS / _tag(model_tag) / "full_fwd"


def _manifest(model_tag: str) -> Path:
    return RUNS / _tag(model_tag) / "run.json"


def _check_manifest(model_tag: str) -> None:
    path = _manifest(model_tag)
    if not path.is_file():
        raise FileNotFoundError(f"Run manifest is missing: {path}")
    recorded = json.loads(path.read_text(encoding="utf-8")).get("model_tag")
    if recorded != model_tag:
        raise ValueError(f"Requested model_tag {model_tag!r}, manifest records {recorded!r}")


def _check_tagged_filename(model_tag: str, filename: str) -> None:
    exact = {
        f"cosine_similarity_matrix_gpt_pre_train_{model_tag}.csv",
        f"cosine_similarity_matrix_gpt_{model_tag}_post_train_.csv",
        f"embedded_repr_gpt_{model_tag}_post_train_.csv",
        f"{model_tag}_train_metrics.csv",
        f"{model_tag}softmax_fixed_output_single_token.csv",
    }
    epoch_prefix = f"cosine_similarity_matrix_gpt_{model_tag}_train_epoch_"
    distribution_prefix = f"{model_tag}_gpt_"
    valid_epoch = filename.startswith(epoch_prefix) and filename[len(epoch_prefix):-4].isdigit()
    valid_distribution = filename.startswith(distribution_prefix) and filename.endswith(".csv")
    if filename not in exact and not valid_epoch and not valid_distribution:
        raise ValueError(f"CSV {filename!r} does not belong to model_tag {model_tag!r}")


def _check_analysis_filename(model_tag: str, filename: str) -> None:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid analysis filename: {filename!r}")
    valid = {
        f"{model_tag}single_token_hidden_activations.parquet",
        f"{model_tag}single_token_spatial_maps.npz",
    }
    if filename not in valid:
        raise ValueError(f"Analysis {filename!r} does not belong to model_tag {model_tag!r}")


def save_analysis_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_analysis_filename(model_tag, filename)
    directory = RUNS / model_tag / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(model_tag)
    if manifest.exists():
        _check_manifest(model_tag)
    else:
        manifest.write_text(json.dumps({"model_tag": model_tag}, indent=2) + "\n", encoding="utf-8")
    return directory / filename


def load_analysis_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_analysis_filename(model_tag, filename)
    _check_manifest(model_tag)
    path = RUNS / model_tag / "analysis" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _check_generation_filename(model_tag: str, filename: str) -> None:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid generation filename: {filename!r}")
    prefix = f"{model_tag}_gpt_generation_tokens_"
    if not filename.startswith(prefix) or not filename.endswith(".pkl") or not filename[len(prefix):-4].isdigit():
        raise ValueError(f"Generation {filename!r} does not belong to model_tag {model_tag!r}")


def save_generation_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_generation_filename(model_tag, filename)
    directory = RUNS / model_tag / "generations"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(model_tag)
    if manifest.exists():
        _check_manifest(model_tag)
    else:
        manifest.write_text(json.dumps({"model_tag": model_tag}, indent=2) + "\n", encoding="utf-8")
    return directory / filename


def load_generation_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_generation_filename(model_tag, filename)
    _check_manifest(model_tag)
    path = RUNS / model_tag / "generations" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _check_checkpoint_filename(model_tag: str, filename: str) -> None:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid checkpoint filename: {filename!r}")
    valid = {f"gpt_model_{model_tag}.pt"}
    if filename not in valid and not (filename.startswith(f"gpt_model_{model_tag}epoch_") and filename.endswith(".pt") and filename[len(f"gpt_model_{model_tag}epoch_"):-3].isdigit()):
        raise ValueError(f"Checkpoint {filename!r} does not belong to model_tag {model_tag!r}")


def save_checkpoint_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_checkpoint_filename(model_tag, filename)
    directory = RUNS / model_tag / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(model_tag)
    if manifest.exists():
        _check_manifest(model_tag)
    else:
        manifest.write_text(json.dumps({"model_tag": model_tag}, indent=2) + "\n", encoding="utf-8")
    return directory / filename


def load_checkpoint_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_checkpoint_filename(model_tag, filename)
    _check_manifest(model_tag)
    path = RUNS / model_tag / "checkpoints" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _check_full_fwd_filename(model_tag: str, filename: str) -> None:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid full_fwd filename: {filename!r}")
    exact = {f"full_fwd_{kind}_{model_tag}" for kind in ("input", "target", "pre_train")}
    exact.add(f"full_fwd_{model_tag}")
    epoch_prefix = "full_fwd_train_epoch_"
    valid_epoch = filename.startswith(epoch_prefix) and filename.endswith(f"_{model_tag}") and filename[len(epoch_prefix):-(len(model_tag) + 1)].isdigit()
    if filename not in exact and not valid_epoch:
        raise ValueError(f"full_fwd {filename!r} does not belong to model_tag {model_tag!r}")


def save_full_fwd_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_full_fwd_filename(model_tag, filename)
    directory = run_full_fwd_dir(model_tag)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(model_tag)
    if manifest.exists():
        _check_manifest(model_tag)
    else:
        manifest.write_text(json.dumps({"model_tag": model_tag}, indent=2) + "\n", encoding="utf-8")
    return directory / filename


def load_full_fwd_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    _check_full_fwd_filename(model_tag, filename)
    _check_manifest(model_tag)
    path = run_full_fwd_dir(model_tag) / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_csv_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    filename = _filename(filename)
    _check_tagged_filename(model_tag, filename)
    directory = run_csv_dir(model_tag)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(model_tag)
    if manifest.exists():
        _check_manifest(model_tag)
    else:
        manifest.write_text(json.dumps({"model_tag": model_tag}, indent=2) + "\n", encoding="utf-8")
    return directory / filename


def load_csv_path(model_tag: str, filename: str) -> Path:
    model_tag = _tag(model_tag)
    filename = _filename(filename)
    _check_tagged_filename(model_tag, filename)
    _check_manifest(model_tag)
    path = run_csv_dir(model_tag) / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def register_ps_mapping(model_tag: str, mapping: dict) -> Path:
    """Bind a HardPos run to one PS-to-ID convention."""
    model_tag = _tag(model_tag)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("PS mapping must be a nonempty dictionary")
    ids = list(mapping.values())
    if not all(isinstance(value, Integral) and value >= 0 for value in ids):
        raise ValueError("PS IDs must be nonnegative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("PS IDs must be unique")
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("PS keys must be strings")
    directory = RUNS / model_tag
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(model_tag)
    if manifest.exists():
        _check_manifest(model_tag)
        data = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        data = {"model_tag": model_tag}
    path = directory / "ps_to_psid.pkl"
    payload = pickle.dumps(mapping, protocol=4)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Run {model_tag!r} already has a different PS mapping: {path}")
    recorded = data.get("ps_mapping_sha256")
    if recorded is not None and recorded != digest:
        raise ValueError(f"Run {model_tag!r} records a different PS mapping")
    path.write_bytes(payload)
    data["ps_mapping_sha256"] = digest
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_ps_mapping(model_tag: str) -> dict:
    model_tag = _tag(model_tag)
    _check_manifest(model_tag)
    path = RUNS / model_tag / "ps_to_psid.pkl"
    payload = path.read_bytes()
    recorded = json.loads(_manifest(model_tag).read_text(encoding="utf-8")).get("ps_mapping_sha256")
    if recorded != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"PS mapping hash mismatch for {model_tag!r}")
    return pickle.loads(payload)


def register_active_run(group: str, model_tag: str) -> None:
    if group not in ("fixed", "sargolini"):
        raise ValueError(f"Unknown run group: {group!r}")
    model_tag = _tag(model_tag)
    _check_manifest(model_tag)
    if group == "fixed":
        load_ps_mapping(model_tag)
    required = (
        f"cosine_similarity_matrix_gpt_pre_train_{model_tag}.csv",
        f"cosine_similarity_matrix_gpt_{model_tag}_post_train_.csv",
        f"{model_tag}_train_metrics.csv",
    )
    missing = [name for name in required if not (run_csv_dir(model_tag) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run {model_tag!r} is incomplete: {missing}")
    current = json.loads(ACTIVE.read_text(encoding="utf-8")) if ACTIVE.exists() else {}
    current[group] = model_tag
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def active_model_tag(group: str) -> str:
    if group not in ("fixed", "sargolini"):
        raise ValueError(f"Unknown run group: {group!r}")
    if not ACTIVE.is_file():
        raise FileNotFoundError(f"No active model run is registered: {ACTIVE}")
    model_tag = json.loads(ACTIVE.read_text(encoding="utf-8")).get(group)
    model_tag = _tag(model_tag)
    _check_manifest(model_tag)
    return model_tag


def notebook_run_path(group: str, filename: str) -> Path:
    group = _tag(group)
    if not isinstance(filename, str) or Path(filename).name != filename or not re.fullmatch(r"run_[0-9_]+\.pkl", filename):
        raise ValueError(f"Invalid notebook run filename: {filename!r}")
    directory = ROOT / "artifacts" / "notebooks" / group / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def notebook_csv_path(group: str, filename: str) -> Path:
    """Organized output for notebooks without a model_tag."""
    group = _tag(group)
    filename = _filename(filename)
    directory = ROOT / "artifacts" / "notebooks" / group / "csv"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def load_notebook_csv_path(group: str, filename: str) -> Path:
    """Read a new notebook output, or a pre-migration CSV from legacy."""
    group = _tag(group)
    filename = _filename(filename)
    organized = ROOT / "artifacts" / "notebooks" / group / "csv" / filename
    if organized.is_file():
        return organized
    legacy = ROOT / "legacy" / "csv" / filename
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"CSV not found in {organized} or {legacy}")
