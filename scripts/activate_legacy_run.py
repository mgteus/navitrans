"""Activate an existing legacy CSV run without retraining its model.

Example:
    .nav/bin/python scripts/activate_legacy_run.py \
        --group fixed --model-tag HardPos_embed16_context10_epochs1000_
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_support.csv_runs import register_ps_mapping, register_active_run, save_csv_path, save_full_fwd_path, save_checkpoint_path, save_generation_path, save_analysis_path

NAMES = {
    "fixed": (
        "embedded_repr_gpt_{tag}_post_train_.csv",
        "cosine_similarity_matrix_gpt_pre_train_{tag}.csv",
        "cosine_similarity_matrix_gpt_{tag}_train_epoch_199.csv",
        "cosine_similarity_matrix_gpt_{tag}_post_train_.csv",
        "{tag}_gpt_probs_distribs_681.csv",
        "{tag}_gpt_probs_distribs_507.csv",
        "{tag}_gpt_logits_distribs_681.csv",
        "{tag}_gpt_logits_distribs_507.csv",
        "{tag}_gpt_logits_raw_distribs_681.csv",
        "{tag}softmax_fixed_output_single_token.csv",
        "{tag}_train_metrics.csv",
    ),
    "sargolini": (
        "embedded_repr_gpt_{tag}_post_train_.csv",
        "cosine_similarity_matrix_gpt_pre_train_{tag}.csv",
        "cosine_similarity_matrix_gpt_{tag}_train_epoch_199.csv",
        "cosine_similarity_matrix_gpt_{tag}_post_train_.csv",
        "{tag}_gpt_probs_distribs_383.csv",
        "{tag}_gpt_logits_distribs_383.csv",
        "{tag}_gpt_logits_raw_distribs_383.csv",
        "{tag}_train_metrics.csv",
    ),
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(NAMES), required=True)
    parser.add_argument("--model-tag", required=True)
    args = parser.parse_args()
    names = [template.format(tag=args.model_tag) for template in NAMES[args.group]]
    sources = [ROOT / "legacy" / "csv" / name for name in names]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing legacy CSVs: {missing}")
    for source in sources:
        target = save_csv_path(args.model_tag, source.name)
        if target.exists():
            if digest(source) != digest(target):
                raise FileExistsError(f"Different CSV already exists: {target}")
        else:
            shutil.copy2(source, target)
    full_fwd_sources = sorted((ROOT / "legacy" / "full_fwd" / args.model_tag).glob("full_fwd*"))
    for source in full_fwd_sources:
        target = save_full_fwd_path(args.model_tag, source.name)
        if target.exists():
            if digest(source) != digest(target):
                raise FileExistsError(f"Different full_fwd already exists: {target}")
        else:
            shutil.copy2(source, target)
    checkpoint_sources = sorted((ROOT / "legacy" / "checkpoints" / args.model_tag).glob("gpt_model_*.pt"))
    for source in checkpoint_sources:
        target = save_checkpoint_path(args.model_tag, source.name)
        if target.exists():
            if digest(source) != digest(target):
                raise FileExistsError(f"Different checkpoint already exists: {target}")
        else:
            shutil.copy2(source, target)
    generation_sources = sorted((ROOT / "legacy" / "generations" / args.model_tag).glob("*_gpt_generation_tokens_*.pkl"))
    for source in generation_sources:
        target = save_generation_path(args.model_tag, source.name)
        if target.exists():
            if digest(source) != digest(target):
                raise FileExistsError(f"Different generation already exists: {target}")
        else:
            shutil.copy2(source, target)
    analysis_sources = [ROOT / "legacy" / "analysis" / f"{args.model_tag}{suffix}" for suffix in (
        "single_token_hidden_activations.parquet", "single_token_spatial_maps.npz"
    )]
    analysis_sources = [source for source in analysis_sources if source.is_file()]
    for source in analysis_sources:
        target = save_analysis_path(args.model_tag, source.name)
        if target.exists():
            if digest(source) != digest(target):
                raise FileExistsError(f"Different analysis already exists: {target}")
        else:
            shutil.copy2(source, target)
    if args.group == "fixed":
        # The archived 948-column matrices used compact ngroup IDs, unlike
        # the current root runs/datasets/hardpos/ps_to_psid.pkl. Keep that convention with this run.
        import pandas as pd
        dataset = pd.read_parquet(ROOT / "runs/datasets/hardpos/bert_trainset_squared_25.gzip")
        states = dataset["position_state"]
        compact_ids = states.groupby(states).ngroup()
        legacy_mapping = {
            state: int(ps_id) for state, ps_id in zip(states, compact_ids)
        }
        register_ps_mapping(args.model_tag, legacy_mapping)
    register_active_run(args.group, args.model_tag)
    print(f"Activated {args.group}: {args.model_tag} ({len(names)} CSVs, {len(full_fwd_sources)} full_fwd, {len(checkpoint_sources)} checkpoints, {len(generation_sources)} generations, {len(analysis_sources)} analyses)")


if __name__ == "__main__":
    main()
