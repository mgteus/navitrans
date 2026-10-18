"""Read-only smoke check for the data used by the HardPos training notebook."""

from __future__ import annotations

import hashlib
import pickle
import sys
from numbers import Integral
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csv_runs import _tag


def main() -> None:
    dataset_path = ROOT / "bert_trainset_squared_25.gzip"
    mapping_path = ROOT / "ps_to_psid.pkl"
    sargolini_path = ROOT / "sargolini_trainset_squared_25.gzip"
    for path in (dataset_path, mapping_path, sargolini_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    df = pd.read_parquet(dataset_path, columns=["position_state"])
    mapping = pickle.loads(mapping_path.read_bytes())
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("ps_to_psid.pkl must contain a nonempty dictionary")
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("PS keys must be strings")
    ids = list(mapping.values())
    if not all(isinstance(value, Integral) and value >= 0 for value in ids):
        raise ValueError("PS IDs must be nonnegative integers")
    if len(ids) != len(set(ids)):
        raise ValueError("PS IDs must be unique")
    observed = set(df["position_state"])
    missing = observed - mapping.keys()
    if missing:
        raise ValueError(f"{len(missing)} training PS are missing from ps_to_psid.pkl")
    mapped_ids = df["position_state"].map(mapping)
    if mapped_ids.isna().any():
        raise ValueError("The training sequence contains unmapped PS")
    pad_token_id = int(mapped_ids.max()) + 1
    fingerprint = hashlib.sha256(pickle.dumps(mapping, protocol=4)).hexdigest()[:8]
    _tag(f"HardPos_embed16_context10_epochs1000_psmap{fingerprint}_")
    print(f"HardPos inputs OK: {len(df)} rows, {len(observed)} observed PS, "
          f"{len(mapping)} mapping entries, pad_token_id={pad_token_id}, "
          f"psmap={fingerprint}")


if __name__ == "__main__":
    main()
