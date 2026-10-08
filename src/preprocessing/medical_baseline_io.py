"""Input provenance and saved-table checks for the accepted medical baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "data/exp/data-preprocessing/medical-baseline"


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def configuration():
    config = json.loads((AUDIT / "decisions.json").read_text())
    assert config["authorization"] == "MILI-D-20261006-018"
    assert config["destination_category_codes"] == ["1", "2"]
    return config


def verify_source(path):
    config = configuration()
    relative = str(path.relative_to(ROOT))
    expected = config["expected_source_hashes"][relative]
    if file_hash(path) != expected:
        raise ValueError(f"Input differs from the approved proposal: {relative}")


def verify_saved(frame, destination):
    restored = pd.read_parquet(destination)
    # Files intentionally store no implicit pandas index; explicit source IDs
    # remain columns and are checked in the full row-and-column comparison.
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored, check_dtype=False)
    report = {"status": "passed", "path": str(destination.relative_to(ROOT)),
              "rows": len(frame), "columns": list(frame.columns),
              "sha256": file_hash(destination), "parquet_round_trip_verified": True}
    (AUDIT / f"{destination.stem}_validation.json").write_text(json.dumps(report, indent=2) + "\n")


def save(frame, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, engine="pyarrow")
    verify_saved(frame, destination)
