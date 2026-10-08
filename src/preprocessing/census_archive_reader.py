"""Read the approved census archive format without altering source values."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-preprocessing/population"


def read_census_archives(source: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        ROOT / "data/exp/data-acquisition/population_2020_download_manifest.csv",
        dtype=str, keep_default_na=False,
    ).set_index("filename")
    if not manifest.index.is_unique:
        raise ValueError("Duplicate archive filenames in acquisition manifest")
    frames, receipts = [], []
    archives = sorted(source.glob("*.zip"))
    if not archives:
        raise ValueError(f"No census archives in {source}")
    for archive in archives:
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != manifest.loc[archive.name, "sha256"]:
            raise ValueError(f"Source hash mismatch: {archive.name}")
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"Unexpected census archive contents: {archive.name}")
            with bundle.open(members[0]) as handle:
                frame = pd.read_csv(
                    handle, encoding="cp932", skiprows=[1],
                    dtype=str, keep_default_na=False,
                )
        frames.append(frame)
        receipts.append({"path": str(archive.relative_to(ROOT)), "member": members[0],
                         "sha256": digest, "rows": len(frame)})
    result = pd.concat(frames, ignore_index=True)
    if result["KEY_CODE"].duplicated().any():
        raise ValueError("Duplicate census mesh codes")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{source.name}_sources.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return result


def verify_saved_table(frame: pd.DataFrame, destination: Path) -> None:
    restored = pd.read_parquet(destination)
    pd.testing.assert_frame_equal(frame, restored, check_dtype=False)
    with destination.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"status": "passed", "rows": len(frame), "columns": list(frame.columns),
              "output": str(destination.relative_to(ROOT)), "sha256": digest,
              "source_symbols_preserved": True, "round_trip_verified": True,
              "raw_data_modified": False}
    (OUT / f"{destination.stem}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
