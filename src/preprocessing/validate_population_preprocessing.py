"""Check demographic conservation and publication support after preprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-preprocessing/population"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preflight", action="store_true", help="Validate approved source selection before parquet dependencies are available.")
    args = parser.parse_args()
    decisions = json.loads((ROOT / "data/exp/data-preprocessing/decisions.json").read_text())
    if args.source_preflight:
        from census_archive_reader import read_census_archives
        frames = []
        for source, dataset in decisions["datasets"].items():
            mapping = {v["original_name"]: v["readable_name"] for v in dataset["variables"]}
            frame = read_census_archives(ROOT / source)
            frames.append(frame[list(mapping)].rename(columns=mapping).set_index("Mesh Code"))
    else:
        frames = [pd.read_parquet(ROOT / f"data/processed/population/{t}_preprocessed.parquet")
                  .set_index("Mesh Code") for t in ["T001140", "T001188"]]
    first, age = frames
    assert first.index.is_unique and age.index.is_unique
    assert first.index.equals(age.index)
    metadata = ["Suppression Status", "Aggregation Recipient Mesh", "Contributing Suppressed Meshes", "Local Total Population"]
    pd.testing.assert_frame_equal(first[metadata], age[metadata])
    status = first["Suppression Status"]
    assert set(status) == {"0", "1", "2"}
    codes = pd.Series(first.index, index=first.index)
    support = codes.where(status.ne("2"), first["Aggregation Recipient Mesh"])
    for key in first.index[status.eq("2")]:
        recipient = support.loc[key]
        assert recipient in first.index and status.loc[recipient] == "1"
        assert key in first.loc[recipient, "Contributing Suppressed Meshes"].split(";")
    for recipient in first.index[status.eq("1")]:
        members = first.loc[recipient, "Contributing Suppressed Meshes"].split(";")
        assert len(members) == len(set(members))
        assert all(status.loc[key] == "2" and support.loc[key] == recipient for key in members)
    assert first.loc[status.eq("0"), metadata[1:3]].eq("").all().all()
    population = pd.to_numeric(first["Local Total Population"], errors="raise")
    band_names = [f"Published Population Aged {i}-{i+4}" if i < 95 else "Published Population Aged 95+" for i in range(0, 100, 5)]
    available = status.ne("2")
    assert first.loc[~available, ["Published Population Aged 65+", "Published Population Aged 75+"]].eq("*").all().all()
    assert age.loc[~available, band_names].eq("*").all().all()
    bands = age.loc[available, band_names].apply(pd.to_numeric, errors="raise")
    older = pd.to_numeric(first.loc[available, "Published Population Aged 65+"], errors="raise")
    oldest = pd.to_numeric(first.loc[available, "Published Population Aged 75+"], errors="raise")
    pd.testing.assert_series_equal(bands.iloc[:, 13:].sum(axis=1), older, check_names=False)
    pd.testing.assert_series_equal(bands.iloc[:, 15:].sum(axis=1), oldest, check_names=False)
    total = population.groupby(support).sum().loc[older.index]
    known = bands.sum(axis=1)
    assert (older <= known).all() and (known <= total).all()
    # These are accounting checks, not an adopted spatial allocation of age counts.
    expected = json.loads((ROOT / "data/exp/data-briefing/population-sample-audit/age_aggregation_summary.json").read_text())
    actual = {"source_rows_per_table": len(first), "original_population_total": int(population.sum()),
              "publication_support_units": len(total), "published_age65_total": int(older.sum()),
              "diagnostic_unknown_age_total": int((total-known).sum()),
              "reciprocal_links_checked": int(status.eq("2").sum())}
    assert all(actual[k] == expected[k] for k in actual)
    receipts = [json.loads((OUT / f"{t}_sources.json").read_text()) for t in ["T001140", "T001188"]]
    assert sum(len(x) for x in receipts) == expected["source_archives_verified"]
    actual.update(status="passed", selected_source_fields=32, source_archives_verified=sum(len(x) for x in receipts),
                  age_share_denominator=decisions["age_share_denominator"],
                  suppressed_counts_kept_as_symbols=True, all_meshes_retained=True,
                  elderly_location_imputed=False, raw_data_modified=False,
                  final_accessibility_age_link_selected=False,
                  validation_scope="source_preflight" if args.source_preflight else "processed_parquet",
                  parquet_outputs_verified=not args.source_preflight)
    filename = "source_preflight.json" if args.source_preflight else "validation.json"
    (OUT / filename).write_text(json.dumps(actual, indent=2) + "\n")
    print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
