"""Construct the approved known-age share on official census publication supports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from census_archive_reader import verify_saved_table

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/population"
AUDIT = ROOT / "data/exp/data-preprocessing/population"
SHARE = "Population Aged 65+ Share Among Known Ages"


def main():
    decisions = json.loads((ROOT / "data/exp/data-preprocessing/decisions.json").read_text())
    assert decisions["age_measure"]["readable_name"] == SHARE
    assert decisions["age_share_denominator"] == "known-age population on official publication support"
    first = pd.read_parquet(OUT / "T001140_preprocessed.parquet").set_index("Mesh Code")
    age = pd.read_parquet(OUT / "T001188_preprocessed.parquet").set_index("Mesh Code")
    assert first.index.is_unique and age.index.equals(first.index)
    status = first["Suppression Status"]
    codes = pd.Series(first.index, index=first.index)
    support = codes.where(status.ne("2"), first["Aggregation Recipient Mesh"])
    local = pd.to_numeric(first["Local Total Population"], errors="raise").astype("int64")
    membership = pd.DataFrame({"Mesh Code": codes, "Publication Support Code": support,
                               "Suppression Status": status, "Local Total Population": local}).reset_index(drop=True)
    published = first.loc[status.ne("2")]
    bands = [f"Published Population Aged {i}-{i+4}" if i < 95 else "Published Population Aged 95+" for i in range(0, 100, 5)]
    counts = age.loc[published.index, bands].apply(pd.to_numeric, errors="raise").astype("int64")
    groups = pd.DataFrame(index=published.index)
    groups.index.name = "Publication Support Code"
    groups["Member Mesh Count"] = membership.groupby("Publication Support Code").size()
    groups["Support Total Population"] = local.groupby(support).sum()
    groups["Known Age Population"] = counts.sum(axis=1)
    groups["Unknown Age Population"] = groups["Support Total Population"] - groups["Known Age Population"]
    groups["Population Aged 65+"] = pd.to_numeric(published["Published Population Aged 65+"], errors="raise").astype("int64")
    groups["Population Aged 75+"] = pd.to_numeric(published["Published Population Aged 75+"], errors="raise").astype("int64")
    groups["Known Age Denominator Is Zero"] = groups["Known Age Population"].eq(0)
    groups[SHARE] = groups["Population Aged 65+"] / groups["Known Age Population"].where(groups["Known Age Population"].gt(0))
    assert groups["Unknown Age Population"].ge(0).all()
    assert groups["Population Aged 65+"].le(groups["Known Age Population"]).all()
    assert groups[SHARE].dropna().between(0, 1).all()
    assert groups[SHARE].isna().equals(groups["Known Age Denominator Is Zero"])
    # Independently prepared audit groups provide row-level accounting validation.
    previous = pd.read_csv(
        ROOT / "data/exp/data-briefing/population-sample-audit/publication_group_checks.csv",
        dtype={"diagnostic_group_id": str},
    ).set_index("diagnostic_group_id").loc[groups.index]
    pairs = {"Member Mesh Count": "member_meshes", "Support Total Population": "diagnostic_member_population_sum",
             "Known Age Population": "published_known_age_sum", "Population Aged 65+": "published_recipient_or_singleton_age65",
             "Unknown Age Population": "diagnostic_unknown_age_residual"}
    for name, old in pairs.items():
        pd.testing.assert_series_equal(groups[name], previous[old].astype("int64"), check_names=False)
    destinations = [(membership, OUT / "population_support_membership.parquet"),
                    (groups.reset_index(), OUT / "population_age_support.parquet")]
    for frame, destination in destinations:
        frame.to_parquet(destination, index=False, engine="pyarrow")
        verify_saved_table(frame, destination)
    report = {"status": "passed", "authorization": decisions["age_measure"]["authorization"],
              "membership_rows": len(membership), "support_rows": len(groups),
              "total_population": int(groups["Support Total Population"].sum()),
              "known_age_population": int(groups["Known Age Population"].sum()),
              "unknown_age_population": int(groups["Unknown Age Population"].sum()),
              "population_aged_65_plus": int(groups["Population Aged 65+"].sum()),
              "zero_known_age_denominator_supports": int(groups["Known Age Denominator Is Zero"].sum()),
              "row_level_independent_audit_match": True, "elderly_location_imputed": False,
              "final_accessibility_age_link_selected": False, "raw_data_modified": False,
              "inputs": []}
    for name in ["T001140_preprocessed.parquet", "T001188_preprocessed.parquet"]:
        path = OUT / name
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        report["inputs"].append({"path": str(path.relative_to(ROOT)), "sha256": digest})
    (AUDIT / "age_support_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
