"""Construct accepted member and age-support medical-access baseline variables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from medical_baseline_io import ROOT, AUDIT, configuration, save, verify_source

PROC = ROOT / "data/processed/medical-baseline"
DISTANCE = "Candidate Medical Network Distance"
MEAN = "Population-Weighted Candidate Medical Network Distance"
STATES = ["finite_candidate_path", "eligible_entry_no_candidate_path", "entry_pending"]
COUNTS = ["Population With Candidate Medical Path", "Population Without Candidate Medical Path", "Population With Pending Road Entry"]
SHARES = ["Population Share With Candidate Medical Path", "Population Share Without Candidate Medical Path", "Population Share With Pending Road Entry"]


def main():
    config = configuration()
    for relative in config["expected_source_hashes"]:
        verify_source(ROOT / relative)
    membership = pd.read_parquet(ROOT / "data/processed/population/population_support_membership.parquet")
    ages = pd.read_parquet(ROOT / "data/processed/population/population_age_support.parquet")
    source_distance = pd.read_parquet(PROC / "medical_mesh_distances_preprocessed.parquet")
    assert source_distance["Service Target"].eq("medical").all()
    assert membership["Mesh Code"].is_unique and ages["Publication Support Code"].is_unique
    assert set(source_distance["Mesh Code"]) == set(membership["Mesh Code"])
    member = membership.merge(source_distance.drop(columns="Service Target"), on="Mesh Code", validate="one_to_one", how="left")
    source_points = ROOT / "data/exp/national-population-entry/point_attachment_diagnostics.csv"
    points = pd.read_csv(source_points, dtype={"point_id": str, "population_raw": str}, keep_default_na=False, low_memory=False)
    points = points[points.dataset.eq("population")].copy()
    points["Mesh Code"] = points.point_id.str.removeprefix("mesh:")
    points = points.set_index("Mesh Code").loc[member["Mesh Code"]]
    np.testing.assert_array_equal(pd.to_numeric(points.population_raw), member["Local Total Population"])
    np.testing.assert_array_equal(points.routing_eligible, member["Candidate Road Entry Eligible"])
    finite = np.isfinite(member[DISTANCE])
    assert not np.isinf(member[DISTANCE]).any()
    assert member.loc[finite, DISTANCE].ge(0).all()
    assert not finite[~member["Candidate Road Entry Eligible"]].any()
    np.testing.assert_array_equal(finite, points.medical_component_reachable_exact)
    member["Medical Route Status"] = np.where(finite, STATES[0], np.where(member["Candidate Road Entry Eligible"], STATES[1], STATES[2]))
    member["Candidate Road Entry Method"] = points.entry_method.to_numpy()
    member["Longitude"] = points.longitude.to_numpy()
    member["Latitude"] = points.latitude.to_numpy()
    group = member["Publication Support Code"]
    indexed = ages.set_index("Publication Support Code")
    assert member.groupby("Publication Support Code").size().reindex(indexed.index).equals(indexed["Member Mesh Count"])
    assert member.groupby("Publication Support Code")["Local Total Population"].sum().reindex(indexed.index).equals(indexed["Support Total Population"])
    support = indexed.copy()
    for state, count, share in zip(STATES, COUNTS, SHARES):
        summed = member.loc[member["Medical Route Status"].eq(state)].groupby("Publication Support Code")["Local Total Population"].sum()
        support[count] = summed.reindex(support.index, fill_value=0).astype("int64")
        support[share] = support[count] / support["Support Total Population"].where(support["Support Total Population"].gt(0))
    weighted = member.loc[finite, "Local Total Population"] * member.loc[finite, DISTANCE]
    numerator = weighted.groupby(group[finite]).sum().reindex(support.index, fill_value=0)
    denominator = support[COUNTS[0]]
    support[MEAN] = numerator / denominator.where(denominator.gt(0))
    support["Medical Distance Denominator Is Zero"] = denominator.eq(0)
    assert support[COUNTS].sum(axis=1).equals(support["Support Total Population"])
    assert support[MEAN].isna().equals(support["Medical Distance Denominator Is Zero"])
    np.testing.assert_allclose(support.loc[support["Support Total Population"].gt(0), SHARES].sum(axis=1), 1, rtol=0, atol=1e-12)
    for share in SHARES:
        assert support[share].dropna().between(0, 1).all()
    save(member, PROC / "medical_mesh_baseline.parquet")
    save(support.reset_index(), PROC / "medical_age_support_baseline.parquet")
    status = [{"status": state, "meshes": int(member["Medical Route Status"].eq(state).sum()),
               "population": int(support[count].sum())} for state, count in zip(STATES, COUNTS)]
    total_weighted = float(weighted.sum() / denominator.sum())
    summary = {"authorization": config["authorization"], "member_rows": len(member), "age_support_rows": len(support),
               "total_population": int(support["Support Total Population"].sum()), "route_status_partition": status,
               "population_weighted_finite_path_mean_metres": total_weighted,
               "supports_with_zero_finite_path_population": int(support["Medical Distance Denominator Is Zero"].sum()),
               "supports_with_zero_total_population": int(support["Support Total Population"].eq(0).sum()),
               "new_route_calculation_performed": False, "elderly_location_imputed": False,
               "interpretation": "Potential registered hospital/general-clinic access under the accepted candidate undirected road graph; not observed travel time or individual elderly access."}
    (AUDIT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
