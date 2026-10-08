"""Validate saved national baseline accounting independently from its constructor."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from medical_baseline_io import ROOT, AUDIT, configuration, file_hash, verify_source
from construct_medical_baseline import PROC, DISTANCE, MEAN, STATES, COUNTS, SHARES


def main():
    config = configuration()
    member = pd.read_parquet(PROC / "medical_mesh_baseline.parquet")
    support = pd.read_parquet(PROC / "medical_age_support_baseline.parquet").set_index("Publication Support Code")
    facilities = pd.read_parquet(PROC / "medical_facilities_preprocessed.parquet")
    membership = pd.read_parquet(ROOT / "data/processed/population/population_support_membership.parquet")
    ages = pd.read_parquet(ROOT / "data/processed/population/population_age_support.parquet").set_index("Publication Support Code")
    assert len(member) == len(membership) == 176964 and len(support) == len(ages) == 146540
    assert member["Mesh Code"].is_unique and support.index.is_unique and facilities["Medical Facility ID"].is_unique
    pd.testing.assert_frame_equal(member[membership.columns], membership)
    pd.testing.assert_frame_equal(support[ages.columns], ages)
    assert len(facilities) == 181312
    assert facilities["Medical Facility Category"].value_counts().to_dict() == {"1": 8269, "2": 104183, "3": 68860}
    assert facilities["Selected Medical Destination"].equals(facilities["Medical Facility Category"].isin(["1", "2"]))
    assert facilities["Included Medical Network Destination"].equals(facilities["Selected Medical Destination"] & facilities["Candidate Road Entry Eligible"])
    assert int(facilities["Selected Medical Destination"].sum()) == 112452
    assert int(facilities["Included Medical Network Destination"].sum()) == 112401
    source = pd.read_csv(ROOT / "data/exp/national-population-entry/population_path_diagnostics.csv", dtype={"mesh_code": str})
    original = source[source.target.eq("medical")].set_index("mesh_code").loc[member["Mesh Code"]]
    np.testing.assert_allclose(member[DISTANCE], original.network_distance_m, equal_nan=True, rtol=0, atol=1e-9)
    finite = member[DISTANCE].notna()
    assert member.loc[finite, "Medical Route Status"].eq(STATES[0]).all()
    assert member.loc[~finite & member["Candidate Road Entry Eligible"], "Medical Route Status"].eq(STATES[1]).all()
    assert member.loc[~member["Candidate Road Entry Eligible"], "Medical Route Status"].eq(STATES[2]).all()
    # Independent array accumulation verifies every support without repeating
    # the constructor's pandas group-by calculation.
    keys = support.index.get_indexer(member["Publication Support Code"])
    assert (keys >= 0).all()
    n = len(support)
    pop = member["Local Total Population"].to_numpy(dtype="int64")
    dist = member[DISTANCE].to_numpy()
    masses = []
    for state, count, share in zip(STATES, COUNTS, SHARES):
        mask = member["Medical Route Status"].eq(state).to_numpy()
        accumulated = np.zeros(n, dtype="int64")
        np.add.at(accumulated, keys[mask], pop[mask])
        np.testing.assert_array_equal(accumulated, support[count])
        masses.append(accumulated)
        totals = support["Support Total Population"].to_numpy()
        expected_share = np.divide(accumulated, totals, out=np.full(n, np.nan), where=totals > 0)
        np.testing.assert_allclose(expected_share, support[share], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(np.sum(masses, axis=0), support["Support Total Population"])
    mask = np.isfinite(dist)
    numerator = np.bincount(keys[mask], weights=pop[mask] * dist[mask], minlength=n)
    expected_mean = np.divide(numerator, masses[0], out=np.full(n, np.nan), where=masses[0] > 0)
    np.testing.assert_allclose(expected_mean, support[MEAN], rtol=1e-12, atol=1e-8, equal_nan=True)
    minimum, maximum = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(minimum, keys[mask], dist[mask])
    np.maximum.at(maximum, keys[mask], dist[mask])
    valid = masses[0] > 0
    assert np.all(expected_mean[valid] >= minimum[valid] - 1e-8)
    assert np.all(expected_mean[valid] <= maximum[valid] + 1e-8)
    np.testing.assert_array_equal(support[MEAN].isna(), masses[0] == 0)
    np.testing.assert_array_equal(support["Medical Distance Denominator Is Zero"], masses[0] == 0)
    checked = n
    assert int(support["Support Total Population"].sum()) == 126146099
    saved = []
    for name in ["medical_facilities_preprocessed.parquet", "medical_mesh_distances_preprocessed.parquet", "medical_mesh_baseline.parquet", "medical_age_support_baseline.parquet"]:
        path = PROC / name
        receipt = json.loads((AUDIT / f"{path.stem}_validation.json").read_text())
        assert receipt["status"] == "passed" and receipt["sha256"] == file_hash(path)
        assert len({c.casefold() for c in receipt["columns"]}) == len(receipt["columns"])
        assert all(c.isascii() for c in receipt["columns"])
        saved.append(receipt)
    for relative in config["expected_source_hashes"]:
        verify_source(ROOT / relative)
    result = {"status": "passed", "authorization": config["authorization"], "member_rows": len(member),
              "support_rows_checked_independently": checked, "medical_inventory_rows_retained": len(facilities),
              "all_demographic_records_and_age_counts_preserved": True, "source_distances_preserved": True,
              "status_population_conserved": True, "row_level_weighted_means_verified": True,
              "missing_distance_not_zero_filled": True, "elderly_location_imputed": False,
              "approved_sources_unchanged": True, "raw_data_modified": False, "saved_outputs": saved}
    (AUDIT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "saved_outputs"}, indent=2))


if __name__ == "__main__":
    main()
