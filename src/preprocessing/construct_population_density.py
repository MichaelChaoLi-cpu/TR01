#!/usr/bin/env python3
"""Construct the confirmed publication-support population-density input.

Full member footprints, including water, use the JGD2011 ellipsoid. Original
population/age/medical baselines are read only; no age counts are redistributed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS

ROOT = Path(__file__).resolve().parents[2]
MEMBERSHIP = ROOT / "data/processed/population/population_support_membership.parquet"
SUPPORT = ROOT / "data/processed/population/population_age_support.parquet"
SAMPLE = ROOT / "data/processed/medical-analysis-sample/medical_age_support_analysis_preprocessed.parquet"
OUT = ROOT / "data/processed/population-density/population_density_constructed.parquet"
EXP = ROOT / "data/exp/data-preprocessing/population-density"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    decision = json.loads((EXP / "decisions.json").read_text())
    assert decision["authorization"] == "MILI-D-20261007-007"
    sources = [MEMBERSHIP, SUPPORT, SAMPLE]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    membership = pd.read_parquet(MEMBERSHIP)
    supports = pd.read_parquet(SUPPORT).set_index("Publication Support Code")
    sample = pd.read_parquet(SAMPLE)
    assert membership["Mesh Code"].is_unique and supports.index.is_unique
    code = membership["Mesh Code"].astype(str)
    assert code.str.fullmatch(r"\d{8}").all()
    latitude_row = code.str[:2].astype(int)*80 + code.str[4].astype(int)*10 + code.str[6].astype(int)
    longitude_col = (code.str[2:4].astype(int)+100)*80 + code.str[5].astype(int)*10 + code.str[7].astype(int)
    geod = CRS.from_epsg(6668).get_geod()
    by_row = {}
    for row in np.unique(latitude_row):
        south, north = row/120, (row+1)/120
        area, _ = geod.polygon_area_perimeter([0, 1/80, 1/80, 0], [south, south, north, north])
        by_row[int(row)] = abs(area)/1e6
    area = latitude_row.map(by_row)
    assert np.isfinite(area).all() and area.gt(0).all()
    # Independent geographic-position check of the latitude-only area lookup.
    for idx in np.linspace(0, len(membership)-1, 100, dtype=int):
        west, east = longitude_col.iloc[idx]/80, (longitude_col.iloc[idx]+1)/80
        south, north = latitude_row.iloc[idx]/120, (latitude_row.iloc[idx]+1)/120
        independent, _ = geod.polygon_area_perimeter([west, east, east, west], [south, south, north, north])
        np.testing.assert_allclose(abs(independent)/1e6, area.iloc[idx], atol=1e-8, rtol=0)
    grouped = membership.assign(_area=area).groupby("Publication Support Code", sort=False).agg(
        **{"Member Mesh Count": ("Mesh Code", "size"),
           "Support Total Population": ("Local Total Population", "sum"),
           "Support Footprint Area": ("_area", "sum")})
    grouped = grouped.loc[supports.index]
    for column in ["Member Mesh Count", "Support Total Population"]:
        pd.testing.assert_series_equal(grouped[column], supports[column], check_dtype=False)
    grouped["Population Density"] = grouped["Support Total Population"] / grouped["Support Footprint Area"]
    assert np.isfinite(grouped["Population Density"]).all() and grouped["Population Density"].gt(0).all()
    assert len(grouped) == 146540 and grouped["Member Mesh Count"].sum() == 176964
    chosen = grouped.loc[sample["Publication Support Code"]]
    for column in ["Member Mesh Count", "Support Total Population"]:
        np.testing.assert_array_equal(chosen[column], sample[column])
    frame = grouped.reset_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(OUT))
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected
    report = {
        "status": "passed", "authorization": decision["authorization"],
        "readable_name": "Population Density", "unit": "residents/km2",
        "support_rows": len(frame), "member_meshes": len(membership),
        "selected_support_rows": len(sample),
        "support_total_population": int(frame["Support Total Population"].sum()),
        "single_mesh_area_range_km2": [float(area.min()), float(area.max())],
        "support_density_range": [float(frame["Population Density"].min()), float(frame["Population Density"].max())],
        "crs": "EPSG:6668", "ellipsoid": "GRS80",
        "area_method": "Ellipsoidal native-mesh corner polygons; complete footprints including water; sum all official member meshes",
        "area_scope": "Complete observed member-mesh footprints, without land clipping or treating nominal meshes as exactly 1 km2",
        "independent_geographic_position_area_check": "passed, 100 native meshes",
        "support_population_and_membership_check": "passed for every original and selected support",
        "source_hashes_verified": hashes, "parquet_readback": "passed",
        "source_tables_modified": False, "elderly_location_imputed": False,
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": digest(OUT),
    }
    (EXP / "construction_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Saved {len(frame):,} rows x {len(frame.columns)} columns: {OUT.relative_to(ROOT)}")
    print("Validated all support populations, memberships, complete footprints, source hashes and parquet read-back.")


if __name__ == "__main__":
    main()
