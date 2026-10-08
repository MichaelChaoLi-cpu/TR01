"""Validate the selected baseline sample against preserved input observations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data/exp/data-preprocessing/medical-analysis-sample"
BASE = ROOT / "data/processed/medical-baseline"
OUT = ROOT / "data/processed/medical-analysis-sample"


def main():
    m = pd.read_parquet(BASE / "medical_mesh_baseline.parquet")
    a = pd.read_parquet(BASE / "medical_age_support_baseline.parquet")
    mesh = pd.read_parquet(OUT / "medical_mesh_analysis_preprocessed.parquet")
    support = pd.read_parquet(OUT / "medical_age_support_analysis_preprocessed.parquet")
    age_invalid = a["Population Aged 65+ Share Among Known Ages"].isna()
    mesh_age_invalid = m["Publication Support Code"].isin(a.loc[age_invalid, "Publication Support Code"])
    mesh_distance_invalid = m["Candidate Medical Network Distance"].isna()
    excluded = mesh_age_invalid | mesh_distance_invalid
    support_excluded = age_invalid | a["Population-Weighted Candidate Medical Network Distance"].isna()
    pd.testing.assert_frame_equal(mesh, m.loc[~excluded].reset_index(drop=True))
    pd.testing.assert_frame_equal(support, a.loc[~support_excluded].reset_index(drop=True))
    assert mesh["Mesh Code"].is_unique and support["Publication Support Code"].is_unique
    assert set(mesh["Publication Support Code"]) == set(support["Publication Support Code"])
    assert np.isfinite(mesh["Candidate Medical Network Distance"]).all()
    assert np.isfinite(support["Population Aged 65+ Share Among Known Ages"]).all()
    assert np.isfinite(support["Population-Weighted Candidate Medical Network Distance"]).all()
    assert (mesh["Candidate Medical Network Distance"] >= 0).all()
    # All support counts and proportions remain original; no age redistribution.
    for src, expected in json.loads((EXP / "input_hashes.json").read_text()).items():
        assert hashlib.sha256((ROOT / src).read_bytes()).hexdigest() == expected

    rows = []
    masks = {
        "All baseline meshes": np.ones(len(m), dtype=bool),
        "Undefined ageing share": mesh_age_invalid,
        "Eligible road entry without baseline medical path": m["Medical Route Status"].eq("eligible_entry_no_candidate_path"),
        "Pending baseline road entry": m["Medical Route Status"].eq("entry_pending"),
        "Overlap of age and distance exclusions": mesh_age_invalid & mesh_distance_invalid,
        "Excluded union": excluded,
        "Retained analysis sample": ~excluded,
    }
    for category, mask in masks.items():
        population = int(m.loc[mask, "Local Total Population"].sum())
        rows.append({"Category": category, "Unit": "mesh", "Record Count": int(np.sum(mask)),
                     "Resident Count": population,
                     "Share of Baseline Population": population / int(m["Local Total Population"].sum())})
    pd.DataFrame(rows).to_csv(EXP / "sample_accounting.csv", index=False)
    summary = {
        "authorization": "MILI-D-20261006-019", "validation": "passed",
        "baseline_meshes": len(m), "baseline_age_supports": len(a),
        "undefined_age_supports": int(age_invalid.sum()),
        "excluded_meshes": int(excluded.sum()),
        "excluded_residents": int(m.loc[excluded, "Local Total Population"].sum()),
        "retained_meshes": len(mesh), "retained_supports": len(support),
        "retained_mesh_residents": int(mesh["Local Total Population"].sum()),
        "excluded_supports": int(support_excluded.sum()),
        "retained_support_full_population": int(support["Support Total Population"].sum()),
        "inputs_unchanged": True, "values_unchanged": True,
        "age_redistributed": False,
        "interpretation": "Selected baseline residents with calculable candidate medical distances and defined official-support ageing shares. Full support population is not selected mesh population. Subsequent scenario-induced disconnection is an outcome, not a new sample exclusion.",
    }
    (EXP / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
