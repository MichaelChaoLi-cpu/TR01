"""Read-only source-scope and baseline compatibility audit for approved forecasts."""
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import shapefile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-briefing/demographic-projection"
ARCHIVE = ROOT / "data/exp/data-acquisition/demographic-projection/1km_mesh_2024_SHP.zip"
YEARS = [2030, 2040, 2050]
FIELDS = ["MESH_ID", "SHICODE", "PTN_2020"] + [field for year in YEARS for field in
          [f"HITOKU{year}", f"GASSAN{year}", f"PTN_{year}", f"PT00_{year}", f"PTC_{year}", f"RTC_{year}"]]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet"
    before = {str(p.relative_to(ROOT)): digest(p) for p in [ARCHIVE, baseline_path]}
    frames, fields_inventory = [], []
    with zipfile.ZipFile(ARCHIVE) as nation:
        members = sorted(x for x in nation.namelist() if x.endswith("_SHP.zip"))
        assert len(members) == 47
        for member in members:
            prefecture = member.rsplit("_", 2)[1]
            with zipfile.ZipFile(io.BytesIO(nation.read(member))) as local:
                dbf = next(x for x in local.namelist() if x.endswith(".dbf"))
                cpg = next(x for x in local.namelist() if x.endswith(".cpg"))
                encoding = local.read(cpg).decode().strip()
                reader = shapefile.Reader(dbf=io.BytesIO(local.read(dbf)), encoding=encoding)
                available = [x[0] for x in reader.fields[1:]]
                assert set(FIELDS) <= set(available)
                for field in reader.fields[1:]:
                    fields_inventory.append({"Source Prefecture": prefecture, "Field": field[0],
                                             "Type": field[1], "Width": field[2], "Decimals": field[3],
                                             "In Current Audit Scope": field[0] in FIELDS})
                frame = pd.DataFrame([x.as_dict() for x in reader.iterRecords(fields=FIELDS)])
                frame["Source Prefecture"] = prefecture
                frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    source.to_parquet(OUT / "source_scope.parquet", index=False)
    pd.DataFrame(fields_inventory).to_csv(OUT / "all_source_field_inventory.csv", index=False)
    duplicate = source["MESH_ID"].duplicated(keep=False)
    source.loc[duplicate].to_csv(OUT / "duplicate_mesh_rows.csv", index=False)
    assert not source.duplicated(["MESH_ID", "SHICODE"]).any()
    assert source["SHICODE"].str[:2].equals(source["Source Prefecture"])
    # Diagnostic totals combine municipality components, preserving all source rows.
    # No age counts are assigned to individual meshes: their suppression supports differ.
    totals_by_mesh = source.groupby("MESH_ID", sort=False)[["PTN_2020"] + [f"PTN_{y}" for y in YEARS]].sum().reset_index()
    baseline = pd.read_parquet(baseline_path)
    joined = baseline.merge(totals_by_mesh, left_on="Mesh Code", right_on="MESH_ID", how="left", validate="one_to_one", indicator=True)
    matched = joined["_merge"].eq("both")
    joined.loc[~matched, ["Mesh Code", "Publication Support Code", "Local Total Population"]].to_csv(OUT / "unmatched_selected_meshes.csv", index=False)
    report = {"status": "source_audit_complete", "authorization": "MILI-D-20261007-035",
              "source_rows": len(source), "source_fields_available": len(available),
              "scoped_source_fields": FIELDS, "source_prefectures": source["Source Prefecture"].nunique(),
              "source_meshes_unique": not duplicate.any(), "source_mesh_municipality_pairs_unique": True,
              "source_distinct_meshes": len(totals_by_mesh), "multi_component_meshes": int(source.loc[duplicate,"MESH_ID"].nunique()),
              "diagnostic_mesh_totals": "Sum municipality-specific unsuppressed totals; original rows preserved.",
              "selected_meshes": len(baseline),
              "matched_selected_meshes": int(matched.sum()), "unmatched_selected_meshes": int((~matched).sum()),
              "unmatched_selected_population": int(joined.loc[~matched, "Local Total Population"].sum()),
              "source_reference_population": float(source["PTN_2020"].sum()),
              "matched_reference_population": float(joined.loc[matched, "PTN_2020"].sum()),
              "matched_study_population": int(joined.loc[matched, "Local Total Population"].sum()),
              "zero_forecast_reference_positive_study_population": int((matched & joined["PTN_2020"].eq(0) & joined["Local Total Population"].gt(0)).sum()),
              "years": {}, "original_source_hashes": before, "raw_data_modified": False,
              "new_scientific_variables_adopted": False}
    ref_delta = joined.loc[matched, "PTN_2020"] - joined.loc[matched, "Local Total Population"]
    report["reference_count_difference_quantiles"] = {str(q): float(ref_delta.quantile(q)) for q in [0, .25, .5, .75, 1]}
    for year in YEARS:
        flag = source[f"HITOKU{year}"]
        flags = flag.value_counts(dropna=False).to_dict()
        totals = source[f"PTN_{year}"]
        positive = totals.gt(0)
        valid_age = ~flag.eq("*")
        detail = {"flag_counts": {str(k): int(v) for k,v in flags.items()},
                  "source_total_population": float(totals.sum()), "zero_total_meshes": int(totals.eq(0).sum()),
                  "missing_total_meshes": int(totals.isna().sum()), "negative_total_meshes": int(totals.lt(0).sum()),
                  "source_published_age_total_population": float(source[f"PT00_{year}"].sum()),
                  "source_published_population_aged65": float(source[f"PTC_{year}"].sum()),
                  "suppressed_age_meshes": int(flag.eq("*").sum()), "aggregation_recipient_meshes": int(flag.eq("@").sum()),
                  "reference_positive_growth_ratio_quantiles": {str(q): float((totals/source["PTN_2020"].where(source["PTN_2020"].gt(0))).quantile(q)) for q in [0,.25,.5,.75,1]},
                  "selected_future_population_without_reference_adjustment": float(joined.loc[matched,f"PTN_{year}"].sum())}
        assert totals.notna().all() and totals.ge(0).all()
        assert source.loc[valid_age,f"PTC_{year}"].ge(0).all()
        assert (source.loc[valid_age,f"PTC_{year}"] <= source.loc[valid_age,f"PT00_{year}"] + .001).all()
        report["years"][str(year)] = detail
    assert all(digest(ROOT / path)==sha for path,sha in before.items())
    report["source_hashes_unchanged"] = True
    (OUT / "source_compatibility_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
