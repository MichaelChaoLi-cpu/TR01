"""Prepare a bounded medical baseline proposal; adopt no new analysis rules."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-preprocessing/medical-baseline-proposal"
CLASS = {1: "Hospital", 2: "General clinic", 3: "Dental clinic"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sop_before = digest(ROOT / "docs/AnaSOP.md")
    source = ROOT / "data/raw/poi/medical/P04-20_GML.zip"
    manifest = pd.read_csv(ROOT / "data/exp/data-acquisition/facility_download_manifest.csv", dtype=str)
    expected = manifest.loc[manifest.category.eq("medical"), "sha256"].item()
    assert digest(source) == expected
    with zipfile.ZipFile(source) as bundle:
        members = [name for name in bundle.namelist() if name.endswith(".shp")]
        assert len(members) == 1
        reader = shapefile.Reader(dbf=io.BytesIO(bundle.read(members[0][:-4] + ".dbf")), encoding="cp932")
        fields = [item[0] for item in reader.fields[1:]]
        facilities = pd.DataFrame.from_records(list(reader.iterRecords()), columns=fields)
    inventory = []
    for col in fields:
        series = facilities[col]
        empty = series.isna() | series.astype(str).eq("")
        inventory.append({"original_name": col, "dtype": str(series.dtype),
                          "null_or_empty_pct": float(empty.mean() * 100),
                          "sample_values": "; ".join(series.dropna().head(2).astype(str)),
                          "status": "discovered; selection pending"})
    pd.DataFrame(inventory).to_csv(OUT / "medical_source_variable_inventory.csv", index=False)
    attachment_path = ROOT / "data/exp/national-population-entry/point_attachment_diagnostics.csv"
    paths_path = ROOT / "data/exp/national-population-entry/population_path_diagnostics.csv"
    points = pd.read_csv(attachment_path, dtype={"point_id": str, "category": str, "population_raw": str},
                         keep_default_na=False, low_memory=False)
    medical = points[points.dataset.eq("medical")].reset_index(drop=True)
    assert len(medical) == len(facilities) == 181312
    assert medical.point_id.tolist() == [f"medical:{i}" for i in range(len(facilities))]
    assert medical.name.tolist() == facilities.P04_002.tolist()
    assert medical.category.tolist() == facilities.P04_001.astype(str).tolist()
    categories = []
    for code, label in CLASS.items():
        selected = medical.category.eq(str(code))
        categories.append({"code": code, "category": label, "source_records": int(selected.sum()),
                           "candidate_entry_eligible": int(medical.loc[selected, "routing_eligible"].sum()),
                           "candidate_entry_pending": int((~medical.loc[selected, "routing_eligible"]).sum())})
    pd.DataFrame(categories).to_csv(OUT / "medical_category_entry_summary.csv", index=False)
    paths = pd.read_csv(paths_path, dtype={"point_id": str, "mesh_code": str}, float_precision="round_trip")
    paths = paths[paths.target.eq("medical")].set_index("mesh_code")
    pop_points = points[points.dataset.eq("population")].copy()
    pop_points["mesh_code"] = pop_points.point_id.str.removeprefix("mesh:")
    pop_points = pop_points.set_index("mesh_code")
    membership_path = ROOT / "data/processed/population/population_support_membership.parquet"
    support_path = ROOT / "data/processed/population/population_age_support.parquet"
    membership = pd.read_parquet(membership_path).set_index("Mesh Code")
    support = pd.read_parquet(support_path).set_index("Publication Support Code")
    assert len(paths) == len(pop_points) == len(membership) == 176964
    assert paths.index.is_unique and pop_points.index.is_unique and membership.index.is_unique
    assert set(paths.index) == set(membership.index) == set(pop_points.index)
    paths = paths.loc[membership.index]
    pop_points = pop_points.loc[membership.index]
    population = membership["Local Total Population"]
    np.testing.assert_array_equal(pd.to_numeric(pop_points.population_raw), population)
    assert paths.routing_eligible.equals(pop_points.routing_eligible)
    finite = np.isfinite(paths.network_distance_m)
    assert paths.loc[finite, "network_distance_m"].ge(0).all()
    assert not finite[~paths.routing_eligible].any()
    assert finite.equals(pop_points.medical_component_reachable_exact)
    state = pd.Series("entry_pending", index=paths.index)
    state.loc[paths.routing_eligible & ~finite] = "eligible_entry_no_candidate_path"
    state.loc[finite] = "finite_candidate_path"
    counts = []
    for label in ["finite_candidate_path", "eligible_entry_no_candidate_path", "entry_pending"]:
        mask = state.eq(label)
        counts.append({"existing_diagnostic_status": label, "meshes": int(mask.sum()),
                       "population": int(population[mask].sum())})
    assert sum(row["meshes"] for row in counts) == len(membership)
    assert sum(row["population"] for row in counts) == 126146099
    pd.DataFrame(counts).to_csv(OUT / "existing_medical_route_status_summary.csv", index=False)
    group = membership["Publication Support Code"]
    all_states = pd.crosstab(group, state).reindex(support.index, fill_value=0)
    assert all_states.sum(axis=1).equals(support["Member Mesh Count"])
    group_checks = {"supports": len(support), "multi_mesh_supports": int(support["Member Mesh Count"].gt(1).sum()),
                    "supports_with_entry_pending_members": int(all_states.entry_pending.gt(0).sum()),
                    "supports_with_eligible_no_path_members": int(all_states.eligible_entry_no_candidate_path.gt(0).sum()),
                    "supports_with_mixed_route_statuses": int(all_states.gt(0).sum(axis=1).gt(1).sum())}
    # Inspect support compatibility only. Do not construct a proposed weighted-distance outcome.
    variables = [
        ("Candidate Medical Network Distance", "Member mesh", "Existing candidate graph shortest distance in metres; missing for no path or pending entry."),
        ("Population-Weighted Candidate Medical Network Distance", "Official age publication support", "Sum of local population times finite candidate distance divided by local population with finite paths; missing when the denominator is zero."),
        ("Population Share With Candidate Medical Path", "Official age publication support", "Local population in member meshes with finite paths divided by support total population."),
        ("Population Share Without Candidate Medical Path", "Official age publication support", "Local population with eligible entries but no candidate path divided by support total population."),
        ("Population Share With Pending Road Entry", "Official age publication support", "Local population with unresolved candidate entry divided by support total population."),
    ]
    pd.DataFrame([{"proposed_readable_name": n, "support": s, "proposed_definition": definition,
                   "status": "awaiting human selection"} for n, s, definition in variables]).to_csv(OUT / "variable_proposal.csv", index=False)
    record = {"status": "passed", "proposal_adopted": False, "official_category_source": "https://nlftp.mlit.go.jp/ksj/gml/codelist/MedClassCd.html",
              "source_archive_sha256_verified": True, "medical_records_matched_by_original_order_name_and_category": len(medical),
              "all_population_meshes_joined_one_to_one": True, "existing_diagnostic_status_counts": counts,
              "publication_support_compatibility": group_checks,
              "missing_distance_not_zero_filled": True, "elderly_location_imputed": False,
              "new_distance_calculation_performed": False, "anaSOP_modified": False, "raw_data_modified": False,
              "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in [source, attachment_path, paths_path, membership_path, support_path]],
              "anaSOP_sha256": sop_before}
    assert digest(ROOT / "docs/AnaSOP.md") == sop_before
    (OUT / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    text = """# National medical-access baseline: proposal for confirmation

## Bounded purpose

Prepare the already accepted baseline service-access question without extending bridge investigation or maintenance simulation. This proposal does not select a final outcome, write new processed baseline data or replace AnaSOP. Existing diagnostic distances are inspected for join compatibility, not accepted as observed travel behavior.

## Destination recommendation

Use the 2020 medical register's hospitals (code 1) and general clinics (code 2); retain dental clinics (code 3) in the source inventory but exclude them from this proposed non-dental destination set. The official class definitions are https://nlftp.mlit.go.jp/ksj/gml/codelist/MedClassCd.html. The source archive has 8,269 hospitals, 104,183 general clinics and 68,860 dental clinics. Existing graph entry rules attach all hospitals and 104,132 general clinics; the 51 remaining general clinic records remain retained and flagged.

The cached official source page expressly states that inactive and enterprise-internal facilities are included. Its ten source attributes do not supply a uniform open-to-public operating-status variable. No deletion based on names is proposed. The estimand is potential spatial access to registered facilities under the candidate network scenario, not verified access to currently operating public medical services, emergency capability or clinical capacity. Hospital-only analysis is an alternative human choice and would require recalculating target-specific distances.

## Network and origins

Reuse the previously validated candidate structure-continuity graph and population entry scenario. No new junction, geometry, speed, turn, legal-access, ferry or network correction is introduced. The graph distances use per-tile AEQD lengths and omit attachment connector lengths. The graph is undirected; exact source endpoints and audited structure transitions do not prove legal vehicle routing. Eligible anchors use ordinary non-expressway ground road attachment rules; graph routes may contain other included road classes. Mesh representatives are not observed residences or inferred traversable off-road connectors. The 200 unresolved origins and all records without candidate paths remain retained.

## Publication support and proposed variables

For each official age publication group, retain member-mesh distances and total populations. Propose a population-weighted mean distance conditional on a finite candidate path. Separately report the support population fractions with finite paths, eligible entries without a path, and pending entries. These three statuses partition the support population; a pending entry must not be described as verified geographic disconnection. Do not set missing distances to zero or infinity within the finite-distance mean. A zero denominator produces a missing value and a flag, never row deletion.

Compare this area-level access context with the previously confirmed share aged 65+ among known ages. Total-population weights describe all residents' location context; they neither assign group elderly counts to individual meshes nor measure older residents' individual travel distances. See variable_proposal.csv for the five proposed readable output names and construction rules. Implementation identifiers, coordinates, source-record provenance and categorical status flags are retained as technical metadata.

## Verification and pending choice

The source archive hash and original facility row order, names and class codes match the existing attachment inventory. All 176,964 demographic meshes match existing medical diagnostics one to one; local populations and route-eligibility flags agree. Route status partitions preserve every record and the complete acquired population total. publication-support compatibility is checked without constructing a weighted-distance outcome. The saved summaries are audit evidence, not national access findings.

The human must confirm the destination and aggregation rules before formal baseline preprocessing. No new dependency, source modification, Git/DVC operation or AnaSOP change is made. After confirmation, generate source-traceable member and support baseline tables and reconcile AnaSOP Section 4 through data-preprocessing before planning final figures and tables.

Reproduce with `.venv/bin/python src/exp/prepare_medical_baseline_proposal.py`.
"""
    (OUT / "README.md").write_text(text)
    print(json.dumps({"status": "passed", "medical_categories": categories, "existing_diagnostic_status_counts": counts,
                      "publication_support_compatibility": group_checks, "proposal_adopted": False}, indent=2))


if __name__ == "__main__":
    main()
