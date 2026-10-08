"""Reproduce the national baseline and benchmark four source-road removals.

The four cases are technical checks, not representative national findings.
"""
from pathlib import Path
import gc
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
from road_service_loss import RoadServiceLoss, loss_summary

OUT = ROOT / "data/exp/road-service-loss-validation"
PROC = ROOT / "data/processed/national-network-rule-validation"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [PROC / "graph_exact.npz", PROC / "edge_nodes_exact.npy", PROC / "edge_lengths.npy",
               PROC / "edge_source_ordinals.npy", PROC / "tile_offsets.npy",
               ROOT / "data/exp/national-population-entry/point_attachment_diagnostics.csv",
               ROOT / "data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet",
               ROOT / "data/exp/national-population-entry/representative_routes.geojson"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    sample = pd.read_parquet(sources[6])
    assert len(sample) == 175612
    points = pd.read_csv(sources[5], dtype={"point_id": str, "category": str, "population_raw": str,
                         "road_source_id": str}, keep_default_na=False, low_memory=False,
                         float_precision="round_trip")
    points["along_m"] = pd.to_numeric(points["along_m"], errors="coerce")
    origins = points.set_index("point_id").loc["mesh:"+sample["Mesh Code"].astype(str)].reset_index()
    np.testing.assert_array_equal(pd.to_numeric(origins["population_raw"]), sample["Local Total Population"])
    assert origins["routing_eligible"].all()
    facilities = points[points["dataset"].eq("medical") & points["category"].isin(["1", "2"]) & points["routing_eligible"]].copy()
    assert len(facilities) == 112401
    pairs = np.load(sources[1], mmap_mode="r")
    lengths = np.load(sources[2], mmap_mode="r")
    graph = load_npz(sources[0])
    print(f"Loaded graph: {graph.shape[0]:,} nodes; {len(pairs):,} source roads", flush=True)
    kernel = RoadServiceLoss(pairs, lengths, origins["edge_id"].to_numpy(), origins["along_m"].to_numpy(),
                             facilities["edge_id"].to_numpy(), facilities["along_m"].to_numpy(), graph=graph)
    del graph, points
    gc.collect()
    baseline_start = time.perf_counter()
    baseline = kernel.distances()
    baseline_seconds = time.perf_counter()-baseline_start
    expected = sample["Candidate Medical Network Distance"].to_numpy()
    np.testing.assert_allclose(baseline, expected, rtol=1e-11, atol=1e-7)
    maximum_error = float(np.max(np.abs(baseline-expected)))
    np.save(OUT / "reproduced_baseline_distances.npy", baseline)
    print(f"Baseline reproduced: max error {maximum_error:.3g} m; {baseline_seconds:.2f} seconds", flush=True)
    offsets = np.load(sources[4]); ordinals = np.load(sources[3], mmap_mode="r")
    def source_key(edge):
        i = int(np.searchsorted(offsets[:, 1], edge, side="right")-1)
        return f"{int(offsets[i, 0]):04d}:{int(ordinals[edge])}"
    # Bounded technical cases: an existing traffic-candidate road, an origin
    # entry serving the largest selected mesh, and a facility-heavy source road.
    candidates = pd.read_csv(ROOT / "data/exp/estimation-framework-planning/road-service-dependence/candidate_source_graph_lookup.csv")
    origin_edge = int(origins.iloc[int(np.argmax(sample["Local Total Population"].to_numpy()))]["edge_id"])
    counts = facilities.groupby("edge_id").size()
    facility_edge = int(counts.idxmax())
    route_parts = [feature["properties"] for feature in json.loads(sources[7].read_text())["features"]
                   if feature["properties"]["path_id"] == "medical:02_northeast"]
    route_parts.sort(key=lambda item: item["segment_order"])
    route_edge = int(route_parts[len(route_parts)//2]["edge_id"])
    cases = [("Existing traffic-candidate source road", int(candidates.iloc[0]["Graph source edge ID"])),
             ("Entry road of the largest selected population mesh", origin_edge),
             ("Source road with the most included medical entries", facility_edge),
             ("Middle road of an existing northeastern medical route", route_edge)]
    population = sample["Local Total Population"].to_numpy()
    rows = []
    for label, edge in cases:
        scenario_start = time.perf_counter()
        scenario = kernel.distances(edge)
        summary = loss_summary(baseline, scenario, population)
        summary.update({"Validation Case": label, "Candidate Road Source Record": source_key(edge),
                        "Closed Source Edge ID": edge, "Closed Source Length": float(lengths[edge]),
                        "Closed Origin Entry Population": int(population[origins["edge_id"].to_numpy() == edge].sum()),
                        "Closed Medical Entry Count": int(facilities["edge_id"].eq(edge).sum()),
                        "Scenario Runtime Seconds": time.perf_counter()-scenario_start,
                        "Result Role": "Technical validation only"})
        assert summary["Lost Medical Path Population"]+summary["Remaining Reachable Population"] == int(population.sum())
        rows.append(summary)
        np.save(OUT / f"scenario_distances_{edge}.npy", scenario)
        print(f"Checked {source_key(edge)}: lost paths {summary['Lost Medical Path Population']:,} people; extra distance {summary['Additional Medical Distance Burden']:.3f} person-km; {summary['Scenario Runtime Seconds']:.2f} seconds", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "technical_scenario_results.csv", index=False)
    for name, expected_hash in hashes.items():
        assert digest(ROOT / name) == expected_hash
    receipt = {"status": "passed", "scenario_authorization": "MILI-D-20261007-013",
               "national_baseline_reproduced_meshes": len(sample), "population_accounted": int(population.sum()),
               "included_medical_destinations": len(facilities), "maximum_baseline_error_metres": maximum_error,
               "baseline_runtime_seconds": baseline_seconds, "technical_scenarios": rows,
               "all_scenarios_monotonic_and_population_conserved": True,
               "source_hashes_verified_unchanged": hashes, "wall_seconds": time.perf_counter()-start,
               "national_dependence_complete": False, "traffic_values_assigned": False,
               "role": "Whole-national-graph engineering checks for four purposive road cases; not a national estimate or a final figure"}
    (OUT / "national_benchmark_validation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Passed national baseline and {len(cases)} technical removals in {receipt['wall_seconds']:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
