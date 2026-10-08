"""Bounded input audit for the next planned road-service-dependence figure.

Reads existing traffic samples and graph provenance. No matching rule, traffic
assignment, closure scenario or scientific variable is adopted by this audit.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/estimation-framework-planning/road-service-dependence"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    sources = {}
    section_rows = []
    audit_rows = []
    source_keys = set()
    configuration = [
        ("Tokyo", "source-matching-sample", "traffic_tokyo_2021.csv", "traffic_point_candidates.csv"),
        ("Nikko", "bridge-access-rural-sample", "traffic_tochigi_2021.csv", "mountain_traffic_point_candidates.csv"),
    ]
    for area, directory, filename, points_filename in configuration:
        folder = ROOT / "data/exp/feasibility-check" / directory
        source = folder / "sources" / filename
        sources[str(source.relative_to(ROOT))] = digest(source)
        rows = list(csv.reader(source.read_text(encoding="cp932").splitlines()))
        header, records = rows[0], rows[1:]
        assert len(header) == 159 and all(len(row) == 159 for row in records)
        by_id = {row[0].zfill(11): row for row in records}
        assert len(by_id) == len(records)
        geometries = sorted((folder / "sources").glob("*.geojson"))
        selected = []
        for geometry in geometries:
            sources[str(geometry.relative_to(ROOT))] = digest(geometry)
            document = json.loads(geometry.read_text())
            assert document["features"]
            assert all(str(feature["properties"]["R03cen"]) == geometry.stem for feature in document["features"])
            row = by_id[geometry.stem]
            selected.append(row)
            section_rows.append({
                "Area": area, "Survey section ID": geometry.stem,
                "Traffic observation unit source": row[31], "Up observation section source": row[32],
                "Down observation section source": row[44], "Road class source": row[3],
                "Up observation flag source": row[34], "Up 12/24-hour flag source": row[36],
                "Down observation flag source": row[46], "Down 12/24-hour flag source": row[48],
                "12-hour vehicle total source": row[58], "24-hour vehicle total source": row[61],
                "Approved graph traffic assignment": False,
            })
        points_path = folder / points_filename
        sources[str(points_path.relative_to(ROOT))] = digest(points_path)
        with points_path.open(newline="") as handle:
            points = list(csv.DictReader(handle))
        source_keys.update(row["nearest_n13_id"] for row in points)
        assert all(row["accepted_match"].lower() == "false" for row in points)
        audit_rows.append({
            "Area": area, "Available traffic table rows": len(records),
            "Available geometry sections": len(geometries), "Verified section-to-CSV joins": len(selected),
            "Distinct traffic observation units": len({row[31] for row in selected}),
            "Distinct up observation sections": len({row[32] for row in selected}),
            "Distinct down observation sections": len({row[44] for row in selected}),
            "Candidate sample points": len(points),
            "Points with multiple nearest-plus-5m candidates": sum(int(row["candidates_within_nearest_plus_5m"]) > 1 for row in points),
            "Approved graph traffic assignments": 0,
        })
    graph_base = ROOT / "data/processed/national-network-quality"
    paths = [graph_base / "tile_offsets.npy", graph_base / "edge_source_ordinals.npy",
             graph_base / "edge_lengths.npy",
             ROOT / "data/processed/national-network-rule-validation/edge_nodes_exact.npy"]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = digest(path)
    offsets = np.load(paths[0], mmap_mode="r")
    ordinals = np.load(paths[1], mmap_mode="r")
    lengths = np.load(paths[2], mmap_mode="r")
    nodes = np.load(paths[3], mmap_mode="r")
    assert len(ordinals) == len(lengths) == len(nodes) == 21415800
    tile_offsets = {str(int(tile)): (int(start), int(count)) for tile, start, count in offsets}
    mapped = []
    for key in sorted(source_keys):
        tile, ordinal = key.split(":")
        start, count = tile_offsets[tile]
        subset = ordinals[start:start + count]
        matches = np.flatnonzero(subset == int(ordinal))
        assert len(matches) == 1, key
        edge = start + int(matches[0])
        assert np.isfinite(lengths[edge]) and lengths[edge] > 0
        mapped.append({"Candidate source road key": key, "Graph source edge ID": edge,
                       "Length (m)": float(lengths[edge]), "First node": int(nodes[edge, 0]),
                       "Second node": int(nodes[edge, 1]), "Approved traffic match": False})
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("traffic_section_inventory.csv", section_rows),
                           ("traffic_input_audit.csv", audit_rows),
                           ("candidate_source_graph_lookup.csv", mapped)]:
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, expected in sources.items():
        assert digest(ROOT / name) == expected
    receipt = {
        "status": "input_audit_passed_scenario_and_traffic_definitions_pending",
        "planned_output": "Road Service Dependence and Measured Traffic",
        "traffic_samples": audit_rows, "geometry_sections": len(section_rows),
        "distinct_candidate_source_roads_found_in_existing_graph": len(mapped),
        "approved_traffic_assignments": 0, "new_simulation_performed": False,
        "national_traffic_dataset_complete": False,
        "source_hashes_verified_unchanged": sources,
        "interpretation": "Section-to-table joins and candidate source-key lookup pass. This is neither an accepted spatial crosswalk nor nationally representative traffic coverage. Repeated traffic observation units must not be counted as independent measurements. Source 12/24-hour and observation flags remain undecoded source codes in this audit.",
    }
    (OUT / "input_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "source_hashes_verified_unchanged"}, indent=2))


if __name__ == "__main__":
    main()
