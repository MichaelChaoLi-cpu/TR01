"""Validate saved visual judgments and produce a review ledger; never edit the graph.

Map inspection is manual. Rerunning this script validates and summarizes the
saved observations; it does not independently reproduce their visual judgment.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/network-map-review"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    queue = read_csv(OUT / "review_queue.csv")
    annotations = json.loads((OUT / "visual_judgments.json").read_text())
    ids = [row["review_id"] for row in queue]
    assert len(ids) == len(set(ids)) == 360
    assert set(ids) == set(annotations), "Missing or extraneous visual judgments"
    statuses = {"connect_supported", "no_connect_supported", "unresolved"}
    results = []
    for row in queue:
        observation = annotations[row["review_id"]]
        assert observation["status"] in statuses and observation["basis"].strip()
        card = OUT / "cards" / (row["review_id"] + ".png")
        sheet = OUT / "sheets" / f"review_{observation['sheet']:02}.png"
        assert card.is_file() and sheet.is_file()
        if observation["status"] == "connect_supported":
            action = ("join_reviewed_participating_endpoint_nodes"
                      if row["kind"] == "multilevel_endpoint"
                      else "split_and_join_reviewed_participating_lines")
        elif observation["status"] == "no_connect_supported":
            action = "retain_separate_participating_lines"
        else:
            action = "hold_for_additional_evidence"
        results.append(dict(row, **observation, proposed_action=action,
                            reviewer="Codex visual source review",
                            review_date="2026-10-06",
                            evidence_card=str(card.relative_to(ROOT)),
                            evidence_card_sha256=hashlib.sha256(card.read_bytes()).hexdigest(),
                            evidence_sheet=str(sheet.relative_to(ROOT)),
                            evidence_limit="Mixed/unknown imagery dates; no dated ground truth or vehicle access verification"))
    tiles = read_csv(OUT / "tile_manifest.csv")
    assert len(tiles) == 1802
    for row in tiles:
        assert row["status"] != "failed" and not row["error"]
        tile = OUT / "tile_cache" / row["filename"]
        payload = tile.read_bytes()
        assert len(payload) == int(row["bytes"])
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    acquisition = json.loads((OUT / "map_acquisition.json").read_text())
    assert acquisition["failed_tiles"] == acquisition["cards_with_missing_tiles"] == 0
    counts = Counter(row["status"] for row in results)
    grade = Counter(row["status"] for row in results if row["kind"] == "multilevel_endpoint")
    crossing = Counter(row["status"] for row in results if row["kind"] != "multilevel_endpoint")
    write_csv(OUT / "review_results.csv", results)
    write_csv(OUT / "unresolved_queue.csv", [row for row in results if row["status"] == "unresolved"])
    write_csv(OUT / "proposed_topology_actions.csv", [row for row in results if row["status"] == "connect_supported"])
    summary = dict(review_records=len(results), overall=dict(counts),
                   multilevel_endpoints=dict(grade), same_level_intersection_records=dict(crossing),
                   supported_crossing_connections=[row["review_id"] for row in results
                       if row["kind"] != "multilevel_endpoint" and row["status"] == "connect_supported"],
                   tile_hashes_validated=len(tiles), validation="passed",
                   graph_modified=False, raw_data_modified=False,
                   unresolved_policy="Retain records without asserting connectivity; do not automatically delete roads",
                   reviewer="Codex visual source review, not a recorded human research decision")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# Pilot network map and imagery review

All {len(results)} candidate records were reviewed against GSI standard maps and seamless aerial photographs. These are agent visual observations and proposed actions, not approved nationwide topology rules. The pilot graph and raw data were not changed.

| Candidate group | Connection supported | Separation supported | Unresolved | Total |
|---|---:|---:|---:|---:|
| Multilevel shared endpoints | {grade['connect_supported']} | {grade['no_connect_supported']} | {grade['unresolved']} | 151 |
| Same-level intersection records | {crossing['connect_supported']} | {crossing['no_connect_supported']} | {crossing['unresolved']} | 209 |
| Total | {counts['connect_supported']} | {counts['no_connect_supported']} | {counts['unresolved']} | 360 |

## Interpretation and proposed treatment

1. Reviewed bridge approaches frequently continue across a change in the local level code. Requiring identical level codes at every endpoint would break these supported connections. Join only the participating endpoint nodes at the recorded location after checking the geometry; do not connect nearby crossing roads.
2. Many interior geometric crossings occur at separate decks or tunnel alignments, even when both lines have the same level code. Splitting and connecting every same-code intersection would create unsupported shortcuts.
3. Three interior/touch records support a surface junction: C001-1, C003-1 and C044-1. These are candidates for a selective pilot split-and-join operation, subject to preserving the original source identity and lengths.
4. Unresolved records remain in the ledger. Vegetation, ambiguous portal or ramp geometry, and a mapped corridor missing from older imagery prevent reliable judgments. They must not be silently connected or treated as observed road closures.

The previous count of 204 referred to crossing **line pairs**, not distinct geographic sites. The complete queue includes 204 crossing pairs and one non-endpoint touch pair; multipoint geometries produce 209 intersection records, plus 151 endpoint records. Multiple records can concern the same physical structure. Counts are candidate records, not unique junction counts.

## Evidence and limitations

Each row of [review_results.csv](review_results.csv) includes source IDs, coordinates, proposed action, basis, and an evidence card with a SHA-256 hash. [unresolved_queue.csv](unresolved_queue.csv) contains pending cases; [proposed_topology_actions.csv](proposed_topology_actions.csv) contains only supported connection candidates. Cards are in `cards/`, and 23 contact sheets are in `sheets/`.

Source: [Geospatial Information Authority of Japan tile catalogue](https://maps.gsi.go.jp/development/ichiran.html), layers `std` and `seamlessphoto`, zoom 18, downloaded on 2026-10-06. Maps have source line overlays; aerial photographs retain their pixels except a candidate marker. Source road-state codes are interpreted as physical structures, not deterioration measures.

The two layers do not provide synchronized, candidate-specific acquisition dates in this review. A visual verdict expresses support from the available sources; it does not establish the exact 2024 network, present legal vehicle access, one-way rules, turning permissions, ambulance availability, or actual response time. Source-coded tunnels help interpret otherwise hidden alignments, so some judgments are not independent of the road source. No national error rate can be inferred from this geographically selected pilot.

## Reproduction and validation

Run from the repository root:

```sh
.venv/bin/python src/exp/prepare_network_map_review.py
.venv/bin/python src/exp/fetch_network_review_maps.py
.venv/bin/python src/exp/finalize_network_map_review.py
```

`visual_judgments.json` stores manual visual observations. Regenerating cards does not regenerate these judgments; inspect changed evidence before reusing annotations. Finalization checks exact coverage of 360 unique record IDs, valid status and basis, existence of every card and referenced sheet, and hashes and byte counts for all 1,802 cached tiles. Validation passed. It writes only exploratory review outputs.
"""
    (OUT / "README.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
