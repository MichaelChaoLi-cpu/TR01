"""Validate bounded crosswalk evidence and report the saved visual review.

Visual judgments are human-readable analyst observations in visual_review.csv;
rerunning this script validates their references, not their visual truth.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/bridge-crosswalk-review'


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    cases = read_csv(OUT / 'case_attributes.csv')
    candidates = read_csv(OUT / 'candidate_attributes.csv')
    reviews = read_csv(OUT / 'visual_review.csv')
    prep = json.loads((OUT / 'preparation_validation.json').read_text())
    maps = json.loads((OUT / 'map_validation.json').read_text())
    parent = json.loads((OUT.parent / 'validation.json').read_text())
    by_id = {row['review_id']: row for row in cases}
    assert len(by_id) == len(cases) == 10
    assert len({row['facility_id'] for row in cases}) == 10
    assert len(reviews) == len({row['review_id'] for row in reviews}) == 10
    assert {row['review_id'] for row in reviews} == set(by_id)
    candidate_refs = {(row['review_id'], row['source_id']) for row in candidates}
    for row in reviews:
        case = by_id[row['review_id']]
        if row['supported_source_id']:
            assert (row['review_id'], row['supported_source_id']) in candidate_refs
            assert row['supported_source_id'] == case['nearest_source_id']
        else:
            assert row['association_status'] == 'withheld'
        assert row['extent_status'] == 'unresolved'
        assert (OUT / 'cards' / (row['review_id'] + '.png')).is_file()
    assert all(row['accepted_match'] == 'False' for row in candidates)
    assert maps['failed_tiles'] == maps['missing_card_tiles'] == 0
    tiles = read_csv(OUT / 'tile_manifest.csv')
    assert len(tiles) == maps['tiles'] == 140
    assert all(digest(OUT / 'tile_cache' / row['file']) == row['sha256'] for row in tiles)
    road = ROOT / 'data/raw/roads/n13_2024_geojson/N13-24_5439_GEOJSON.zip'
    source = OUT.parent / 'bridge-access-rural-sample/sources/bridge_nikko_latest.json'
    assert digest(road) == prep['raw_n13_sha256']
    assert digest(source) == prep['asset_source_sha256']
    assert digest(ROOT / 'docs/AnaSOP.md') == parent['anasop_sha256']
    statuses = dict(Counter(row['association_status'] for row in reviews))
    assert sum(statuses.values()) == 10
    rule = {
        'status': 'analyst_proposal_not_accepted_preprocessing_rule',
        'scope': 'Ten convenience-sample latest bridge records in Nikko; no national accuracy estimate.',
        'identity': 'Keep the API facility ID opaque, explicit latitude/longitude, inspection year, retrieval time and source hash. N13 file-row IDs are scoped to the hashed archive.',
        'candidate_search': 'The 200 m window and 50 m screening summary generate candidates only. No universal acceptance radius is established.',
        'route_class': 'For these named sample routes only: national=1, prefectural=2, municipal=3 in N13_003. Class conflict requires review; class agreement does not establish route-number identity.',
        'bridge_state': 'N13_004=2 is bridge/elevated geometry, not inspection condition. An ordinary-road code does not establish absence of a short bridge.',
        'association': 'Use point placement, route class, competing candidates and map/image evidence jointly. Retain provisional, withheld and unresolved records.',
        'old_assets': 'An old-bridge label triggers review. Do not move the record to a new crossing, infer legal closure, or delete it from an image alone.',
        'asset_extent': 'Road-corridor association is separate from physical bridge extent and affected graph links. Neither source-feature length nor bridge length automatically defines an intervention.',
        'shared_features': 'Preserve distinct assets sharing a source feature. Do not duplicate that graph feature as independent repair units or assume one repair protects all assets.',
        'temporal_alignment': 'The latest inspections span 2021-2025, N13 is September 2024, and GSI imagery dates are mixed/unverified. Current presence, permissions and closures are not validated.',
        'national_acquisition': 'A separately flagged national basic inventory can be acquired without treating it as a completed crosswalk. Preserve unmatched cases and report source/manager/year coverage.',
        'allocation_gate': 'Before policy ranking: human-approved decision units, affected graph links, traffic coverage, comparable costs and defensible intervention effects are still required.',
    }
    save_json('matching_rule_proposal.json', rule)
    rows = [
        '# Bridge-to-N13 visual crosswalk review', '',
        'Review date: 2026-10-06. This is a feasibility diagnostic under the existing national all-road research contract. It does not narrow the study to bridges, create analysis-ready assignments, or revise AnaSOP.', '',
        '## Finding', '',
        'Nearest-road matching alone is insufficient. Nine records have provisional corridor associations: four use bridge/elevated source geometry and five use ordinary-road geometry. The old bridge B03 is withheld because the nearest feature is municipal while the asset record names a national road, and the crossing context does not support accepting that feature. All ten intervention extents remain unresolved; accepted asset-to-maintenance assignments: zero.', '',
        'B01 and B02 are distinct 2.0 m and 2.3 m assets whose nearest candidate is the same 768.0 m N13 feature. B05 is a 4 m bridge on a 1171.9 m ordinary-road feature. Removing or protecting an entire source feature would introduce an unvalidated intervention. B06 and B08 have similar asset/feature lengths, but this does not prove endpoint or graph-link correspondence.', '',
        '## Case-level evidence', '',
        '| Case | Asset | Nearest distance (m) | Source state | Association | Extent |',
        '| --- | --- | ---: | --- | --- | --- |',
    ]
    for row in reviews:
        case = by_id[row['review_id']]
        state = 'bridge/elevated' if case['nearest_state'] == '2' else 'ordinary'
        rows.append(f"| [{row['review_id']}](cards/{row['review_id']}.png) | {case['bridge_name']} | {float(case['nearest_distance_m']):.2f} | {state} | {row['association_status']} | unresolved |")
    rows += ['',
        'The four bridge-corridor associations are B04, B06, B08 and B09. B04/B06/B08 have visible crossing context in the aerial cards; B09 is partly obscured by trees. The five short-bridge road associations are B01/B02/B05/B07/B10; the cards support road context, not delineation of each small structure. These are analyst judgments, not observed match ground truth or a national success rate. See [visual_review.csv](visual_review.csv) for observations and limitations for every case.', '',
        '## Sources and interpretation limits', '',
        '- [MLIT N13 September 2024 catalog](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html): source reference date and positional standard deviation of 25 m. This is not a guarantee that every asset point is within 25 m or a matching acceptance radius.',
        '- [N13 road category](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadCategory.html) and [road state](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadState.html): road-manager categories and geometric structure codes; state 2 is not a health category.',
        '- [Public basic bridge API specification](https://rirs.or.jp/tenken-db/pdf/view/api_specification.pdf): ten actual bounded latest-record responses retained in the previous sample. The sample does not validate all national records or simultaneous condition observations.',
        '- [GSI tile list and attribution](https://maps.gsi.go.jp/development/ichiran.html): GSI standard map and seamless aerial image tiles, with N13/API overlays added by the project. Photograph dates are mixed/unverified. The 140 cached tiles all passed recorded-hash validation; ten cards have no missing tiles.', '',
        'The 10 records are the first ten returned by a bounded latest Nikko query, not a probability sample. No motorway interchange, stacked urban corridor or national completeness claim is supported. Inspection years span 2021-2025. The cards cannot establish legal vehicle access, current bridge function, repair costs, or maintenance-caused preservation of access.', '',
        '## Proposed next step', '',
        'Collect the public national basic bridge inventory in resumable batches, preserving unmatched records, provenance and coverage flags. Keep that inventory separate from a completed road crosswalk. Then apply candidate screening and review conflicts/shared features; confirm maintenance decision units and intervention effects before allocation modeling. The current findings do not justify freezing an automatic asset-to-edge rule. See [matching_rule_proposal.json](matching_rule_proposal.json).', '',
        '## Reproduction', '',
        'From the project root:', '',
        '```bash',
        '.venv/bin/python src/exp/prepare_bridge_crosswalk_review.py',
        '.venv/bin/python src/exp/render_bridge_crosswalk_maps.py',
        '.venv/bin/python src/exp/report_bridge_crosswalk_review.py',
        '```', '',
        'The renderer fetches only missing bounded GSI tiles; cached files support rerendering offline. Visual judgments in visual_review.csv were written after inspecting all ten cards and require analyst reinspection if sources change. The report script validates record references, source/tile hashes and raw/AnaSOP preservation; it does not reperform visual interpretation. No raw write, dependency installation, Git or DVC operation is part of this workflow.', '',
    ]
    (OUT / 'README.md').write_text('\n'.join(rows), encoding='utf-8')
    validation = {
        'status': 'passed', 'reviewed_assets': len(reviews), 'candidate_rows': len(candidates),
        'association_status_counts': statuses, 'unresolved_intervention_extents': len(reviews),
        'accepted_maintenance_assignments': 0, 'shared_nearest_features': prep['shared_nearest_features'],
        'tile_hashes_verified': len(tiles), 'raw_n13_unchanged': True, 'asset_source_unchanged': True,
        'anasop_unchanged': True, 'anasop_sha256': parent['anasop_sha256'],
        'national_accuracy_estimated': False, 'scope_revised': False,
        'visual_review_sha256': digest(OUT / 'visual_review.csv'),
        'card_sha256': {row['review_id']: digest(OUT / 'cards' / (row['review_id'] + '.png')) for row in reviews},
    }
    save_json('validation.json', validation)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
