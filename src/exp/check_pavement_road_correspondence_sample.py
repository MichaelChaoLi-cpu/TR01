"""Local feasibility screen of saved pavement points against traffic-matched roads.

Full original road curves are used. No physical interval match or policy input
assignment is accepted; the comparison domain is explicitly traffic-matched.
"""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / 'data/exp/feasibility-check/pavement-scope-review/public_api_probe'
OUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/road_correspondence_sample'
GEOMETRY = ROOT / 'data/exp/figure-table-generation/road_service_dependence_and_measured_traffic/source_road_display_geometry.parquet'
CROSS = ROOT / 'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet'
TRAFFIC = ROOT / 'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet'
REGISTRY = ROOT / 'data/exp/national-traffic-geometry/registry/geometry_registry.parquet'
RADIUS = 200.0


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [PROBE / 'sample_response.json'] + sorted(PROBE.glob('pref_[0-9][0-9].json'))
    used = sources + [GEOMETRY, CROSS, TRAFFIC, REGISTRY, ROOT / 'docs/AnaSOP.md']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in used}
    receipt = json.loads((GEOMETRY.parent / 'geometry_validation.json').read_text())
    assert hashes[str(GEOMETRY.relative_to(ROOT))] == receipt['output_sha256']
    assert hashes[str(CROSS.relative_to(ROOT))] == receipt['crosswalk_sha256']
    requests = json.loads((PROBE / 'prefecture_requests.json').read_text())
    for request in requests:
        assert hashes[str((PROBE / ('pref_' + request['prefecture'] + '.json')).relative_to(ROOT))] == request['sha256']
    assert hashes[str((PROBE / 'sample_response.json').relative_to(ROOT))] == json.loads((PROBE / 'request.json').read_text())['sha256']
    records = {}
    appearances = Counter()
    for path in sources:
        obj = json.loads(path.read_text())
        assert obj['resultset']['is_error'] is False
        for row in obj['result']:
            key = row['shisetsu_id']
            if key in records:
                assert records[key] == row
            records[key] = row
            appearances[key] += 1
    assert len(records) == 56
    registry = pd.read_parquet(REGISTRY, columns=['Target N13 Road Class'])
    cross = pd.read_parquet(CROSS, columns=['Graph Source Edge ID', 'Registry Group Index',
                                         'Candidate Road Source Record', 'Baseline Source Road Length',
                                         'Traffic Observation Key', 'Observed Traffic Volume',
                                         'N13 Road Level', 'N13 Road State']).set_index('Graph Source Edge ID')
    geometry = pd.read_parquet(GEOMETRY, columns=['Graph Source Edge ID', 'Geometry WKB'])
    assert len(geometry) == 1183367 and geometry['Graph Source Edge ID'].is_unique
    ids = geometry['Graph Source Edge ID'].to_numpy()
    np.testing.assert_array_equal(ids, cross.index.to_numpy())
    curves = shapely.from_wkb(geometry['Geometry WKB'].to_numpy())
    assert shapely.is_valid(curves).all() and (~shapely.is_empty(curves)).all()
    tree = shapely.STRtree(curves)
    traffic = pd.read_parquet(TRAFFIC, columns=['Traffic Observation Key', 'Traffic Road Class', 'Traffic Route Number'])
    national = traffic[traffic['Traffic Road Class'].astype(int) == 3]
    route_numbers = {}
    for key, group in national.groupby('Traffic Observation Key'):
        route_numbers[str(key)] = {int(value) for value in group['Traffic Route Number'] if str(value).isdigit()}
    summaries, pairs = [], []
    for key, record in sorted(records.items(), key=lambda item: int(item[0])):
        source = record['syogen']
        lon, lat = float(source['ichi']['keido']), float(source['ichi']['ido'])
        assert np.isfinite([lon, lat]).all() and 122 <= lon <= 154 and 20 <= lat <= 46
        # This envelope exceeds the metric radius everywhere in Japan. Exact
        # distances use a point-centred metric projection and the full curves.
        local = tree.query(shapely.box(lon-.01, lat-.01, lon+.01, lat+.01))
        transform = Transformer.from_crs(6668, f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m', always_xy=True)
        lines = shapely.transform(curves[local], transform.transform, interleaved=False)
        point = shapely.Point(*transform.transform(lon, lat))
        distances = shapely.distance(lines, point)
        chosen = np.flatnonzero(distances <= RADIUS)
        chosen = sorted(chosen, key=lambda i: (distances[i], int(ids[local[i]])))
        name = source.get('rosen', {}).get('meisyou', '')
        match = re.fullmatch(r'国道\s*(\d+)\s*号', name)
        route = int(match.group(1)) if match else None
        flags = []
        for rank, i in enumerate(chosen, 1):
            edge = int(ids[local[i]])
            metadata = cross.loc[edge]
            length = float(shapely.length(lines[i]))
            along = float(shapely.line_locate_point(lines[i], point))
            road_class = int(registry.iloc[int(metadata['Registry Group Index'])]['Target N13 Road Class'])
            compatible = road_class == 1 and route in route_numbers.get(str(metadata['Traffic Observation Key']), set()) if route is not None else None
            flags.append(compatible)
            pairs.append({'pavement_record_id': key, 'candidate_rank': rank, 'graph_source_edge_id': edge,
                          'candidate_source_record': metadata['Candidate Road Source Record'],
                          'distance_m': float(distances[i]), 'point_position_on_curve_m': along,
                          'projected_curve_length_m': length, 'baseline_source_road_length_m': float(metadata['Baseline Source Road Length']),
                          'pavement_interval_length_m': source.get('enchou'),
                          'road_level': int(metadata['N13 Road Level']), 'road_state': int(metadata['N13 Road State']),
                          'road_class': road_class, 'traffic_observation_key': metadata['Traffic Observation Key'],
                          'observed_traffic_volume': float(metadata['Observed Traffic Volume']),
                          'api_route_number': route, 'traffic_route_number_compatible': compatible,
                          'physical_interval_match_accepted': False})
        nearest = float(distances[chosen[0]]) if chosen else None
        summaries.append({'pavement_record_id': key, 'source_appearances': appearances[key],
                          'prefecture': source['gyousei_kuiki']['todoufuken_code'], 'route_name': name,
                          'source_interval_label': source['shisetsu']['meisyou'], 'longitude': lon, 'latitude': lat,
                          'interval_length_m': source.get('enchou'), 'inspection_year': record['tenken'].get('nendo'),
                          'inspection_code': record['tenken'].get('kiroku', {}).get('hantei_kubun'),
                          'candidates_within_200m': len(chosen),
                          'nearest_candidate_distance_m': nearest,
                          'candidates_within_nearest_plus_5m': sum(distances[i] <= nearest+5 for i in chosen) if chosen else 0,
                          'route_compatible_candidates_within_200m': sum(flag is True for flag in flags),
                          'route_compatible_candidates_within_50m': sum(flag is True and distances[i] <= 50 for flag, i in zip(flags, chosen)),
                          'nearest_candidate_source_record': cross.loc[int(ids[local[chosen[0]]]), 'Candidate Road Source Record'] if chosen else '',
                          'nearest_candidate_baseline_length_m': float(cross.loc[int(ids[local[chosen[0]]]), 'Baseline Source Road Length']) if chosen else None,
                          'physical_interval_match_accepted': False})
    write_csv('point_candidate_summary.csv', summaries)
    write_csv('point_road_candidate_pairs.csv', pairs)
    finite = [row['nearest_candidate_distance_m'] for row in summaries if row['nearest_candidate_distance_m'] is not None]
    nearest_sources = Counter(row['nearest_candidate_source_record'] for row in summaries if row['nearest_candidate_source_record'])
    summary = {'sample_records': len(records), 'source_appearances': sum(appearances.values()), 'prefectures': len({row['prefecture'] for row in summaries}),
               'comparison_domain_roads': len(geometry), 'diagnostic_radius_m': RADIUS,
               'points_with_candidate_within_200m': len(finite),
               'points_with_candidate_within_25m': sum(value <= 25 for value in finite),
               'points_with_candidate_within_50m': sum(value <= 50 for value in finite),
               'nearest_distance_median_m': float(np.median(finite)), 'nearest_distance_max_within_radius_m': max(finite),
               'points_with_multiple_candidates_within_nearest_plus_5m': sum(row['candidates_within_nearest_plus_5m'] > 1 for row in summaries),
               'points_with_route_compatible_candidate_within_50m': sum(row['route_compatible_candidates_within_50m'] > 0 for row in summaries),
               'points_with_route_compatible_candidate_within_200m': sum(row['route_compatible_candidates_within_200m'] > 0 for row in summaries),
               'nearest_source_roads_shared_by_multiple_sample_records': sum(count > 1 for count in nearest_sources.values()),
               'accepted_physical_interval_matches': 0, 'nationwide_matching_accuracy_established': False,
               'comparison_domain': 'Previously accepted measured-traffic road correspondence; not all source roads.',
               'sample_design': 'First records from the public API; convenience sample, not random or representative.',
               'coordinate_interpretation': 'Geographic API points treated as compatible with the candidate geographic locator; datum correspondence remains unverified.'}
    assert all(digest(ROOT / path) == value for path, value in hashes.items())
    validation = {'status': 'passed_diagnostic_only', 'sample_identity_deduplication_verified': True,
                  'all_47_prefectures_represented': summary['prefectures'] == 47,
                  'geometry_identity_matches_crosswalk': True, 'full_source_road_curves_used': True,
                  'candidate_rows_reconcile': len(pairs) == sum(row['candidates_within_200m'] for row in summaries),
                  'source_inputs_and_anasop_unchanged': True, 'accepted_policy_assignments': 0, 'input_sha256': hashes}
    assert validation['all_47_prefectures_represented'] and validation['candidate_rows_reconcile']
    summary = {key: value.item() if isinstance(value, np.generic) else value for key, value in summary.items()}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
