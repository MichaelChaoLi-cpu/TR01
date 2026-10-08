"""Bounded local-source follow-up for pavement points absent from traffic matches."""
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/road_correspondence_sample'
INPUT = OUT / 'point_candidate_summary.csv'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    input_hash = digest(INPUT)
    sop_hash = digest(ROOT / 'docs/AnaSOP.md')
    points = pd.read_csv(INPUT, dtype={'pavement_record_id': str, 'prefecture': str})
    points = points[points['candidates_within_200m'] == 0]
    assert len(points) == 11
    bounds = {str(row.pavement_record_id): (row.longitude-.01, row.latitude-.01,
                                          row.longitude+.01, row.latitude+.01) for row in points.itertuples()}
    available = {path.name.split('_')[1]: path for path in (ROOT / 'data/raw/roads/n13_2024_geojson').glob('*.zip')}
    tiles = set()
    for west, south, east, north in bounds.values():
        for x in range(math.floor(west-100), math.floor(east-100)+1):
            for y in range(math.floor(south*1.5), math.floor(north*1.5)+1):
                code = f'{y:02d}{x:02d}'
                if code in available:
                    tiles.add(code)
    manifest = {row['filename']: row['sha256'] for row in csv.DictReader((ROOT / 'data/exp/data-acquisition/n13_2024_download_manifest.csv').open())}
    counts = {row['archive']: int(row['features']) for row in csv.DictReader((ROOT / 'data/exp/road-quality-check/archive_summary.csv').open())}
    curves, metadata, hashes, feature_counts = [], [], {}, {}
    for code in sorted(tiles):
        path = available[code]
        before = digest(path)
        assert before == manifest[path.name]
        count = 0
        with zipfile.ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if name.endswith('.geojson'))
            with io.TextIOWrapper(archive.open(member), encoding='utf-8-sig') as stream:
                for line in stream:
                    if not line.lstrip().startswith('{ "type": "Feature",'):
                        continue
                    count += 1
                    feature = json.loads(line.strip().rstrip(','))
                    p = feature['properties']
                    if str(p['N13_002']) != '1' or str(p['N13_004']) not in ['1', '2', '3', '4']:
                        continue
                    xy = np.asarray(feature['geometry']['coordinates'])
                    minimum, maximum = xy.min(axis=0), xy.max(axis=0)
                    if not any(maximum[0] >= west and minimum[0] <= east and maximum[1] >= south and minimum[1] <= north
                               for west, south, east, north in bounds.values()):
                        continue
                    curve = shapely.LineString(xy)
                    if curve.length == 0:
                        continue
                    curves.append(curve)
                    metadata.append({'candidate_source_record': f'{code}:{count}', 'road_class': int(p['N13_003']),
                                     'road_level': int(p['N13_005']), 'road_state': int(p['N13_004']),
                                     'road_width_category': int(p['N13_006'])})
        assert count == counts[path.name]
        assert digest(path) == before
        hashes[str(path.relative_to(ROOT))] = before
        feature_counts[code] = count
        print(json.dumps({'completed_source_tile': code, 'source_features_checked': count,
                          'local_curves_retained_so_far': len(curves)}), flush=True)
    curves = np.asarray(curves, dtype=object)
    tree = shapely.STRtree(curves)
    summaries, pairs = [], []
    for row in points.itertuples():
        key = str(row.pavement_record_id)
        local = tree.query(shapely.box(*bounds[key]))
        transform = Transformer.from_crs(6668, f'+proj=aeqd +lat_0={row.latitude} +lon_0={row.longitude} +datum=WGS84 +units=m', always_xy=True)
        lines = shapely.transform(curves[local], transform.transform, interleaved=False)
        point = shapely.Point(*transform.transform(row.longitude, row.latitude))
        distances = shapely.distance(lines, point)
        selected = [int(i) for i in np.flatnonzero(distances <= 200)]
        selected.sort(key=lambda i: (distances[i], metadata[local[i]]['candidate_source_record']))
        for rank, i in enumerate(selected, 1):
            pairs.append({'pavement_record_id': key, 'rank': rank, **metadata[local[i]],
                          'distance_m': float(distances[i]), 'curve_length_m': float(shapely.length(lines[i])),
                          'physical_interval_match_accepted': False})
        summaries.append({'pavement_record_id': key, 'prefecture': row.prefecture, 'route_name': row.route_name,
                          'traffic_matched_candidates_within_200m': 0,
                          'local_source_candidates_within_200m': len(selected),
                          'nearest_local_source_distance_m': float(distances[selected[0]]) if selected else None,
                          'nearest_local_source_record': metadata[local[selected[0]]]['candidate_source_record'] if selected else '',
                          'nearest_local_source_road_class': metadata[local[selected[0]]]['road_class'] if selected else None,
                          'physical_interval_match_accepted': False})
    for name, rows in [('unmatched_local_source_candidates.csv', pairs), ('unmatched_local_source_summary.csv', summaries)]:
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    assert digest(INPUT) == input_hash and digest(ROOT / 'docs/AnaSOP.md') == sop_hash
    summary = {'status': 'passed_diagnostic_only', 'followup_records': len(points), 'source_tiles_checked': len(tiles),
               'source_feature_ordinals_verified': feature_counts,
               'records_with_local_source_candidate_within_200m': sum(row['local_source_candidates_within_200m'] > 0 for row in summaries),
               'records_with_local_source_candidate_within_25m': sum(row['nearest_local_source_distance_m'] is not None and row['nearest_local_source_distance_m'] <= 25 for row in summaries),
               'accepted_policy_assignments': 0, 'raw_source_hashes_unchanged': hashes,
               'input_summary_sha256': input_hash, 'anasop_sha256_unchanged': sop_hash,
               'meaning': 'Nearby source roads can exist where the accepted measured-traffic correspondence supplies no candidate; this is not a verified pavement footprint or traffic imputation.'}
    (OUT / 'unmatched_local_source_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
