"""Summarize all national candidate pairs, retaining unresolved asset records.

Distances, radius, ambiguity bands and lexical route hints are diagnostics only.
Full pairs remain in tile arrays; this summary never accepts a maintenance unit.
"""
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_national_bridge_candidates import OUT, ROOT, SOURCE, RADIUS, digest


def class_hint(name, prefecture):
    text = unicodedata.normalize('NFKC', name).strip()
    if text.startswith('高速自動車国道'):
        return 4
    if text.startswith(('一般国道', '国道')):
        return 1
    if text.startswith(('県道', '都道', '府道', '道道')) or text.startswith(prefecture + '道'):
        return 2
    if text.startswith(('市道', '町道', '村道', '区道', '特別区道')):
        return 3
    return 0


def write_csv(name, rows, fields):
    with (OUT / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    config = json.loads((OUT / 'candidate_config.json').read_text())
    summaries = json.loads((OUT / 'tile_summary.json').read_text())
    assert len(summaries) == 160 and len({r['tile'] for r in summaries}) == 160
    assert digest(SOURCE) == config['inventory_sha256']
    assert digest(ROOT / 'docs/AnaSOP.md') == config['anasop_sha256']
    n = config['source_rows']
    counts = np.zeros(n, dtype=np.int32)
    counts50 = np.zeros(n, dtype=np.int32)
    bridges50 = np.zeros(n, dtype=np.int32)
    nearest = np.full(n, np.inf)
    nearest_structure = np.full(n, np.inf)
    total = 0
    for summary in summaries:
        path = OUT / 'tiles' / (summary['tile'] + '.npz')
        assert digest(path) == summary['candidate_array_sha256']
        with np.load(path) as data:
            ids = data['bridge_index']; distance = data['distance_m']; state = data['road_state']
            assert len(ids) == summary['candidate_pairs']
            assert np.isfinite(distance).all() and (distance >= 0).all() and (distance <= RADIUS + 1e-7).all()
            assert np.all((ids >= 0) & (ids < n))
            np.add.at(counts, ids, 1)
            np.add.at(counts50, ids[distance <= 50], 1)
            structural = state == 2
            np.add.at(bridges50, ids[structural & (distance <= 50)], 1)
            np.minimum.at(nearest, ids, distance)
            np.minimum.at(nearest_structure, ids[structural], distance[structural])
            total += len(ids)
    assert int(counts.sum()) == total == sum(r['candidate_pairs'] for r in summaries)
    sentinel = np.iinfo(np.uint64).max
    nearest_key = np.full(n, sentinel, dtype=np.uint64)
    close_count = np.zeros(n, dtype=np.int32)
    level_min = np.full(n, 32767, dtype=np.int16)
    level_max = np.full(n, -32768, dtype=np.int16)
    nearest_fields = {name: np.full(n, np.nan) for name in
                      ['distance_m', 'position_m', 'endpoint_distance_m', 'feature_length_m',
                       'road_class', 'road_state', 'road_level', 'road_width_class']}
    # A one-micrometre tie tolerance makes source-key tie resolution explicit.
    for summary in summaries:
        with np.load(OUT / 'tiles' / (summary['tile'] + '.npz')) as data:
            ids = data['bridge_index']; distance = data['distance_m']; fids = data['feature_ordinal']
            assert (fids > 0).all() and (fids < 2**32).all()
            key = (np.uint64(int(summary['tile'])) << np.uint64(32)) | fids.astype(np.uint64)
            close = distance <= nearest[ids] + 5
            np.add.at(close_count, ids[close], 1)
            np.minimum.at(level_min, ids[close], data['road_level'][close].astype(np.int16))
            np.maximum.at(level_max, ids[close], data['road_level'][close].astype(np.int16))
            tied = distance <= nearest[ids] + 1e-6
            np.minimum.at(nearest_key, ids[tied], key[tied])
            selected = tied & (key == nearest_key[ids])
            selected_ids = ids[selected]
            for name in nearest_fields:
                nearest_fields[name][selected_ids] = data[name][selected]
    matched = np.isfinite(nearest)
    assert np.all(nearest_key[matched] != sentinel)
    assert np.all(np.abs(nearest_fields['distance_m'][matched] - nearest[matched]) <= 1e-6)
    shared_counts = np.zeros(n, dtype=np.int32)
    keys, inverse, frequencies = np.unique(nearest_key[matched], return_inverse=True, return_counts=True)
    shared_counts[matched] = frequencies[inverse]
    shared_rows = [{'source_id': f'{int(k >> np.uint64(32)):04}:{int(k & np.uint64(2**32-1))}',
                    'assets_with_this_nearest_candidate': int(freq)} for k,freq in zip(keys,frequencies) if freq > 1]
    write_csv('shared_nearest_features.csv', shared_rows, ['source_id', 'assets_with_this_nearest_candidate'])
    quality = {}
    quality_path = SOURCE.parent / 'quality_flags.csv.gz'
    with gzip.open(quality_path, 'rt', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            quality[row['shisetsu_id']] = row['flags']
    coordinates = np.load(OUT / 'point_coordinates.npy', mmap_mode='r')
    fields = ['bridge_index', 'shisetsu_id', 'bridge_name', 'route_name', 'source_prefecture',
              'source_longitude', 'source_latitude', 'inspection_year', 'health_code',
              'bridge_length_m', 'bridge_width_m', 'coordinate_status', 'candidate_status',
              'candidates_within_200m', 'candidates_within_50m', 'bridge_geometry_candidates_within_50m',
              'nearest_source_id', 'nearest_distance_m', 'nearest_bridge_geometry_distance_within_200m',
              'nearest_source_length_m', 'source_to_asset_length_ratio', 'nearest_road_class',
              'nearest_road_state', 'nearest_road_level', 'nearest_road_width_class',
              'nearest_position_m', 'nearest_endpoint_distance_m', 'route_name_class_hint',
              'candidates_within_nearest_plus_5m', 'near_candidate_min_level', 'near_candidate_max_level',
              'assets_sharing_nearest_feature', 'source_quality_flags', 'review_flags',
              'source_page', 'source_row', 'accepted_match', 'intervention_extent_status']
    database_path = OUT / 'candidate_summary.sqlite'
    database = sqlite3.connect(database_path)
    database.execute('DROP TABLE IF EXISTS bridge_summary')
    declaration = ','.join(f'"{field}" TEXT' for field in fields)
    database.execute('CREATE TABLE bridge_summary (' + declaration + ')')
    insert = 'INSERT INTO bridge_summary VALUES (' + ','.join('?' for _ in fields) + ')'
    batch = []
    status_counts, flags, coverage = Counter(), Counter(), defaultdict(Counter)
    examples = defaultdict(list)
    queue_rows = 0
    with gzip.open(SOURCE, 'rt', encoding='utf-8', newline='') as source, \
         gzip.open(OUT / 'bridge_candidate_summary.csv.gz', 'wt', encoding='utf-8', newline='') as output, \
         gzip.open(OUT / 'review_queue.csv.gz', 'wt', encoding='utf-8', newline='') as queue:
        writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
        queue_writer = csv.DictWriter(queue, fieldnames=fields); queue_writer.writeheader()
        for i, asset in enumerate(csv.DictReader(source)):
            valid = bool(np.isfinite(coordinates[i]).all())
            present = bool(matched[i])
            status = 'candidate_present' if present else 'no_candidate_within_200m' if valid else 'coordinate_withheld'
            pref = asset['source_pref_query']
            hint = class_hint(asset['syogen.rosen.meisyou'], asset['syogen.gyousei_kuiki.todoufuken_mei'])
            reasons = []
            if not valid: reasons.append('coordinate_withheld')
            elif not present: reasons.append('no_candidate_within_200m')
            if present and nearest[i] > 50: reasons.append('nearest_beyond_50m')
            if present and bridges50[i] == 0: reasons.append('no_bridge_geometry_within_50m')
            if present and hint and hint != nearest_fields['road_class'][i]: reasons.append('named_route_class_conflict')
            if present and nearest_fields['road_state'][i] == 3: reasons.append('nearest_source_is_tunnel')
            if close_count[i] > 1: reasons.append('multiple_candidates_within_nearest_plus_5m')
            if present and level_min[i] != level_max[i]: reasons.append('near_candidates_have_different_levels')
            if shared_counts[i] > 1: reasons.append('shared_nearest_source_feature')
            if '旧' in unicodedata.normalize('NFKC', asset['syogen.shisetsu.meisyou']): reasons.append('old_name_indicator_requires_review')
            if asset['shisetsu_id'] in quality: reasons.append('source_quality_flag')
            try: length = float(asset['syogen.kyouchou'])
            except ValueError: length = math.nan
            key = nearest_key[i]
            source_id = f'{int(key >> np.uint64(32)):04}:{int(key & np.uint64(2**32-1))}' if present else ''
            scalar = lambda values: float(values[i]) if present else ''
            row = dict(zip(fields, [''] * len(fields)))
            row.update(bridge_index=i, shisetsu_id=asset['shisetsu_id'], bridge_name=asset['syogen.shisetsu.meisyou'],
                       route_name=asset['syogen.rosen.meisyou'], source_prefecture=pref,
                       source_longitude=asset['syogen.ichi.keido'], source_latitude=asset['syogen.ichi.ido'],
                       inspection_year=asset['tenken.nendo'], health_code=asset['tenken.kiroku.hantei_kubun'],
                       bridge_length_m=asset['syogen.kyouchou'], bridge_width_m=asset['syogen.fukuin'],
                       coordinate_status='source_geographic_screen_passed' if valid else 'withheld',
                       candidate_status=status, candidates_within_200m=int(counts[i]), candidates_within_50m=int(counts50[i]),
                       bridge_geometry_candidates_within_50m=int(bridges50[i]), nearest_source_id=source_id,
                       nearest_distance_m=float(nearest[i]) if present else '',
                       nearest_bridge_geometry_distance_within_200m=float(nearest_structure[i]) if math.isfinite(nearest_structure[i]) else '',
                       nearest_source_length_m=scalar(nearest_fields['feature_length_m']),
                       source_to_asset_length_ratio=float(nearest_fields['feature_length_m'][i]/length) if present and length > 0 else '',
                       nearest_road_class=int(nearest_fields['road_class'][i]) if present else '',
                       nearest_road_state=int(nearest_fields['road_state'][i]) if present else '',
                       nearest_road_level=int(nearest_fields['road_level'][i]) if present else '',
                       nearest_road_width_class=int(nearest_fields['road_width_class'][i]) if present else '',
                       nearest_position_m=scalar(nearest_fields['position_m']), nearest_endpoint_distance_m=scalar(nearest_fields['endpoint_distance_m']),
                       route_name_class_hint=hint, candidates_within_nearest_plus_5m=int(close_count[i]),
                       near_candidate_min_level=int(level_min[i]) if present else '', near_candidate_max_level=int(level_max[i]) if present else '',
                       assets_sharing_nearest_feature=int(shared_counts[i]), source_quality_flags=quality.get(asset['shisetsu_id'], ''),
                       review_flags=';'.join(reasons), source_page=asset['source_page'], source_row=asset['source_row'],
                       accepted_match=False, intervention_extent_status='unresolved')
            writer.writerow(row)
            if reasons:
                queue_writer.writerow(row); queue_rows += 1
            batch.append(tuple(row[field] for field in fields))
            if len(batch) >= 10000:
                database.executemany(insert, batch); database.commit(); batch.clear()
            status_counts[status] += 1; flags.update(reasons)
            coverage[pref]['source_records'] += 1
            coverage[pref][status] += 1
            coverage[pref]['nearest_within_50m'] += int(present and nearest[i] <= 50)
            coverage[pref]['bridge_geometry_within_50m'] += int(bridges50[i] > 0)
            for reason in reasons:
                if len(examples[reason]) < 5: examples[reason].append(row)
    database.executemany(insert, batch); database.commit()
    database.execute('CREATE UNIQUE INDEX facility_id_lookup ON bridge_summary(shisetsu_id)')
    database.execute('CREATE INDEX source_feature_lookup ON bridge_summary(nearest_source_id)')
    database.commit()
    assert database.execute('SELECT count(*) FROM bridge_summary').fetchone()[0] == n == sum(status_counts.values())
    assert database.execute("SELECT count(*) FROM bridge_summary WHERE accepted_match != '0'").fetchone()[0] == 0
    database.close()
    with gzip.open(OUT / 'bridge_candidate_summary.csv.gz', 'rt', newline='', encoding='utf-8') as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == n
    write_csv('prefecture_candidate_coverage.csv', [{'source_prefecture':p, **dict(coverage[p])} for p in sorted(coverage)],
              ['source_prefecture'] + sorted({key for counter in coverage.values() for key in counter}))
    write_csv('review_flag_counts.csv', [{'flag':key, 'asset_records':value} for key,value in sorted(flags.items())], ['flag','asset_records'])
    write_csv('review_examples.csv', [dict(row, example_reason=reason) for reason, rows in examples.items() for row in rows], ['example_reason'] + fields)
    validation = {'status':'passed', 'inventory_records_retained':n, 'source_tiles':160,
                  'source_features_scanned':sum(r['source_features'] for r in summaries),
                  'eligible_road_features':sum(r['eligible_features'] for r in summaries),
                  'candidate_pairs':total, 'candidate_status_counts':dict(status_counts),
                  'assets_nearest_within_50m':int(np.sum(nearest <= 50)),
                  'assets_bridge_geometry_within_50m':int(np.sum(bridges50 > 0)),
                  'nearest_distance_median':float(np.median(nearest[matched])),
                  'nearest_distance_p95':float(np.quantile(nearest[matched],.95)),
                  'shared_nearest_feature_groups':len(shared_rows),
                  'assets_sharing_nearest_feature':int(np.sum(shared_counts > 1)),
                  'review_queue_records':queue_rows, 'review_flag_counts':dict(flags),
                  'source_hash_and_feature_ordinal_checks_passed':all(r['raw_unchanged'] and r['source_ordinal_count_validated'] for r in summaries),
                  'tile_array_hashes_verified':True, 'saved_summary_rows_verified':True,
                  'inventory_sha256':config['inventory_sha256'], 'anasop_unchanged':True,
                  'anasop_sha256':config['anasop_sha256'], 'accepted_maintenance_matches':0,
                  'intervention_extents_unresolved':n, 'national_match_accuracy_estimated':False,
                  'alternative_coordinates_in_primary_run':False}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
