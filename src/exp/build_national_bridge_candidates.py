"""Bounded-memory, resumable national point-to-N13 candidate screening.

No source coordinates are corrected, no candidate is accepted, and no road is
removed or protected. All ordinary positive-length centerlines of source states
1-4 are retained, including expressways; vehicle permissions remain unverified.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-crosswalk'
SOURCE = ROOT / 'data/exp/feasibility-check/national-bridge-inventory/bridge_basic_index.csv.gz'
RADIUS = 200.0
ARRAY_FIELDS = ['bridge_index', 'feature_ordinal', 'distance_m', 'position_m',
                'endpoint_distance_m', 'feature_length_m', 'road_class', 'road_state',
                'road_level', 'road_width_class']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tiles').mkdir(exist_ok=True)
    source_hash = digest(SOURCE)
    path = OUT / 'point_coordinates.npy'
    config_path = OUT / 'candidate_config.json'
    if config_path.exists() and path.exists():
        old = json.loads(config_path.read_text())
        if old['inventory_sha256'] == source_hash and old['radius_m'] == RADIUS:
            assert digest(path) == old['point_coordinates_sha256']
            return old
        raise RuntimeError('Input/config changed; preserve the previous output before replacing the candidate run')
    coordinates = []
    withheld = []
    with gzip.open(SOURCE, 'rt', encoding='utf-8', newline='') as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            try:
                longitude = float(row['syogen.ichi.keido'])
                latitude = float(row['syogen.ichi.ido'])
            except (ValueError, TypeError):
                longitude = latitude = math.nan
            valid = math.isfinite(longitude) and math.isfinite(latitude) and 122 <= longitude <= 154 and 20 <= latitude <= 46
            coordinates.append((longitude, latitude) if valid else (math.nan, math.nan))
            if not valid:
                withheld.append({'bridge_index': i, 'shisetsu_id': row['shisetsu_id'],
                                 'source_longitude': row['syogen.ichi.keido'], 'source_latitude': row['syogen.ichi.ido'],
                                 'status': 'coordinate_withheld_pending_independent_verification',
                                 'source_page': row['source_page'], 'source_row': row['source_row']})
    np.save(path, np.asarray(coordinates, dtype=np.float64))
    with (OUT / 'coordinate_withheld.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(withheld[0]))
        writer.writeheader(); writer.writerows(withheld)
    config = {'inventory_path': str(SOURCE.relative_to(ROOT)), 'inventory_sha256': source_hash,
              'source_rows': len(coordinates), 'coordinate_withheld': len(withheld),
              'radius_m': RADIUS, 'point_coordinates_sha256': digest(path),
              'coordinate_assumption': 'Explicit API coordinates interpreted geographically; datum remains unverified. No ID-derived coordinate substitution.',
              'road_selection': 'N13_002=1, N13_004 in 1-4, positive length; road category and level are retained without attaching condition to graph edges.',
              'anasop_sha256': digest(ROOT / 'docs/AnaSOP.md'), 'accepted_matches': 0}
    json_write(config_path, config)
    return config


def work(tile, config):
    path = ROOT / f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip'
    result_path = OUT / 'tiles' / (tile + '.json')
    array_path = OUT / 'tiles' / (tile + '.npz')
    if result_path.exists() and array_path.exists():
        result = json.loads(result_path.read_text())
        assert result['inventory_sha256'] == config['inventory_sha256']
        assert result['radius_m'] == RADIUS
        assert digest(array_path) == result['candidate_array_sha256']
        assert digest(path) == result['raw_sha256']
        return result
    source_manifest = {r['filename']: r for r in csv.DictReader((ROOT / 'data/exp/data-acquisition/n13_2024_download_manifest.csv').open())}
    expected_counts = {r['archive']: int(r['features']) for r in csv.DictReader((ROOT / 'data/exp/road-quality-check/archive_summary.csv').open())}
    source_hash = digest(path)
    assert source_hash == source_manifest[path.name]['sha256']
    all_points = np.load(OUT / 'point_coordinates.npy', mmap_mode='r')
    coordinate_blocks, offsets, metadata, total = [], [0], [], 0
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith('.geojson'))
        with io.TextIOWrapper(archive.open(member), encoding='utf-8-sig') as handle:
            for line in handle:
                if not line.lstrip().startswith('{ "type": "Feature",'):
                    continue
                feature = json.loads(line.strip().rstrip(',')); total += 1
                properties = feature['properties']; coords = feature['geometry']['coordinates']
                if str(properties['N13_002']) != '1' or str(properties['N13_004']) not in ['1', '2', '3', '4']:
                    continue
                if all(p == coords[0] for p in coords):
                    continue
                coordinate_blocks.extend(coords); offsets.append(len(coordinate_blocks))
                metadata.append((total, int(properties['N13_003']), int(properties['N13_004']),
                                 int(properties['N13_005']), int(properties['N13_006'])))
    assert total == expected_counts[path.name], 'Line parser must preserve audited source-feature ordinals'
    x0 = 100 + int(tile[2:]); y0 = int(tile[:2]) / 1.5
    transformer = Transformer.from_crs(6668, f'+proj=aeqd +lat_0={y0+1/3} +lon_0={x0+.5} +datum=WGS84 +units=m', always_xy=True)
    arrays = {name: np.empty(0, dtype=np.float64) for name in ARRAY_FIELDS}
    arrays['bridge_index'] = np.empty(0, dtype=np.int32)
    arrays['feature_ordinal'] = np.empty(0, dtype=np.int32)
    local_points = 0
    if metadata:
        coords = np.asarray(coordinate_blocks, dtype=np.float64); del coordinate_blocks
        ends = np.asarray(offsets, dtype=np.int64)
        meta = np.asarray(metadata, dtype=np.int32); del metadata
        lines = shapely.from_ragged_array(shapely.GeometryType.LINESTRING, coords, (ends,))
        projected = shapely.transform(lines, transformer.transform, interleaved=False)
        lengths = shapely.length(projected)
        assert np.all(lengths > 0)
        # Use actual full-coordinate bounds rather than assuming source lines stay inside their tile.
        lower = coords.min(axis=0) - .005; upper = coords.max(axis=0) + .005
        pi = np.flatnonzero((all_points[:,0] >= lower[0]) & (all_points[:,0] <= upper[0]) &
                            (all_points[:,1] >= lower[1]) & (all_points[:,1] <= upper[1]))
        local_points = len(pi)
        if len(pi):
            points = shapely.points(*transformer.transform(all_points[pi,0], all_points[pi,1]))
            indices = shapely.STRtree(projected).query(points, predicate='dwithin', distance=RADIUS)
            if indices.shape[1]:
                point_local, road_local = indices
                chosen_lines = projected[road_local]; chosen_points = points[point_local]
                distances = shapely.distance(chosen_lines, chosen_points)
                along = shapely.line_locate_point(chosen_lines, chosen_points)
                geographic_ends = np.stack((coords[ends[road_local]], coords[ends[road_local+1]-1]), axis=1)
                ex, ey = transformer.transform(geographic_ends[:,:,0], geographic_ends[:,:,1])
                px, py = transformer.transform(all_points[pi[point_local],0], all_points[pi[point_local],1])
                endpoint_distance = np.min(np.hypot(ex-px[:,None], ey-py[:,None]), axis=1)
                arrays = {'bridge_index': pi[point_local].astype(np.int32),
                          'feature_ordinal': meta[road_local,0], 'distance_m': distances,
                          'position_m': along, 'endpoint_distance_m': endpoint_distance,
                          'feature_length_m': lengths[road_local], 'road_class': meta[road_local,1],
                          'road_state': meta[road_local,2], 'road_level': meta[road_local,3],
                          'road_width_class': meta[road_local,4]}
        del coords, lines, projected
    temporary = array_path.with_suffix('.part')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, array_path)
    assert digest(path) == source_hash
    result = {'tile': tile, 'source_features': total, 'eligible_features': len(offsets)-1,
              'local_point_queries': local_points, 'candidate_pairs': len(arrays['bridge_index']),
              'raw_sha256': source_hash, 'raw_unchanged': True,
              'inventory_sha256': config['inventory_sha256'], 'radius_m': RADIUS,
              'local_crs': transformer.target_crs.to_string(), 'candidate_array_sha256': digest(array_path),
              'source_ordinal_count_validated': True, 'accepted_matches': 0}
    json_write(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--tiles', nargs='*')
    args = parser.parse_args()
    config = prepare()
    available = sorted(p.name.split('_')[1] for p in (ROOT / 'data/raw/roads/n13_2024_geojson').glob('*.zip'))
    tiles = args.tiles or available
    assert all(tile in available for tile in tiles)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, tile, config): tile for tile in tiles}
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps({k:result[k] for k in ['tile','source_features','eligible_features','candidate_pairs']}), flush=True)
            json_write(OUT / 'run_progress.json', {'completed_tiles_this_run': len(results),
                       'requested_tiles': len(tiles), 'candidate_pairs_this_run': sum(r['candidate_pairs'] for r in results)})
    assert digest(SOURCE) == config['inventory_sha256']
    assert digest(ROOT / 'docs/AnaSOP.md') == config['anasop_sha256']
    if set(tiles) == set(available):
        assert len(results) == 160
        json_write(OUT / 'tile_summary.json', sorted(results, key=lambda r:r['tile']))
    print(json.dumps({'status': 'candidate_generation_complete', 'tiles': len(results),
                      'candidate_pairs': sum(r['candidate_pairs'] for r in results), 'accepted_matches': 0}), flush=True)


if __name__ == '__main__':
    main()
