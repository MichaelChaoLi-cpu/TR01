"""Preserve full source-road curves for the adopted traffic-comparison map."""
from pathlib import Path
import csv
import hashlib
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/figure-table-generation/road_service_dependence_and_measured_traffic'
CROSS = ROOT / 'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet'
GRAPH = ROOT / 'data/processed/national-network-rule-validation'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'source_road_display_geometry.parquet'
    cross_hash = digest(CROSS)
    if output.exists() and (OUT / 'geometry_validation.json').exists():
        receipt = json.loads((OUT / 'geometry_validation.json').read_text())
        assert receipt['crosswalk_sha256'] == cross_hash and receipt['output_sha256'] == digest(output)
        print('Verified full-curve display cache.', flush=True)
        return
    cross = pd.read_parquet(CROSS, columns=['Graph Source Edge ID'])
    ids = np.sort(cross['Graph Source Edge ID'].to_numpy(dtype=np.int64))
    assert len(ids) == 1183367 and len(np.unique(ids)) == len(ids)
    offsets = np.load(GRAPH / 'tile_offsets.npy')
    ordinals = np.load(GRAPH / 'edge_source_ordinals.npy', mmap_mode='r')
    source_hashes = {r['filename']:r['sha256'] for r in csv.DictReader((ROOT / 'data/exp/data-acquisition/n13_2024_download_manifest.csv').open())}
    count = 0
    verified_hashes = {}
    writer = None
    started = time.perf_counter()
    try:
        for ti, (tile, start, size) in enumerate(offsets):
            tile, start, size = int(tile), int(start), int(size)
            selected = ids[np.searchsorted(ids,start):np.searchsorted(ids,start+size)]
            if not len(selected):
                continue
            wanted = ordinals[selected]
            path = ROOT / f'data/raw/roads/n13_2024_geojson/N13-24_{tile:04d}_GEOJSON.zip'
            assert digest(path) == source_hashes[path.name]
            pointer = 0
            feature_ordinal = 0
            edge_ids, coordinates, ends = [], [], [0]

            def flush():
                nonlocal writer, count
                if not edge_ids:
                    return
                curves = shapely.from_ragged_array(shapely.GeometryType.LINESTRING, np.asarray(coordinates),
                                                  (np.asarray(ends, dtype=np.int64),))
                assert shapely.is_valid(curves).all() and (~shapely.is_empty(curves)).all()
                middle = shapely.line_interpolate_point(curves, .5, normalized=True)
                xy = shapely.get_coordinates(middle)
                table = pa.table({'Graph Source Edge ID': np.asarray(edge_ids,dtype=np.int32),
                                  'Geometry WKB': shapely.to_wkb(curves),
                                  'Display Longitude': xy[:,0], 'Display Latitude':xy[:,1]})
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression='zstd')
                writer.write_table(table)
                count += len(edge_ids)
                edge_ids.clear(); coordinates.clear(); ends[:] = [0]

            with zipfile.ZipFile(path) as archive:
                name = next(n for n in archive.namelist() if n.endswith('.geojson'))
                with io.TextIOWrapper(archive.open(name), encoding='utf-8-sig') as handle:
                    for line in handle:
                        if not line.lstrip().startswith('{ "type": "Feature",'):
                            continue
                        feature_ordinal += 1
                        if pointer == len(wanted):
                            break
                        if feature_ordinal != wanted[pointer]:
                            continue
                        feature = json.loads(line.strip().rstrip(','))
                        assert feature['geometry']['type'] == 'LineString'
                        edge_ids.append(int(selected[pointer])); pointer += 1
                        coordinates.extend(feature['geometry']['coordinates']); ends.append(len(coordinates))
                        if len(edge_ids) == 25000:
                            flush()
                flush()
            assert pointer == len(wanted)
            assert digest(path) == source_hashes[path.name]
            verified_hashes[str(path.relative_to(ROOT))] = source_hashes[path.name]
            progress = {'completed_source_tiles':len(verified_hashes), 'road_curves_saved':count,
                        'requested_road_curves':len(ids), 'elapsed_seconds':time.perf_counter()-started}
            (OUT / 'geometry_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(f'Full source curves {count:,}/{len(ids):,}', flush=True)
    finally:
        if writer:
            writer.close()
    assert count == len(ids)
    np.testing.assert_array_equal(pd.read_parquet(output,columns=['Graph Source Edge ID']).iloc[:,0],ids)
    assert digest(CROSS) == cross_hash
    receipt = {'status':'passed','source_road_curves':count,'crosswalk_sha256':cross_hash,
               'output_sha256':digest(output),'source_hashes_verified_unchanged':verified_hashes,
               'full_source_curves':True,'endpoint_chord_substitution':False,
               'display_midpoint_role':'View partition locator only; no population allocation or physical disruption footprint.'}
    (OUT / 'geometry_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'Full-curve map metadata validated: {count:,} records.',flush=True)


if __name__ == '__main__':
    main()
