"""Full-curve candidate diagnostics for existing official traffic sample lines.

This is a bounded method check, not a national correspondence or traffic
assignment. The 50 m search and 20 m overlap buffer are diagnostic distances,
not accepted matching thresholds. Every source feature keeps its true ordinal.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/exp/national-traffic-geometry/line-correspondence-check'
PROC = ROOT/'data/processed/national-network-rule-validation'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traffic = pd.read_parquet(ROOT/'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet').set_index('Traffic Geometry Section')
    paths = []
    for name in ['source-matching-sample', 'bridge-access-rural-sample']:
        paths.extend(sorted((ROOT/'data/exp/feasibility-check'/name/'sources').glob('*.geojson')))
    records = []
    for path in paths:
        section = path.stem; row = traffic.loc[section]
        assert row['Traffic Measurement Eligible']
        data = json.loads(path.read_text())
        geometries = [shapely.from_geojson(json.dumps(f['geometry'])) for f in data['features']]
        records.append({'section': section, 'unit': row['Traffic Observation Key'],
                        'class': int(row['Traffic Road Class']), 'geometry': shapely.union_all(geometries)})
    # Join fragments from the same observed unit for candidate diagnostics.
    units = sorted(set(r['unit'] for r in records))
    geometries, classes = [], []
    for unit in units:
        selected = [r for r in records if r['unit'] == unit]
        geometries.append(shapely.union_all([r['geometry'] for r in selected]))
        assert len(set(r['class'] for r in selected)) == 1
        classes.append(selected[0]['class'])
    geographic_bounds = shapely.bounds(geometries)
    offsets = np.load(PROC/'tile_offsets.npy')
    ordinals = np.load(PROC/'edge_source_ordinals.npy', mmap_mode='r')
    audited = {r['archive']: int(r['features']) for r in csv.DictReader((ROOT/'data/exp/road-quality-check/archive_summary.csv').open())}
    hashes, results, tiles = {}, [], []
    # Traffic code 1/2 expressway, 3 national, 4/6 prefectural, 5/7 city.
    # N13's official code list: national=1, prefectural=2,
    # municipal=3, expressway=4, other=5, unknown=6.
    compatible_classes = {1: 4, 2: 4, 3: 1, 4: 2, 5: 3, 6: 2, 7: 3}
    for tile, start, count in offsets:
        tile = int(tile); code = f'{tile:04d}'
        x0, y0 = 100+int(code[2:]), int(code[:2])/1.5
        active = np.flatnonzero((geographic_bounds[:,0] <= x0+1+.001) & (geographic_bounds[:,2] >= x0-.001) &
                                (geographic_bounds[:,1] <= y0+2/3+.001) & (geographic_bounds[:,3] >= y0-.001))
        if not len(active): continue
        path = ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip'
        with path.open('rb') as handle: before = hashlib.file_digest(handle, 'sha256').hexdigest()
        hashes[str(path.relative_to(ROOT))] = before
        transformer = Transformer.from_crs(6668, f'+proj=aeqd +lat_0={y0+1/3} +lon_0={x0+.5} +datum=WGS84 +units=m', always_xy=True)
        projected_traffic = shapely.transform(np.asarray(geometries, dtype=object)[active], transformer.transform, interleaved=False)
        tree = shapely.STRtree(projected_traffic)
        buffered = shapely.buffer(projected_traffic, 20)
        local_ordinals = ordinals[int(start):int(start+count)]
        assert (np.diff(local_ordinals) > 0).all()
        total = 0
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith('.geojson'))
            with io.TextIOWrapper(archive.open(name), encoding='utf-8-sig') as handle:
                for line in handle:
                    if not line.lstrip().startswith('{ "type": "Feature",'): continue
                    feature = json.loads(line.strip().rstrip(',')); total += 1
                    props = feature['properties']
                    if str(props['N13_002']) != '1' or str(props['N13_004']) in ['5','6','7']: continue
                    coords = np.asarray(feature['geometry']['coordinates'])
                    lo, hi = coords.min(axis=0), coords.max(axis=0)
                    boxes = geographic_bounds[active]
                    near = (lo[0] <= boxes[:,2]+.001) & (hi[0] >= boxes[:,0]-.001) & (lo[1] <= boxes[:,3]+.001) & (hi[1] >= boxes[:,1]-.001)
                    if not near.any(): continue
                    index = int(np.searchsorted(local_ordinals, total))
                    if index >= len(local_ordinals) or local_ordinals[index] != total: continue
                    road = shapely.transform(shapely.linestrings(coords), transformer.transform, interleaved=False)
                    length = float(shapely.length(road))
                    if not length: continue
                    for match in tree.query(road, predicate='dwithin', distance=50):
                        global_unit = active[match]
                        overlap = float(shapely.length(shapely.intersection(road, buffered[match]))/length)
                        samples = shapely.line_interpolate_point(road, np.linspace(0,1,5), normalized=True)
                        distances = shapely.distance(samples, projected_traffic[match])
                        results.append({'Traffic Observation Key': units[global_unit],
                                        'Candidate Road Source Record': f'{code}:{total}',
                                        'Graph Source Edge ID': int(start)+index,
                                        'Traffic Road Class': classes[global_unit], 'N13 Road Class': int(props['N13_003']),
                                        'Road Class Compatible': compatible_classes[classes[global_unit]] == int(props['N13_003']),
                                        'N13 Road Level': int(props['N13_005']), 'N13 Road State': int(props['N13_004']),
                                        'Source Road Length': length,
                                        'Minimum Line Separation': float(shapely.distance(road, projected_traffic[match])),
                                        'Road Fraction Within Diagnostic Buffer': min(overlap,1.),
                                        'Maximum Sampled Line Separation': float(distances.max()),
                                        'Median Sampled Line Separation': float(np.median(distances)),
                                        'Result Role': 'Unaccepted line-correspondence candidate'})
        assert total == audited[path.name]
        with path.open('rb') as handle: assert hashlib.file_digest(handle,'sha256').hexdigest() == before
        tiles.append(code)
        print(f'Checked source curves in {code}: {total:,} source records; accumulated {len(results):,} candidates', flush=True)
    frame = pd.DataFrame(results)
    frame.to_parquet(OUT/'line_candidates.parquet', index=False)
    frame.to_csv(OUT/'line_candidates.csv', index=False)
    inventory = []
    for unit in units:
        subset = frame[frame['Traffic Observation Key'].eq(unit)]
        inventory.append({'Traffic Observation Key':unit,'Sample Geometry Sections':sum(r['unit']==unit for r in records),
                          'Total Source Sections Sharing Observation':int(traffic['Traffic Observation Key'].eq(unit).sum()),
                          'Candidate Road Records':len(subset),'Road Class Compatible Candidates':int(subset['Road Class Compatible'].sum())})
    pd.DataFrame(inventory).to_csv(OUT/'observation_unit_coverage.csv',index=False)
    receipt = {'status':'passed', 'source_geometry_sections':len(records), 'distinct_observation_units':len(units),
               'source_tiles':tiles, 'line_candidate_pairs':len(frame), 'accepted_correspondences':0,
               'full_curved_source_geometry':True, 'source_ordinals_checked':True, 'source_hashes_unchanged':hashes,
               'diagnostic_search_metres':50,'diagnostic_overlap_buffer_metres':20,
               'coordinate_limit':'Viewer coordinates interpreted geographically; datum correspondence remains unverified.',
               'coverage_limit':'Existing purposive traffic sample; no national accuracy or complete observation-unit geometry claim.',
               'no_final_traffic_assignments':True}
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__': main()
