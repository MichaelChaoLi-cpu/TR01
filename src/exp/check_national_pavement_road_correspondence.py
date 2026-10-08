"""Nationwide conditional-anchor feasibility audit, without final assignments.

All original pavement rows are retained. Full traffic-matched road curves are
used. A 50-metre / nearest-plus-5-metre unique, route-compatible scenario is
screened as a proposal, not adopted physical interval correspondence.
"""
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import Geod, Transformer

from check_pavement_road_correspondence_sample import GEOMETRY, CROSS, TRAFFIC, REGISTRY

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/exp/feasibility-check/national-pavement-inventory/pavement_basic_index.csv.gz'
OUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/national_road_correspondence'
GEOD = Geod(ellps='WGS84')
RADIUS = 200.
BATCH = 2000


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def metric_curves(curves, lon, lat):
    """Vectorized point-centred AEQD; validate against per-point transformers."""
    counts = shapely.get_num_coordinates(curves)
    coords = shapely.get_coordinates(curves)
    azimuth, _, distance = GEOD.inv(np.repeat(lon, counts), np.repeat(lat, counts),
                                  coords[:, 0], coords[:, 1])
    radians = np.deg2rad(azimuth)
    xy = np.column_stack((distance * np.sin(radians), distance * np.cos(radians)))
    return shapely.set_coordinates(curves.copy(), xy)


def route_number(name):
    name = unicodedata.normalize('NFKC', str(name))
    match = re.match(r'^国道\s*(\d+)\s*号(?:$|[_\s(])', name)
    return int(match.group(1)) if match else -1


def validate_metric(curves, source):
    rng = np.random.default_rng(20261007)
    positions = rng.choice(len(curves), size=100, replace=False)
    points = shapely.line_interpolate_point(curves[positions], .45, normalized=True)
    lon, lat = shapely.get_x(points) + .001, shapely.get_y(points) - .001
    actual = metric_curves(curves[positions], lon, lat)
    errors = []
    for i in range(len(actual)):
        transform = Transformer.from_crs(6668,
            f'+proj=aeqd +lat_0={lat[i]} +lon_0={lon[i]} +datum=WGS84 +units=m', always_xy=True)
        expected = shapely.transform(curves[positions[i]], transform.transform, interleaved=False)
        e = float(np.max(np.abs(shapely.get_coordinates(expected) - shapely.get_coordinates(actual[i]))))
        errors.append(e)
        assert e < 1e-5, e
    return {'independent_point_projection_checks': len(errors),
            'maximum_coordinate_difference_m': max(errors), 'tolerance_m': 1e-5,
            'coordinate_interpretation': 'Same geographic compatibility assumption as prior sample; source datum correspondence remains unverified.'}


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'candidate_pairs').mkdir(exist_ok=True)
    used = [SOURCE, GEOMETRY, CROSS, TRAFFIC, REGISTRY, ROOT / 'docs/AnaSOP.md']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in used}
    receipt = json.loads((GEOMETRY.parent / 'geometry_validation.json').read_text())
    assert hashes[str(GEOMETRY.relative_to(ROOT))] == receipt['output_sha256']
    assert hashes[str(CROSS.relative_to(ROOT))] == receipt['crosswalk_sha256']
    assert hashes[str(SOURCE.relative_to(ROOT))] == json.loads((SOURCE.parent / 'validation.json').read_text())['index_sha256']
    source = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    assert len(source) == 472720 and source['shisetsu_id'].is_unique
    cross = pd.read_parquet(CROSS, columns=['Graph Source Edge ID', 'Registry Group Index',
        'Baseline Source Road Length', 'Traffic Observation Key', 'Observed Traffic Volume',
        'N13 Road Level', 'N13 Road State', 'Candidate Road Source Record'])
    geometry = pd.read_parquet(GEOMETRY, columns=['Graph Source Edge ID', 'Geometry WKB'])
    np.testing.assert_array_equal(geometry.iloc[:, 0], cross['Graph Source Edge ID'])
    ids = cross['Graph Source Edge ID'].to_numpy()
    curves = shapely.from_wkb(geometry['Geometry WKB'].to_numpy())
    assert len(curves) == 1183367 and shapely.is_valid(curves).all()
    assert (~shapely.is_empty(curves)).all() and (shapely.get_type_id(curves) == 1).all()
    tree = shapely.STRtree(curves)
    metric_validation = validate_metric(curves, source)
    write_json(OUT / 'metric_validation.json', metric_validation)
    registry = pd.read_parquet(REGISTRY, columns=['Traffic Observation Key', 'Target N13 Road Class'])
    traffic = pd.read_parquet(TRAFFIC, columns=['Traffic Observation Key', 'Traffic Road Class', 'Traffic Route Number'])
    traffic = traffic.loc[traffic['Traffic Road Class'].astype(int).eq(3)].copy()
    # Several geometry-registry rows can share one measured observation key.
    # Match route identities by observation key, not the last registry row.
    obs_to_group = {str(k): i for i, k in enumerate(registry['Traffic Observation Key'].unique())}
    registry_obs_group = np.array([obs_to_group[str(k)] for k in registry['Traffic Observation Key']])
    keys = []
    for key, value in traffic[['Traffic Observation Key', 'Traffic Route Number']].itertuples(index=False, name=None):
        if str(key) in obs_to_group and str(value).isdigit():
            assert int(value) < 10000
            keys.append(obs_to_group[str(key)] * 10000 + int(value))
    compatible_keys = np.unique(keys)
    group = cross['Registry Group Index'].to_numpy(dtype=int)
    np.testing.assert_array_equal(cross['Traffic Observation Key'].astype(str),
                                  registry['Traffic Observation Key'].astype(str).to_numpy()[group])
    observation_group = registry_obs_group[group]
    road_class = registry['Target N13 Road Class'].to_numpy(dtype=int)[group]
    levels = cross['N13 Road Level'].to_numpy(dtype=int)
    names = source['syogen.rosen.meisyou']
    routes = names.map({n: route_number(n) for n in names.unique()}).to_numpy(dtype=int)
    unparsed = source.loc[routes < 0, ['syogen.rosen.meisyou']].value_counts().rename('records').reset_index()
    unparsed.to_csv(OUT / 'unparsed_route_names.csv', index=False)
    lon = source['syogen.ichi.keido'].astype(float).to_numpy()
    lat = source['syogen.ichi.ido'].astype(float).to_numpy()
    assert np.isfinite(lon).all() and np.isfinite(lat).all()
    assert ((122 <= lon) & (lon <= 154) & (20 <= lat) & (lat <= 46)).all()
    n = len(source)
    near = np.full(n, np.inf)
    nearest_index = np.full(n, -1, dtype=int)
    candidate_count = np.zeros(n, dtype=int)
    ambiguity_count = np.zeros(n, dtype=int)
    level_count = np.zeros(n, dtype=int)
    compatible_count = np.zeros(n, dtype=int)
    compatible50 = np.zeros(n, dtype=int)
    nearest_compatible = np.zeros(n, dtype=bool)
    source_order = np.arange(n)
    total_pairs = 0
    sample_checks = 0
    previous = pd.read_csv(OUT.parent / 'road_correspondence_sample/point_candidate_summary.csv', dtype={'pavement_record_id': str})
    previous = previous.set_index('pavement_record_id')
    pref_reports = []
    for pref in sorted(source['source_prefecture'].unique()):
        all_positions = source_order[source['source_prefecture'].eq(pref)]
        writer = None
        pair_file = OUT / 'candidate_pairs' / f'pref_{pref}.parquet'
        for start in range(0, len(all_positions), BATCH):
            positions = all_positions[start:start+BATCH]
            # 0.01 degrees encloses 200 m throughout the checked Japan extent.
            boxes = shapely.box(lon[positions]-.01, lat[positions]-.01,
                                lon[positions]+.01, lat[positions]+.01)
            p, g = tree.query(boxes)
            p = positions[p]
            lines = metric_curves(curves[g], lon[p], lat[p])
            distances = shapely.distance(lines, shapely.Point(0, 0))
            keep = distances <= RADIUS
            p, g, distances, lines = p[keep], g[keep], distances[keep], lines[keep]
            if not len(p):
                continue
            order = np.lexsort((ids[g], distances, p))
            p, g, distances, lines = p[order], g[order], distances[order], lines[order]
            first = np.r_[True, p[1:] != p[:-1]]
            near[p[first]] = distances[first]
            nearest_index[p[first]] = g[first]
            compatible = (routes[p] > 0) & (road_class[g] == 1) & np.isin(observation_group[g]*10000+routes[p], compatible_keys)
            nearest_compatible[p[first]] = compatible[first]
            np.add.at(candidate_count, p, 1)
            ambiguous = distances <= near[p]+5.
            np.add.at(ambiguity_count, p[ambiguous], 1)
            np.add.at(compatible_count, p[compatible], 1)
            np.add.at(compatible50, p[compatible & (distances <= 50)], 1)
            distinct_levels = np.unique(np.column_stack((p[ambiguous], levels[g[ambiguous]])), axis=0)
            np.add.at(level_count, distinct_levels[:, 0], 1)
            pairs = pd.DataFrame({'pavement_source_row': p, 'pavement_record_id': source['shisetsu_id'].to_numpy()[p],
                'graph_source_edge_id': ids[g], 'distance_m': distances,
                'point_position_on_curve_m': shapely.line_locate_point(lines, shapely.Point(0, 0)),
                'projected_curve_length_m': shapely.length(lines),
                'traffic_route_number_compatible': compatible, 'road_level': levels[g]})
            table = pa.Table.from_pandas(pairs, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(pair_file, table.schema, compression='zstd')
            writer.write_table(table)
            total_pairs += len(pairs)
        if writer:
            writer.close()
        pref_reports.append({'prefecture': pref, 'source_records': len(all_positions),
            'candidate_within_200m': int(np.isfinite(near[all_positions]).sum()),
            'nearest_within_50m': int((near[all_positions] <= 50).sum())})
        write_json(OUT / 'progress.json', {'completed_prefectures': len(pref_reports),
            'source_records_processed': sum(r['source_records'] for r in pref_reports),
            'candidate_pairs_written': total_pairs, 'elapsed_seconds': round(time.monotonic()-started, 1)})
        print(json.dumps(pref_reports[-1]), flush=True)
    diagnostics = source.copy()
    diagnostics['parsed_national_route_number'] = routes
    diagnostics['nearest_distance_m'] = np.where(np.isfinite(near), near, np.nan)
    diagnostics['candidate_count_within_200m'] = candidate_count
    diagnostics['candidate_count_within_nearest_plus_5m'] = ambiguity_count
    diagnostics['distinct_levels_within_nearest_plus_5m'] = level_count
    diagnostics['route_compatible_count_within_200m'] = compatible_count
    diagnostics['route_compatible_count_within_50m'] = compatible50
    diagnostics['nearest_route_compatible'] = nearest_compatible
    edge = np.full(n, -1, dtype=np.int64)
    found = nearest_index >= 0
    edge[found] = ids[nearest_index[found]]
    diagnostics['nearest_graph_source_edge_id'] = edge
    grade = source['tenken.kiroku.hantei_kubun'].isin(['1','2','3']).to_numpy()
    # Exclusive screening reasons are a review proposal, not a final sample.
    reason = np.select([~grade, routes < 0, near > 50., ambiguity_count != 1,
                        ~nearest_compatible],
        ['unusable_inspection_category', 'unparsed_route_name', 'no_traffic_road_within_50m',
         'multiple_near_nearest_roads', 'nearest_route_incompatible'], default='proposed_conditional_anchor')
    diagnostics['proposed_screening_reason'] = reason
    diagnostics.to_parquet(OUT / 'point_candidate_diagnostics.parquet', index=False)
    proposals = diagnostics.loc[reason == 'proposed_conditional_anchor'].copy()
    proposals['observed_grade_number'] = proposals['tenken.kiroku.hantei_kubun'].astype(int)
    units = proposals.groupby('nearest_graph_source_edge_id').agg(
        anchored_inspection_records=('shisetsu_id','count'),
        minimum_observed_grade=('observed_grade_number','min'),
        maximum_observed_grade=('observed_grade_number','max'),
        source_prefectures=('source_prefecture','nunique')).reset_index()
    units = units.merge(cross, left_on='nearest_graph_source_edge_id', right_on='Graph Source Edge ID', validate='one_to_one')
    units['mixed_observed_grades'] = units['minimum_observed_grade'] != units['maximum_observed_grade']
    units.to_parquet(OUT / 'proposed_road_grade_aggregation.parquet', index=False)
    pd.DataFrame(pref_reports).to_csv(OUT / 'prefecture_candidate_coverage.csv', index=False)
    coverage = diagnostics.groupby('source_prefecture')['proposed_screening_reason'].value_counts().unstack(fill_value=0)
    coverage.to_csv(OUT / 'prefecture_screening_proposal.csv')
    for key, row in previous.iterrows():
        q = diagnostics.loc[diagnostics['shisetsu_id'].eq(key)].iloc[0]
        assert int(q['candidate_count_within_200m']) == row['candidates_within_200m']
        if pd.notna(row['nearest_candidate_distance_m']):
            assert abs(q['nearest_distance_m']-row['nearest_candidate_distance_m']) < 1e-5
        sample_checks += 1
    summary = {'source_records': n, 'prefectures': 47, 'comparison_domain_roads': len(curves),
        'candidate_pairs': total_pairs, 'points_with_candidate_within_200m': int(np.isfinite(near).sum()),
        'points_with_candidate_within_25m': int((near <= 25).sum()),
        'points_with_candidate_within_50m': int((near <= 50).sum()),
        'points_with_multiple_near_nearest_roads': int((ambiguity_count > 1).sum()),
        'points_with_multiple_near_nearest_levels': int((level_count > 1).sum()),
        'points_with_route_compatible_candidate_within_50m': int((compatible50 > 0).sum()),
        'unparsed_route_records': int((routes < 0).sum()),
        'exclusive_proposed_screening_counts': pd.Series(reason).value_counts().to_dict(),
        'proposed_anchored_records': len(proposals), 'proposed_unique_road_units': len(units),
        'proposed_unit_prefectures': proposals['source_prefecture'].nunique(),
        'roads_with_multiple_records': int((units['anchored_inspection_records'] > 1).sum()),
        'roads_with_mixed_observed_grades': int(units['mixed_observed_grades'].sum()),
        'roads_anchored_by_records_from_multiple_prefectures': int((units['source_prefectures'] > 1).sum()),
        'proposed_unit_total_length_m': float(units['Baseline Source Road Length'].sum()),
        'candidate_grade_aggregation': 'Minimum and maximum reported for review; no final aggregation adopted.',
        'proposed_anchor_rule': 'Usable grade I-III, parsed national route, nearest traffic-matched full curve within 50 m, exactly one candidate within nearest plus 5 m, nearest national-route-compatible.',
        'accepted_physical_interval_matches': 0, 'accepted_policy_assignments': 0,
        'nationwide_matching_accuracy_established': False,
        'elapsed_seconds': round(time.monotonic()-started, 1)}
    assert total_pairs == int(candidate_count.sum())
    assert sum(summary['exclusive_proposed_screening_counts'].values()) == n
    assert int(units['anchored_inspection_records'].sum()) == len(proposals)
    # A network source curve need not share a single administrative source
    # prefecture. Retain such cases and report them; charge the edge once.
    assert units['nearest_graph_source_edge_id'].is_unique
    assert (units['Baseline Source Road Length'] > 0).all()
    assert all(digest(ROOT / p) == h for p,h in hashes.items())
    validation = {'status': 'passed_national_diagnostic_only', 'all_source_rows_retained': True,
        'candidate_pair_counts_reconcile': True, 'exclusive_screening_counts_reconcile': True,
        'unique_road_counts_do_not_repeat_cost': True, 'previous_sample_reproduced_records': sample_checks,
        'metric_validation': metric_validation, 'source_inputs_and_anasop_unchanged': True,
        'shared_observation_registry_rows_preserved': True,
        'accepted_policy_assignments': 0, 'input_sha256': hashes}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'validation.json', validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
