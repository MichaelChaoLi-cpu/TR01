"""Prepare a reproducible diagnostic sample; do not accept asset assignments."""
import csv
import hashlib
import io
import json
import sqlite3
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from build_national_bridge_candidates import ROOT, OUT as NATIONAL, ARRAY_FIELDS, digest

OUT = NATIONAL / 'stratified-review'
SEED = 'TR01-bridge-identity-review-v1'
STRATA = ['different_levels', 'shared_feature', 'close_competitors', 'clearer_reference']


def region(pref):
    p = int(pref)
    if p == 1: return 'Hokkaido'
    if p <= 7: return 'Tohoku'
    if p <= 14: return 'Kanto'
    if p <= 23: return 'Chubu'
    if p <= 30: return 'Kinki'
    if p <= 35: return 'Chugoku'
    if p <= 39: return 'Shikoku'
    return 'Kyushu_Okinawa'


def rank(identifier):
    return hashlib.sha256((SEED + '|' + identifier).encode()).hexdigest()


def get_geometry(task):
    tile, wanted = task
    path = ROOT / f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip'
    receipt = json.loads((NATIONAL / 'tiles' / (tile + '.json')).read_text())
    assert digest(path) == receipt['raw_sha256']
    geometries = {}; count = 0
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith('.geojson'))
        with io.TextIOWrapper(archive.open(member), encoding='utf-8-sig') as handle:
            for line in handle:
                if not line.lstrip().startswith('{ "type": "Feature",'): continue
                count += 1
                if count in wanted:
                    feature = json.loads(line.strip().rstrip(','))
                    geometries[f'{tile}:{count}'] = {'coordinates': feature['geometry']['coordinates'], 'properties': feature['properties']}
    assert count == receipt['source_features'] and len(geometries) == len(wanted)
    assert digest(path) == receipt['raw_sha256']
    return geometries


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validation = json.loads((NATIONAL / 'validation.json').read_text())
    assert digest(ROOT / 'docs/AnaSOP.md') == validation['anasop_sha256']
    con = sqlite3.connect(f'file:{NATIONAL / "candidate_summary.sqlite"}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    best = {}; pool_counts = Counter(); eligibility = Counter()
    for original in con.execute("SELECT * FROM bridge_summary WHERE candidate_status='candidate_present' AND source_quality_flags='' AND CAST(bridge_length_m AS REAL)>0"):
        row = dict(original)
        levels = row['near_candidate_min_level'] != row['near_candidate_max_level']
        shared = int(row['assets_sharing_nearest_feature']) > 1
        competitors = int(row['candidates_within_nearest_plus_5m']) > 1
        if levels: stratum = STRATA[0]
        elif shared: stratum = STRATA[1]
        elif competitors: stratum = STRATA[2]
        elif row['nearest_road_state'] == '2' and float(row['nearest_distance_m']) <= 25 and 'named_route_class_conflict' not in row['review_flags']:
            stratum = STRATA[3]
        else: continue
        key = (stratum, region(row['source_prefecture']))
        pool_counts[key] += 1; eligibility[stratum] += 1
        if key not in best or rank(row['shisetsu_id']) < rank(best[key]['shisetsu_id']): best[key] = row
    selected = []; region_load = Counter()
    for stratum in STRATA:
        available = [reg for st,reg in best if st == stratum]
        assert len(available) >= 3
        for _ in range(3):
            reg = min(available, key=lambda r: (region_load[r], rank(stratum + '|' + r)))
            available.remove(reg); region_load[reg] += 1
            selected.append(best[stratum,reg] | {'review_id': f'S{len(selected)+1:02}', 'stratum': stratum, 'region': reg})
    partners = []; all_assets = {int(r['bridge_index']): r for r in selected}
    primary_ids = {r['shisetsu_id'] for r in selected}
    for case in selected:
        case['partner_indices'] = []
        if case['stratum'] != 'shared_feature': continue
        rows = [dict(r) for r in con.execute('SELECT * FROM bridge_summary WHERE nearest_source_id=? AND shisetsu_id!=?', (case['nearest_source_id'], case['shisetsu_id'])) if r['shisetsu_id'] not in primary_ids]
        rows.sort(key=lambda r: rank(r['shisetsu_id']))
        for row in rows[:2]:
            index = int(row['bridge_index']); case['partner_indices'].append(index)
            partners.append({'parent_review_id': case['review_id'], **row})
            all_assets[index] = row
    con.close()
    candidates = defaultdict(list); wanted = defaultdict(set)
    for path in sorted((NATIONAL / 'tiles').glob('*.npz')):
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == receipt['candidate_array_sha256']
        with np.load(path) as data:
            selected_rows = np.flatnonzero(np.isin(data['bridge_index'], list(all_assets)))
            if not len(selected_rows): continue
            for index in selected_rows:
                row = {key: data[key][index].item() for key in ARRAY_FIELDS}
                row['source_id'] = f'{path.stem}:{row["feature_ordinal"]}'
                candidates[row['bridge_index']].append(row); wanted[path.stem].add(row['feature_ordinal'])
    geometries = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for output in pool.map(get_geometry, sorted(wanted.items())): geometries.update(output)
    for index, rows in candidates.items():
        assert len(rows) == int(all_assets[index]['candidates_within_200m'])
        rows.sort(key=lambda r: (r['distance_m'], r['source_id']))
        for row in rows: row.update(geometries[row['source_id']])
        all_assets[index]['candidate_geometries'] = rows
    cases = []
    for case in selected:
        main = all_assets[int(case['bridge_index'])]
        main['partners'] = [all_assets[i] for i in case['partner_indices']]
        cases.append(main)
    (OUT / 'review_cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
    write_csv(OUT / 'sample_assets.csv', [{k:v for k,v in c.items() if k not in ['candidate_geometries','partners','partner_indices']} for c in cases])
    if partners: write_csv(OUT / 'shared_partner_assets.csv', partners)
    write_csv(OUT / 'sampling_pool_counts.csv', [{'stratum':s, 'region':r, 'eligible_assets':n} for (s,r),n in sorted(pool_counts.items())])
    manifest = {'status':'passed', 'design':'Diagnostic mutually exclusive strata; three geographically balanced hash-ranked cases per stratum. Not an accuracy-estimation sample.',
                'seed':SEED, 'priority_order':STRATA, 'main_cases':len(cases), 'context_partner_records':len(partners),
                'regions':dict(region_load), 'prefectures':sorted(set(c['source_prefecture'] for c in cases)),
                'eligibility_counts':dict(eligibility), 'source_tiles_extracted':len(wanted),
                'retained_sample_candidate_pairs':sum(len(rows) for rows in candidates.values()),
                'national_inventory_hash':validation['inventory_sha256'], 'anasop_sha256':validation['anasop_sha256'],
                'national_validation_sha256':digest(NATIONAL/'validation.json'),
                'cases_sha256':digest(OUT/'review_cases.json'), 'raw_hashes_and_ordinals_verified':True,
                'accepted_maintenance_assignments':0}
    (OUT / 'preparation_validation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
