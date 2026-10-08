"""Construct the human-adopted bounded policy domain and fixed area groups.

Authorization: MILI-D-20261007-026. Raw sources and existing processed inputs
are read-only. Bridge association is a scenario, never an accepted asset match.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/data-preprocessing/common-budget'
AUTH = 'MILI-D-20261007-026'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rels = {
        'bridges': 'data/exp/feasibility-check/national-bridge-crosswalk/bridge_candidate_summary.csv.gz',
        'traffic': 'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet',
        'supports': 'data/processed/medical-analysis-sample/medical_age_support_analysis_preprocessed.parquet',
        'meshes': 'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet',
    }
    hashes = {v: digest(ROOT / v) for v in rels.values()}
    b = pd.read_csv(ROOT / rels['bridges'], dtype=str, keep_default_na=False)
    assert len(b) == 730844 and b['shisetsu_id'].is_unique
    numeric = ['inspection_year', 'health_code', 'bridge_length_m', 'bridge_width_m',
               'nearest_distance_m', 'candidates_within_nearest_plus_5m',
               'near_candidate_min_level', 'near_candidate_max_level',
               'assets_sharing_nearest_feature', 'nearest_source_length_m']
    views = {c: pd.to_numeric(b[c], errors='coerce') for c in numeric}
    flags = b['review_flags']
    masks = {
        'Coordinate or candidate unavailable': b['coordinate_status'].ne('source_geographic_screen_passed') | b['candidate_status'].ne('candidate_present'),
        'Nearest distance above 50 m or undefined': views['nearest_distance_m'].isna() | views['nearest_distance_m'].gt(50),
        'Multiple close candidates': views['candidates_within_nearest_plus_5m'].ne(1),
        'Route class conflict': flags.str.contains('named_route_class_conflict', regex=False),
        'Mixed road levels': views['near_candidate_min_level'].ne(views['near_candidate_max_level']),
        'Nearest tunnel': flags.str.contains('nearest_source_is_tunnel', regex=False),
        'Old asset warning': flags.str.contains('old_name_indicator_requires_review', regex=False),
        'Source quality exception': b['source_quality_flags'].ne(''),
        'Shared nearest source road': views['assets_sharing_nearest_feature'].ne(1),
        'Unusable condition': ~views['health_code'].isin([1, 2, 3, 4]),
        'Unusable dimensions': views['bridge_length_m'].isna() | views['bridge_width_m'].isna() | views['bridge_length_m'].le(0) | views['bridge_width_m'].le(0),
        'Unusable inspection year': views['inspection_year'].isna() | views['inspection_year'].le(0) | views['inspection_year'].gt(2026),
    }
    t = pd.read_parquet(ROOT / rels['traffic'], columns=['Graph Source Edge ID', 'Candidate Road Source Record',
                 'Observed Traffic Volume', 'Traffic Observation Key', 'Baseline Source Road Length'])
    assert t['Candidate Road Source Record'].is_unique and t['Graph Source Edge ID'].is_unique
    masks['No accepted observed traffic'] = ~b['nearest_source_id'].isin(t['Candidate Road Source Record'])
    reject = np.column_stack([x.to_numpy() for x in masks.values()]).any(axis=1)
    b['Policy Common Sample Eligible'] = ~reject
    reasons = np.full(len(b), '', dtype=object)
    for name, mask in masks.items():
        ix = mask.to_numpy()
        reasons[ix] = np.where(reasons[ix] == '', name, reasons[ix] + ';' + name)
    b['Policy Exclusion Reasons'] = reasons
    rename = {'shisetsu_id': 'Bridge Record ID', 'source_prefecture': 'Source Prefecture',
              'nearest_source_id': 'Candidate Road Source Record', 'inspection_year': 'Bridge Inspection Year',
              'health_code': 'Bridge Inspection Category', 'bridge_length_m': 'Bridge Length',
              'bridge_width_m': 'Bridge Width', 'nearest_distance_m': 'Bridge Anchor Separation',
              'nearest_source_length_m': 'Source Road Length', 'nearest_road_class': 'Source Road Class',
              'review_flags': 'Bridge Candidate Review Flags', 'source_quality_flags': 'Bridge Source Quality Flags',
              'accepted_match': 'Physical Asset Match Accepted'}
    retained = b[list(rename) + ['Policy Common Sample Eligible', 'Policy Exclusion Reasons']].rename(columns=rename)
    # Preserve the complete source bridge inventory and original strings in a provenance table.
    retained.to_parquet(OUT / 'bridge_policy_inventory_construction.parquet', index=False)
    u = retained.loc[~reject].copy()
    for source in ['inspection_year', 'health_code', 'bridge_length_m', 'bridge_width_m', 'nearest_distance_m', 'nearest_source_length_m']:
        u[rename[source]] = views[source][~reject].to_numpy()
    u = u.drop(columns=['Source Road Length']).merge(t, on='Candidate Road Source Record', validate='one_to_one')
    assert len(u) > 0 and u['Candidate Road Source Record'].is_unique
    ids = u['Graph Source Edge ID'].to_numpy(dtype=np.int64)
    lost = np.load(ROOT / 'data/exp/national-road-service-loss/lost_medical_path_population.npy', mmap_mode='r')
    extra = np.load(ROOT / 'data/exp/national-road-service-loss/additional_medical_distance_burden.npy', mmap_mode='r')
    u['Lost Medical Path Population'] = lost[ids]
    u['Additional Medical Distance Burden'] = extra[ids]
    u['Relative Intervention Cost'] = u['Bridge Length'] * u['Bridge Width']
    u['Maintenance Intervention Effect'] = 1.0
    u['Conditional Event Weight'] = 1.0 / len(u)
    u = u.sort_values('Candidate Road Source Record').reset_index(drop=True)
    assert u['Relative Intervention Cost'].gt(0).all()
    assert u['Lost Medical Path Population'].sum() > 0 and u['Additional Medical Distance Burden'].sum() > 0, 'Common domain has no informative loss outcomes; return to scope review.'
    u.to_parquet(OUT / 'maintenance_units_construction.parquet', index=False)
    g = pd.read_parquet(ROOT / rels['supports'])
    age_cut = float(g['Population Aged 65+ Share Among Known Ages'].quantile(.75, interpolation='linear'))
    distance_cut = float(g['Population-Weighted Candidate Medical Network Distance'].quantile(.75, interpolation='linear'))
    g['Higher Ageing Area'] = g['Population Aged 65+ Share Among Known Ages'].ge(age_cut)
    g['Baseline Distance Disadvantaged Area'] = g['Population-Weighted Candidate Medical Network Distance'].ge(distance_cut)
    cols = ['Publication Support Code', 'Population Aged 65+ Share Among Known Ages', 'Population-Weighted Candidate Medical Network Distance',
            'Support Total Population', 'Higher Ageing Area', 'Baseline Distance Disadvantaged Area']
    g[cols].to_parquet(OUT / 'area_groups_construction.parquet', index=False)
    m = pd.read_parquet(ROOT / rels['meshes'])
    m = m.merge(g[cols], on='Publication Support Code', validate='many_to_one')
    assert len(m) == 175612 and int(m['Local Total Population'].sum()) == 125874788
    m.to_parquet(OUT / 'mesh_area_groups_construction.parquet', index=False)
    accounting = [{'Exclusion Reason': name, 'Bridge Records': int(mask.sum()), 'Overlapping': True} for name, mask in masks.items()]
    accounting += [{'Exclusion Reason': 'Excluded union', 'Bridge Records': int(reject.sum()), 'Overlapping': False},
                   {'Exclusion Reason': 'Retained common sample', 'Bridge Records': len(u), 'Overlapping': False}]
    pd.DataFrame(accounting).to_csv(OUT / 'sample_accounting.csv', index=False)
    v = {'status': 'passed', 'authorization': AUTH, 'bridge_rows_preserved': len(b), 'common_units': len(u),
         'common_prefectures': sorted(u['Source Prefecture'].unique().tolist()),
         'common_traffic_units': u['Traffic Observation Key'].nunique(),
         'condition_counts': {str(k): int(v) for k, v in u['Bridge Inspection Category'].value_counts().items()},
         'inspection_year_counts': {str(k): int(v) for k, v in u['Bridge Inspection Year'].value_counts().items()},
         'positive_lost_path_units': int(u['Lost Medical Path Population'].gt(0).sum()),
         'positive_distance_burden_units': int(u['Additional Medical Distance Burden'].gt(0).sum()),
         'source_hashes': hashes, 'baseline_meshes': len(m), 'baseline_population': int(m['Local Total Population'].sum()),
         'ageing_cutoff': age_cut, 'baseline_distance_cutoff_metres': distance_cut,
         'higher_ageing_meshes': int(m['Higher Ageing Area'].sum()),
         'higher_ageing_population': int(m.loc[m['Higher Ageing Area'], 'Local Total Population'].sum()),
         'distance_disadvantaged_meshes': int(m['Baseline Distance Disadvantaged Area'].sum()),
         'distance_disadvantaged_population': int(m.loc[m['Baseline Distance Disadvantaged Area'], 'Local Total Population'].sum()),
         'overlap_population': int(m.loc[m['Higher Ageing Area'] & m['Baseline Distance Disadvantaged Area'], 'Local Total Population'].sum()),
         'physical_matches_adopted': False, 'source_accepted_match_flags_changed': False,
         'all_source_hashes_unchanged': all(digest(ROOT / p) == h for p, h in hashes.items())}
    (OUT / 'construction_validation.json').write_text(json.dumps(v, indent=2) + '\n')
    print(json.dumps(v, indent=2), flush=True)


if __name__ == '__main__':
    main()
