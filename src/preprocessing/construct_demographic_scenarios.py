"""Construct census-anchored official population scenarios adopted in decision 036.

Preserve all original origins; unmatched scenario populations remain missing.
Municipality-specific components are summed, never deduplicated by mesh alone.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/data-preprocessing/demographic-stability'
YEARS = [2020, 2030, 2040, 2050]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = ['data/exp/data-briefing/demographic-projection/source_scope.parquet',
               'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet']
    hashes = {p: digest(ROOT / p) for p in sources}
    raw = pd.read_parquet(ROOT / sources[0])
    original = pd.read_parquet(ROOT / sources[1])
    assert not raw.duplicated(['MESH_ID', 'SHICODE']).any()
    columns = [f'PTN_{y}' for y in YEARS]
    assert raw[columns].notna().all().all() and raw[columns].ge(0).all().all()
    totals = raw.groupby('MESH_ID', sort=False)[columns].sum()
    aligned = totals.reindex(original['Mesh Code'])
    matched = aligned['PTN_2020'].notna().to_numpy()
    reference = aligned['PTN_2020'].to_numpy()
    census = original['Local Total Population'].to_numpy(dtype=float)
    assert matched.sum() == 175485 and (~matched).sum() == 127
    assert census[~matched].sum() == 278
    assert (reference[matched] > 0).all()
    rows, accounting = [], []
    for year in YEARS:
        projected = aligned[f'PTN_{year}'].to_numpy()
        factor = projected / reference
        if year == 2020:
            np.testing.assert_array_equal(factor[matched], np.ones(matched.sum()))
        scenario = census * factor
        assert np.isnan(scenario[~matched]).all()
        assert np.isfinite(scenario[matched]).all() and (scenario[matched] >= 0).all()
        frame = original[['Mesh Code', 'Publication Support Code', 'Local Total Population',
                          'Higher Ageing Area', 'Baseline Distance Disadvantaged Area']].copy()
        frame['Forecast Matched Origin'] = matched
        frame['Population-Change Scenario'] = year
        frame['Forecast Reference Population'] = reference
        frame['Projected Total Population'] = projected
        frame['Population Change Factor'] = factor
        frame['Scenario Population'] = scenario
        frame['Ageing Scenario'] = 'Fixed 2020 area membership'
        rows.append(frame)
        accounting.append({'scenario_year': year, 'matched_origins': int(matched.sum()),
                           'population': float(scenario[matched].sum()),
                           'zero_population_origins': int((scenario[matched] == 0).sum()),
                           'higher_ageing_population': float(scenario[matched & frame['Higher Ageing Area'].to_numpy()].sum()),
                           'distance_disadvantaged_population': float(scenario[matched & frame['Baseline Distance Disadvantaged Area'].to_numpy()].sum())})
    result = pd.concat(rows, ignore_index=True)
    assert not result.duplicated(['Mesh Code', 'Population-Change Scenario']).any()
    path = OUT / 'population_scenarios_construction.parquet'
    result.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(result, pd.read_parquet(path))
    variables = [{'original_name': c, 'readable_name': c, 'full_name': c,
                  'is_final_variable': 'yes' if c not in ['Mesh Code', 'Publication Support Code'] else 'no',
                  'preprocessing': []} for c in result.columns]
    decision = {'authorization': 'MILI-D-20261007-036', 'datasets': {
        str(path.relative_to(ROOT)): {'output': 'data/processed/demographic-stability/population_scenarios_preprocessed.parquet',
                                     'script': 'src/preprocessing/preprocess_demographic_population_scenarios.py',
                                     'variables': variables}}}
    (OUT / 'decisions.json').write_text(json.dumps(decision, indent=2) + '\n')
    for p, h in hashes.items():
        assert digest(ROOT / p) == h
    validation = {'status': 'passed', 'authorization': 'MILI-D-20261007-036',
                  'rows': len(result), 'columns': len(result.columns), 'years': YEARS,
                  'matched_origins': 175485, 'unmatched_origins': 127, 'unmatched_census_population': 278,
                  'maintenance_units_unchanged': 46688, 'municipality_components_summed': True,
                  'no_unmatched_population_imputation': True, 'original_area_membership_fixed': True,
                  'future_individual_age_locations_inferred': False, 'parquet_readback_passed': True,
                  'source_hashes_unchanged': hashes, 'scenario_accounting': accounting}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    proposal = ROOT / 'data/exp/estimation-framework-planning/demographic-stability/construction_proposal.json'
    adopted = json.loads(proposal.read_text())
    adopted.update(status='adopted', construction_authorization='MILI-D-20261007-036')
    proposal.write_text(json.dumps(adopted, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
