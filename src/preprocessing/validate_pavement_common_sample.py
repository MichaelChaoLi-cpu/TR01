"""Independent read-back and inspection-to-network validation of pavement input."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'data/exp/data-preprocessing/pavement-inputs'
OUT = ROOT / 'data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet'
AUDIT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/national_road_correspondence'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    decisions = json.loads((EXP / 'decisions.json').read_text())
    source_name, decision = next(iter(decisions['datasets'].items()))
    original = pd.read_parquet(ROOT / source_name)
    output = pd.read_parquet(OUT)
    expected_names = [v['readable_name'] for v in decision['variables']] + ['Pavement Relative Maintenance Cost']
    assert list(output) == expected_names
    assert len(output) == 70049 and output['Graph Source Edge ID'].is_unique
    assert all(n.isascii() and n.strip() == n for n in output.columns)
    assert not output.isna().any().any()
    for v in decision['variables']:
        pd.testing.assert_series_equal(original[v['original_name']],output[v['readable_name']],check_names=False)
    pd.testing.assert_series_equal(output['Pavement Relative Maintenance Cost'],
                                  output['Baseline Source Road Length'],check_names=False)
    points = pd.read_parquet(AUDIT / 'point_candidate_diagnostics.parquet')
    eligible = points['proposed_screening_reason'].eq('proposed_conditional_anchor')
    valid = points.loc[eligible].copy()
    # Reconstruct eligibility from the saved full diagnostics, without using
    # the pre-aggregated grade range table as the grade oracle.
    reconstructed = (points['tenken.kiroku.hantei_kubun'].isin(['1','2','3']) &
        points['parsed_national_route_number'].gt(0) & points['nearest_distance_m'].le(50) &
        points['candidate_count_within_nearest_plus_5m'].eq(1) & points['nearest_route_compatible'])
    pd.testing.assert_series_equal(eligible,reconstructed,check_names=False)
    valid['grade'] = valid['tenken.kiroku.hantei_kubun'].astype(int)
    grouped = valid.groupby('nearest_graph_source_edge_id').agg(
        maximum=('grade','max'), minimum=('grade','min'), records=('shisetsu_id','count'),
        prefectures=('source_prefecture','nunique'))
    np.testing.assert_array_equal(grouped.index,output['Graph Source Edge ID'])
    for field,col in [('maximum','Anchored Pavement Inspection Category'),
                      ('minimum','Minimum Anchored Pavement Inspection Category'),
                      ('records','Anchored Pavement Inspection Record Count'),
                      ('prefectures','Anchored Inspection Prefecture Count')]:
        np.testing.assert_array_equal(grouped[field],output[col])
    np.testing.assert_array_equal(grouped['maximum'].ne(grouped['minimum']),
                                 output['Mixed Anchored Pavement Inspection Grades'])
    lengths_file = ROOT / 'data/processed/national-network-rule-validation/edge_lengths.npy'
    network_lengths = np.load(lengths_file,mmap_mode='r')
    ids = output['Graph Source Edge ID'].to_numpy(dtype=np.int64)
    difference = float(np.max(np.abs(network_lengths[ids]-output['Baseline Source Road Length'].to_numpy())))
    np.testing.assert_allclose(network_lengths[ids],output['Baseline Source Road Length'],rtol=0,atol=1e-9)
    assert output['Pavement Relative Maintenance Cost'].gt(0).all()
    assert output['Observed Traffic Volume'].gt(0).all()
    assert len(valid) == 264099 and valid['source_prefecture'].nunique() == 47
    assert output['Traffic Observation Key'].nunique() == 3368
    total = float(output['Pavement Relative Maintenance Cost'].sum())
    budget = pd.DataFrame({'Requested Pavement Budget Share':[.01,.05,.10,.20]})
    budget['Requested Relative Pavement Budget'] = budget.iloc[:,0]*total
    budget.to_csv(EXP/'reference_budget_accounting.csv',index=False)
    counts = output['Anchored Pavement Inspection Category'].value_counts().sort_index().to_dict()
    result = {'status':'passed_pavement_common_input_construction',
        'authorizations':['MILI-D-20261007-043','MILI-D-20261007-044'],
        'road_units':len(output),'columns':len(output.columns),'retained_source_fields':13,
        'derived_cost_fields':1,'inspection_records':len(valid),'prefectures':47,
        'shared_traffic_observation_units':3368,'grade_unit_counts':{str(k):int(v) for k,v in counts.items()},
        'mixed_grade_units':int(output['Mixed Anchored Pavement Inspection Grades'].sum()),
        'multi_prefecture_record_associations':int(output['Anchored Inspection Prefecture Count'].gt(1).sum()),
        'total_relative_length_cost':total,'maximum_network_length_difference_m':difference,
        'source_field_values_and_dtypes_preserved':True,'independent_anchor_and_grade_reconstruction':True,
        'unique_cost_accounting':True,'readable_ascii_names':True,'missing_values_imputed':False,
        'new_sample_exclusions':0,'routing_or_policy_results_generated':False,
        'input_sha256':{source_name:digest(ROOT/source_name),str((AUDIT/'point_candidate_diagnostics.parquet').relative_to(ROOT)):digest(AUDIT/'point_candidate_diagnostics.parquet')},
        'output_sha256':digest(OUT)}
    assert result['input_sha256'][source_name] == json.loads((EXP/'proposal_validation.json').read_text())['input_sha256']
    (EXP/'construction_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
