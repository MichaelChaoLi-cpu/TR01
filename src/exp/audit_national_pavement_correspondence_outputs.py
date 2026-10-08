"""Independently reconcile saved nationwide candidate partitions and proposal."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/national_road_correspondence'


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    diag = pd.read_parquet(OUT / 'point_candidate_diagnostics.parquet')
    units = pd.read_parquet(OUT / 'proposed_road_grade_aggregation.parquet')
    original = pd.read_csv(ROOT / 'data/exp/feasibility-check/national-pavement-inventory/pavement_basic_index.csv.gz', dtype=str, keep_default_na=False)
    assert diag[original.columns].equals(original)
    count = np.zeros(len(diag), dtype=int)
    near_count = np.zeros(len(diag), dtype=int)
    nearest = np.full(len(diag), np.inf)
    hashes = {}
    files = sorted((OUT / 'candidate_pairs').glob('pref_*.parquet'))
    assert len(files) == 47
    for path in files:
        p = pd.read_parquet(path)
        row = p['pavement_source_row'].to_numpy()
        distance = p['distance_m'].to_numpy()
        assert not p[['pavement_source_row','graph_source_edge_id']].duplicated().any()
        assert ((distance >= 0) & (distance <= 200)).all()
        assert p['point_position_on_curve_m'].ge(-1e-8).all()
        assert p['point_position_on_curve_m'].le(p['projected_curve_length_m']+1e-8).all()
        np.testing.assert_array_equal(p['pavement_record_id'], diag['shisetsu_id'].to_numpy()[row])
        assert (diag['source_prefecture'].to_numpy()[row] == path.stem[-2:]).all()
        np.add.at(count,row,1)
        np.minimum.at(nearest,row,distance)
        near = distance <= diag['nearest_distance_m'].to_numpy()[row]+5.
        np.add.at(near_count,row[near],1)
        with path.open('rb') as stream:
            hashes[str(path.relative_to(ROOT))] = hashlib.file_digest(stream,'sha256').hexdigest()
    np.testing.assert_array_equal(count,diag['candidate_count_within_200m'])
    np.testing.assert_array_equal(near_count,diag['candidate_count_within_nearest_plus_5m'])
    np.testing.assert_allclose(nearest,np.nan_to_num(diag['nearest_distance_m'].to_numpy(),nan=np.inf),rtol=0,atol=0)
    proposal = diag.loc[diag.proposed_screening_reason.eq('proposed_conditional_anchor')]
    grouped = proposal.groupby('nearest_graph_source_edge_id').size()
    np.testing.assert_array_equal(grouped.index,units['nearest_graph_source_edge_id'])
    np.testing.assert_array_equal(grouped.to_numpy(),units['anchored_inspection_records'])
    assert units['Graph Source Edge ID'].is_unique
    assert sum(count) == summary['candidate_pairs']
    assert len(proposal) == summary['proposed_anchored_records']
    assert len(units) == summary['proposed_unique_road_units']
    assert units['Traffic Observation Key'].notna().all()
    assert units['Observed Traffic Volume'].gt(0).all()
    traffic = pd.read_parquet(ROOT / 'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet',
        columns=['Traffic Observation Key','Traffic Road Class','Traffic Route Number'])
    route_counts = traffic.loc[traffic['Traffic Road Class'].astype(int).eq(3)].groupby(
        'Traffic Observation Key')['Traffic Route Number'].nunique()
    assert units['Traffic Observation Key'].map(route_counts).eq(1).all()
    counts = units.maximum_observed_grade.value_counts().sort_index().to_dict()
    grade_table = pd.DataFrame({'observed_grade':[1,2,3],
        'units_using_lowest_observed_grade':[int(units.minimum_observed_grade.eq(g).sum()) for g in [1,2,3]],
        'units_using_worst_observed_grade':[int(units.maximum_observed_grade.eq(g).sum()) for g in [1,2,3]]})
    grade_table.to_csv(OUT / 'grade_aggregation_review.csv',index=False)
    pd.Series(summary['exclusive_proposed_screening_counts'],name='record_count').rename_axis('reason').to_csv(OUT / 'exclusive_screening_counts.csv')
    result = {'status':'passed_independent_saved_output_audit',
        'all_24_source_fields_and_rows_preserved_exactly':True,'prefecture_partitions':len(files),
        'saved_pair_rows_reconciled':int(count.sum()),'candidate_nearest_distances_reproduced':True,
        'pair_ids_unique_and_partial_positions_valid':True,
        'proposal_records':len(proposal),'proposal_road_units':len(units),
        'shared_measured_traffic_observation_units':units['Traffic Observation Key'].nunique(),
        'proposed_observation_units_have_single_national_route_identity':True,
        'proposed_worst_observed_grade_unit_counts':{str(k):int(v) for k,v in counts.items()},
        'accepted_policy_assignments':0,'candidate_partition_sha256':hashes}
    (OUT / 'independent_output_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='candidate_partition_sha256'},indent=2))


if __name__ == '__main__':
    main()
