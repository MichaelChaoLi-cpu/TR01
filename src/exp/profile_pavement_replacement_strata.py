"""Profile existing national road strata before hypothetical input construction.

No missing traffic or condition values are assigned. Cached source classification
and width codes are aligned to the unchanged global source-road identity.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/pavement-scenario-domain-review/strata'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lengths = np.load(ROOT / 'data/processed/national-network-rule-validation/edge_lengths.npy', mmap_mode='r')
    offsets = np.load(ROOT / 'data/processed/national-network-quality/tile_offsets.npy')
    classes = np.zeros(len(lengths), dtype=np.uint8)
    widths = np.zeros(len(lengths), dtype=np.uint8)
    for tile, start, count in offsets:
        with np.load(ROOT / f'data/processed/national-network-quality/tiles/{tile:04}.npz') as z:
            np.testing.assert_array_equal(z['lengths'], lengths[start:start+count])
            classes[start:start+count] = z['classes']
            widths[start:start+count] = z['widths']
    assert (classes > 0).all() and (widths > 0).all()
    assert widths.max() < 16
    strata = classes.astype(np.uint16)*16 + widths
    count = np.bincount(strata)
    length = np.bincount(strata, weights=lengths)
    traffic = pd.read_parquet(ROOT / 'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet',
                             columns=['Graph Source Edge ID', 'Traffic Observation Key', 'Observed Traffic Volume'])
    edges = traffic['Graph Source Edge ID'].to_numpy()
    assert traffic['Graph Source Edge ID'].is_unique
    assert np.isfinite(traffic['Observed Traffic Volume']).all() and traffic['Observed Traffic Volume'].ge(0).all()
    traffic['Road Classification Code'] = classes[edges]
    traffic['Road Width Code'] = widths[edges]
    traffic['Stratum'] = strata[edges]
    # The same traffic count mapped to many source-road records is one count
    # within a stratum, not many independent traffic observations.
    assert traffic.groupby('Traffic Observation Key')['Observed Traffic Volume'].nunique().max() == 1
    units = traffic.drop_duplicates(['Stratum', 'Traffic Observation Key'])
    condition = pd.read_parquet(ROOT / 'data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet',
                               columns=['Graph Source Edge ID', 'Anchored Pavement Inspection Category'])
    edges = condition['Graph Source Edge ID'].to_numpy()
    condition['Road Classification Code'] = classes[edges]
    condition['Road Width Code'] = widths[edges]
    condition['Stratum'] = strata[edges]
    rows = []
    for code in np.flatnonzero(count):
        t = traffic[traffic['Stratum'].eq(code)]
        u = units[units['Stratum'].eq(code)]
        c = condition[condition['Stratum'].eq(code)]
        rows.append({
            'Road Classification Code': int(code//16), 'Road Width Code': int(code%16),
            'Candidate Road Count': int(count[code]), 'Candidate Road Length': float(length[code]),
            'Traffic Matched Road Count': len(t), 'Distinct Traffic Observation Count': len(u),
            'Traffic Median': float(u['Observed Traffic Volume'].median()) if len(u) else None,
            'Condition Matched Road Count': len(c),
            'Condition I Count': int(c['Anchored Pavement Inspection Category'].eq(1).sum()),
            'Condition II Count': int(c['Anchored Pavement Inspection Category'].eq(2).sum()),
            'Condition III Count': int(c['Anchored Pavement Inspection Category'].eq(3).sum()),
            'Direct Traffic Stratum Available': bool(len(u)),
            'Direct Condition Stratum Available': bool(len(c)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'road_class_width_coverage.csv', index=False)
    units[['Road Classification Code','Road Width Code','Traffic Observation Key','Observed Traffic Volume']].to_csv(OUT / 'traffic_calibration_observations.csv', index=False)
    condition[['Road Classification Code','Road Width Code','Graph Source Edge ID','Anchored Pavement Inspection Category']].to_csv(OUT / 'condition_calibration_markers.csv', index=False)
    traffic.drop_duplicates(['Road Classification Code','Traffic Observation Key']).groupby('Road Classification Code')['Observed Traffic Volume'].agg(['count','median']).to_csv(OUT / 'traffic_class_fallback_candidates.csv')
    condition.groupby(['Road Classification Code','Anchored Pavement Inspection Category']).size().unstack(fill_value=0).to_csv(OUT / 'condition_class_fallback_candidates.csv')
    result = {
        'status': 'passed_stratum_coverage_profile', 'road_records': len(lengths),
        'strata': len(table), 'source_codes_preserved': True,
        'traffic_road_records': len(traffic), 'condition_road_records': len(condition),
        'traffic_strata': int(table['Direct Traffic Stratum Available'].sum()),
        'condition_strata': int(table['Direct Condition Stratum Available'].sum()),
        'roads_in_strata_without_traffic': int(table.loc[~table['Direct Traffic Stratum Available'],'Candidate Road Count'].sum()),
        'roads_in_strata_without_condition': int(table.loc[~table['Direct Condition Stratum Available'],'Candidate Road Count'].sum()),
        'traffic_observation_multiplicity_removed_for_stratum_medians': True,
        'hypothetical_inputs_constructed': False, 'source_and_processed_files_modified': False,
        'road_class_label_lookup': 'Use original source codes; do not infer labels from numeric order.',
    }
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
