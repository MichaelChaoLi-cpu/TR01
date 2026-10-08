"""Construct the confirmed, explicitly hypothetical national pavement inputs.

Observed columns stay nullable outside accepted correspondences. Scenario
values and their provenance are separate. Stream by original source identity.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'data/exp/data-preprocessing/national-pavement-inputs'
PROPOSAL = ROOT/'data/exp/pavement-scenario-domain-review'
AUTH = 'MILI-D-20261008-001'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    start_time = time.perf_counter()
    BASE.mkdir(parents=True, exist_ok=True)
    spec = json.loads((PROPOSAL/'input_construction_proposal.json').read_text())
    assert spec['specific_assignments_authorized'] and spec['construction_authorization'] == AUTH
    lookup = pd.read_csv(PROPOSAL/'stratified_assignment_proposal.csv')
    names = pd.read_csv(PROPOSAL/'readable_input_proposal.csv')['Proposed Readable Name'].tolist()
    lengths_file = ROOT/'data/processed/national-network-rule-validation/edge_lengths.npy'
    traffic_file = ROOT/'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet'
    condition_file = ROOT/'data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet'
    source_files = [lengths_file, traffic_file, condition_file,
                    PROPOSAL/'stratified_assignment_proposal.csv', PROPOSAL/'readable_input_proposal.csv',
                    ROOT/'data/processed/national-network-quality/tile_offsets.npy']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    lengths = np.load(lengths_file, mmap_mode='r')
    n = len(lengths)
    assert n == 21415800
    classes = np.zeros(n, dtype=np.uint8)
    widths = np.zeros(n, dtype=np.uint8)
    for tile, offset, count in np.load(source_files[-1]):
        with np.load(ROOT/f'data/processed/national-network-quality/tiles/{tile:04}.npz') as z:
            np.testing.assert_array_equal(z['lengths'], lengths[offset:offset+count])
            classes[offset:offset+count] = z['classes']
            widths[offset:offset+count] = z['widths']
    assert (classes > 0).all() and (widths > 0).all()
    codes = classes.astype(np.uint16)*16+widths
    medians = np.full(128, np.nan)
    probabilities = np.zeros((128, 3))
    traffic_labels = np.full(128, '', dtype=object)
    condition_labels = np.full(128, '', dtype=object)
    traffic_basis = np.full(128, '', dtype=object)
    for row in lookup.to_dict('records'):
        code = int(row['Road Classification Code'])*16+int(row['Road Width Code'])
        medians[code] = row['Proposed Missing Traffic Volume']
        probabilities[code] = [row[f'Proposed Condition {grade} Probability'] for grade in ['I','II','III']]
        traffic_labels[code] = row['Traffic Calibration Reference']
        condition_labels[code] = row['Condition Calibration Reference']
        traffic_basis[code] = row['Missing Traffic Basis']
        assert np.count_nonzero(codes==code) == row['Candidate Road Count']
    assert np.isfinite(medians[codes]).all()
    traffic = pd.read_parquet(traffic_file, columns=['Graph Source Edge ID','Observed Traffic Volume'])
    condition = pd.read_parquet(condition_file, columns=['Graph Source Edge ID','Anchored Pavement Inspection Category'])
    assert traffic['Graph Source Edge ID'].is_unique and condition['Graph Source Edge ID'].is_unique
    observed = np.full(n, np.nan)
    observed[traffic['Graph Source Edge ID']] = traffic['Observed Traffic Volume']
    markers = np.zeros(n, dtype=np.uint8)
    markers[condition['Graph Source Edge ID']] = condition['Anchored Pavement Inspection Category']
    conditional_rng = np.random.default_rng(20261007)
    tie_rng = np.random.default_rng(20261008)
    cats = {
        'Scenario Traffic Basis': ['observed-correspondence']+sorted(set(traffic_basis[codes])),
        'Scenario Condition Basis': ['observed-anchored-marker','assumed-categorical-marker'],
        'Scenario Traffic Calibration Stratum': ['observed-correspondence']+sorted(set(traffic_labels[codes])),
        'Scenario Condition Calibration Stratum': ['observed-anchored-marker']+sorted(set(condition_labels[codes])),
    }
    target = BASE/'national_pavement_inputs.parquet'
    temporary = target.with_suffix('.part')
    writer = None
    try:
        for start in range(0,n,250000):
            stop = min(start+250000,n); code = codes[start:stop]
            obs = observed[start:stop]; grade = markers[start:stop]
            tmissing = ~np.isfinite(obs); cmissing = grade==0
            scenario_grade = grade.copy()
            draw = conditional_rng.random(int(cmissing.sum()))
            cdf = np.cumsum(probabilities[code[cmissing]], axis=1)
            scenario_grade[cmissing] = 1+(draw[:,None]>cdf[:,:2]).sum(axis=1)
            raw_grade = pd.array(grade, dtype='Int8')
            raw_grade[cmissing] = pd.NA
            values = {
                'Graph Source Edge ID': np.arange(start,stop,dtype=np.int64),
                'Baseline Source Road Length': np.asarray(lengths[start:stop]),
                'Road Classification Code': classes[start:stop], 'Road Width Code': widths[start:stop],
                'Observed Traffic Volume': obs, 'Anchored Pavement Inspection Category': raw_grade,
                'Scenario Traffic Volume': np.where(tmissing,medians[code],obs),
                'Scenario Pavement Need Category': scenario_grade,
                'Scenario Traffic Basis': np.where(tmissing,traffic_basis[code],'observed-correspondence'),
                'Scenario Condition Basis': np.where(cmissing,'assumed-categorical-marker','observed-anchored-marker'),
                'Scenario Traffic Calibration Stratum': np.where(tmissing,traffic_labels[code],'observed-correspondence'),
                'Scenario Condition Calibration Stratum': np.where(cmissing,condition_labels[code],'observed-anchored-marker'),
                'Scenario Deterioration Eligible': np.ones(stop-start,dtype=bool),
                'Scenario Maintenance Eligible': np.ones(stop-start,dtype=bool),
                'Scenario Relative Maintenance Cost': np.asarray(lengths[start:stop]),
                'Scenario Priority Tie Key': tie_rng.random(stop-start),
            }
            for field,categories in cats.items():
                values[field] = pd.Categorical(values[field], categories=categories)
            frame = pd.DataFrame(values)[names]
            table = pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary,table.schema,compression='zstd')
            writer.write_table(table)
            if start%2500000==0 or stop==n:
                print(f'National input records {stop:,}/{n:,}',flush=True)
    finally:
        if writer is not None: writer.close()
    temporary.replace(target)
    assert pq.ParquetFile(target).metadata.num_rows == n
    for p in source_files: assert digest(p)==hashes[str(p.relative_to(ROOT))]
    report = {'status':'constructed_confirmed_national_input_source','authorization':AUTH,
              'rows':n,'fields':16,'observed_traffic_roads':len(traffic),'observed_condition_roads':len(condition),
              'missing_observed_traffic_remains_missing':True,'missing_observed_condition_remains_missing':True,
              'hypothetical_scenario_inputs_separate':True,'condition_draw_seed':20261007,'tie_seed':20261008,
              'relative_cost_sum_metres':float(lengths.sum()),'source_hashes':hashes,
              'elapsed_seconds':time.perf_counter()-start_time,'final_readable_output_validation_pending':True}
    (BASE/'source_construction_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
