"""Independent read-back of the adopted national scenario fields and seeds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]


def main():
    exp=ROOT/'data/exp/data-preprocessing/national-pavement-inputs'
    assert json.loads((exp/'readable_output_validation.json').read_text())['status']=='passed'
    proposal=ROOT/'data/exp/pavement-scenario-domain-review'
    medians=np.full(128,np.nan); probabilities=np.zeros((128,3))
    for row in pd.read_csv(proposal/'stratified_assignment_proposal.csv').to_dict('records'):
        code=int(row['Road Classification Code'])*16+int(row['Road Width Code'])
        medians[code]=row['Proposed Missing Traffic Volume']
        probabilities[code]=[row[f'Proposed Condition {g} Probability'] for g in ['I','II','III']]
    lengths=np.load(ROOT/'data/processed/national-network-rule-validation/edge_lengths.npy',mmap_mode='r')
    observed=pd.read_parquet(ROOT/'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet',
        columns=['Graph Source Edge ID','Observed Traffic Volume']).set_index('Graph Source Edge ID')
    markers=pd.read_parquet(ROOT/'data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet',
        columns=['Graph Source Edge ID','Anchored Pavement Inspection Category']).set_index('Graph Source Edge ID')
    rng=np.random.default_rng(20261007);ties=np.random.default_rng(20261008)
    rows=tc=cc=0; cost=0.;hist=np.zeros(3,dtype=np.int64)
    source=ROOT/'data/processed/national-pavement-inputs/national_pavement_inputs_preprocessed.parquet'
    pf=pq.ParquetFile(source)
    assert pf.schema_arrow.names==pd.read_csv(proposal/'readable_input_proposal.csv')['Proposed Readable Name'].tolist()
    for batch in pf.iter_batches(batch_size=250000):
        df=batch.to_pandas(); ids=df['Graph Source Edge ID'].to_numpy()
        np.testing.assert_array_equal(ids,np.arange(rows,rows+len(df)))
        np.testing.assert_array_equal(df['Baseline Source Road Length'],lengths[ids])
        np.testing.assert_array_equal(df['Scenario Relative Maintenance Cost'],lengths[ids])
        assert df['Scenario Deterioration Eligible'].all() and df['Scenario Maintenance Eligible'].all()
        tm=df['Observed Traffic Volume'].notna().to_numpy();cm=df['Anchored Pavement Inspection Category'].notna().to_numpy()
        np.testing.assert_array_equal(df.loc[tm,'Observed Traffic Volume'],observed.loc[ids[tm],'Observed Traffic Volume'])
        np.testing.assert_array_equal(df.loc[cm,'Anchored Pavement Inspection Category'],markers.loc[ids[cm],'Anchored Pavement Inspection Category'])
        np.testing.assert_array_equal(df.loc[tm,'Scenario Traffic Volume'],df.loc[tm,'Observed Traffic Volume'])
        np.testing.assert_array_equal(df.loc[cm,'Scenario Pavement Need Category'],df.loc[cm,'Anchored Pavement Inspection Category'])
        codes=df['Road Classification Code'].to_numpy(dtype=np.int16)*16+df['Road Width Code'].to_numpy(dtype=np.int16)
        np.testing.assert_array_equal(df.loc[~tm,'Scenario Traffic Volume'],medians[codes[~tm]])
        cdf=np.cumsum(probabilities[codes[~cm]],axis=1)
        expected=1+(rng.random(int((~cm).sum()))[:,None]>cdf[:,:2]).sum(axis=1)
        np.testing.assert_array_equal(df.loc[~cm,'Scenario Pavement Need Category'],expected)
        np.testing.assert_array_equal(df['Scenario Priority Tie Key'],ties.random(len(df)))
        np.testing.assert_array_equal(df['Scenario Traffic Basis'].eq('observed-correspondence'),tm)
        np.testing.assert_array_equal(df['Scenario Condition Basis'].eq('observed-anchored-marker'),cm)
        assert df['Scenario Traffic Volume'].ge(0).all() and df['Scenario Pavement Need Category'].isin([1,2,3]).all()
        assert df.iloc[:,6:].notna().all().all()
        hist+=np.bincount(df['Scenario Pavement Need Category'],minlength=4)[1:]
        rows+=len(df);tc+=int(tm.sum());cc+=int(cm.sum());cost+=df['Scenario Relative Maintenance Cost'].sum()
    assert rows==21415800 and tc==len(observed)==1183367 and cc==len(markers)==70049
    np.testing.assert_allclose(cost,float(lengths.sum()),rtol=1e-14)
    result={'status':'passed_confirmed_national_input_construction','authorization':'MILI-D-20261008-001',
        'candidate_road_records':rows,'readable_fields':16,'observed_traffic_records':tc,'observed_condition_records':cc,
        'missing_observed_traffic_records':rows-tc,'missing_observed_condition_records':rows-cc,
        'condition_category_counts':hist.tolist(),'full_value_and_seed_readback_passed':True,
        'relative_cost_sum_metres':cost,'observed_inputs_preserved':True,'original_domain_not_modified':True}
    (exp/'construction_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
