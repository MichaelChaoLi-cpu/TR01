"""Independent ranking, indivisible-prefix and group/mesh outcome accounting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/national-pavement-policy-computation'
EXP=ROOT/'data/exp/data-preprocessing/national-pavement-policy'
DATA=ROOT/'data/processed/national-pavement-policy'
TOL=1e-7


def main():
    receipt=json.loads((OUT/'validation.json').read_text());assert receipt['status']=='passed'
    readable=json.loads((EXP/'readable_output_validation.json').read_text());assert readable['status']=='passed'
    cost=np.load(ROOT/'data/processed/national-network-rule-validation/edge_lengths.npy',mmap_mode='r');total=float(cost.sum())
    inputs=pd.read_parquet(ROOT/'data/processed/national-pavement-inputs/national_pavement_inputs_preprocessed.parquet',
        columns=['Scenario Traffic Volume','Scenario Pavement Need Category','Scenario Priority Tie Key'])
    gains=np.load(ROOT/'data/exp/national-pavement-access-scoring/isolated_restoration_gain.npy',mmap_mode='r')
    groups=pd.read_parquet(DATA/'national_pavement_group_outcomes_preprocessed.parquet')
    assert len(groups)==117 and not groups.duplicated(['Pavement Maintenance Priority','Pavement Budget Share',
        'Pavement Impedance Increment','Pavement Population Group']).any()
    max_resource_error=0.;selected_rows=0
    for rule in ['Traffic','Condition','Access']:
        order=np.load(OUT/f'{rule.lower()}_order.npy',mmap_mode='r')
        primary=inputs['Scenario Traffic Volume'].to_numpy()[order] if rule=='Traffic' else (
            inputs['Scenario Pavement Need Category'].to_numpy()[order] if rule=='Condition' else gains[order]/cost[order])
        assert (primary[:-1]>=primary[1:]).all()
        ties=inputs['Scenario Priority Tie Key'].to_numpy()[order]
        equal=primary[:-1]==primary[1:];assert (ties[:-1][equal]<=ties[1:][equal]).all()
        exact=equal&(ties[:-1]==ties[1:]);assert (order[:-1][exact]<order[1:][exact]).all()
        ranking=pq.ParquetFile(DATA/'national_pavement_rankings_preprocessed.parquet')
        count=0
        for batch in ranking.iter_batches(batch_size=250000,columns=['Graph Source Edge ID','Pavement Maintenance Priority','Pavement Priority Rank']):
            df=batch.to_pandas();part=df[df['Pavement Maintenance Priority'].eq(rule)]
            if len(part):
                np.testing.assert_array_equal(part['Graph Source Edge ID'],order[count:count+len(part)])
                np.testing.assert_array_equal(part['Pavement Priority Rank'],np.arange(count+1,count+1+len(part)))
                count+=len(part)
        assert count==21415800
        cumulative=np.cumsum(cost[order],dtype=np.float64)
        previous=0
        for share in [.01,.05,.10,.20]:
            chosen=np.load(OUT/'cases'/f'{rule.lower()}_{share:.2f}_selected.npy',mmap_mode='r')
            expected_count=int(np.searchsorted(cumulative,share*total,side='right'))
            assert len(chosen)==expected_count and len(chosen)>=previous;previous=len(chosen)
            np.testing.assert_array_equal(chosen,order[:len(chosen)])
            stored=pd.read_parquet(DATA/'national_pavement_selections_preprocessed.parquet',
                filters=[('Pavement Maintenance Priority','=',rule),('Pavement Budget Share','=',share)])
            assert stored['Pavement Maintenance Selected'].all() and stored['Graph Source Edge ID'].is_unique
            np.testing.assert_array_equal(stored['Graph Source Edge ID'],chosen);selected_rows+=len(chosen)
            spent=float(cost[chosen].sum())
            for delta in [.10,.25,.50]:
                part=groups[groups['Pavement Maintenance Priority'].eq(rule)&groups['Pavement Budget Share'].eq(share)&
                    groups['Pavement Impedance Increment'].eq(delta)]
                assert len(part)==3 and (part['Pavement Selected Road Count']==len(chosen)).all()
                np.testing.assert_allclose(part['Pavement Actual Relative Spending'],spent,rtol=1e-14,atol=1e-6)
                error=(part['Pavement Actual Relative Spending']+part['Pavement Unused Relative Budget']-share*total).abs().max()
                max_resource_error=max(max_resource_error,float(error));assert error<1e-6
        del primary,ties,cumulative
    assert selected_rows==pq.ParquetFile(DATA/'national_pavement_selections_preprocessed.parquet').metadata.num_rows
    del inputs
    sample=pd.read_parquet(ROOT/'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet')
    supports=pd.read_parquet(ROOT/'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet').set_index('Mesh Code').loc[sample['Mesh Code']]
    masks={'All Residents':np.ones(len(sample),dtype=bool),'Higher Ageing Area':supports['Higher Ageing Area'].to_numpy(),
        'Baseline Distance Disadvantaged Area':supports['Baseline Distance Disadvantaged Area'].to_numpy()}
    pop=sample['Local Total Population'].to_numpy();baseline=sample['Candidate Medical Network Distance'].to_numpy()
    errors=[];cases=0;zero_counts={}
    for delta in [.10,.25,.50]:
        for rule in ['Reference','Traffic','Condition','Access']:
            previous=None
            for share in ([0.] if rule=='Reference' else [.01,.05,.10,.20]):
                df=pd.read_parquet(DATA/'national_pavement_mesh_scenarios_preprocessed.parquet',filters=[
                    ('Pavement Impedance Increment','=',delta),('Pavement Maintenance Priority','=',rule),('Pavement Budget Share','=',share)])
                assert len(df)==175612 and df['Mesh Code'].is_unique
                np.testing.assert_array_equal(df['Mesh Code'],sample['Mesh Code'])
                np.testing.assert_array_equal(df['Local Total Population'],pop)
                np.testing.assert_array_equal(df['Candidate Medical Network Distance'],baseline)
                values=df['Pavement Medical Access Cost'].to_numpy();increase=df['Pavement Medical Access Cost Increase'].to_numpy()
                assert np.isfinite(values).all() and np.isfinite(increase).all() and (increase>=0).all()
                raw=values-baseline;raw[np.abs(raw)<=TOL]=0
                np.testing.assert_array_equal(increase,raw)
                if rule=='Reference':np.testing.assert_allclose(values,baseline*(1+delta),rtol=1e-12,atol=TOL)
                if previous is not None:assert (values<=previous+TOL).all()
                previous=values
                for group,mask in masks.items():
                    record=groups[groups['Pavement Impedance Increment'].eq(delta)&groups['Pavement Maintenance Priority'].eq(rule)&
                        groups['Pavement Budget Share'].eq(share)&groups['Pavement Population Group'].eq(group)].iloc[0]
                    residual=float(np.sum(pop[mask]*increase[mask])/1000)
                    reference=float(np.sum(pop[mask]*baseline[mask]*delta)/1000)
                    error=abs(residual-record['Pavement Residual Access Burden']);errors.append(error)
                    np.testing.assert_allclose(record['Pavement Reference Access Burden'],reference,rtol=1e-12,atol=1e-6)
                    np.testing.assert_allclose(record['Pavement Residual Access Burden'],residual,rtol=1e-12,atol=1e-7)
                    avoided=record['Pavement Reference Access Burden']-residual
                    np.testing.assert_allclose(record['Pavement Avoided Access Burden'],avoided,rtol=1e-12,atol=1e-7)
                    expected=avoided/record['Pavement Reference Access Burden'] if reference>0 else np.nan
                    np.testing.assert_allclose(record['Pavement Access Protection Share'],expected,rtol=1e-12,atol=1e-12)
                if delta==.25 and share in [0.,.01]:zero_counts[rule]=int(np.count_nonzero(increase==0))
                cases+=1
                print(f'National policy accounting verified {cases}/39',flush=True)
    result={'status':'passed_national_pavement_policy_construction','authorization':'MILI-D-20261008-001',
        'candidate_road_records':21415800,'selected_origins':175612,'original_residents':int(pop.sum()),
        'input_construction_confirmed':True,'replacement_priorities_confirmed':True,'expanded_kernel_validation_passed':True,
        'joint_policy_cases':36,'reference_cases':3,'group_summaries':117,'selection_storage':'selected-only; omitted candidate IDs mean false',
        'independent_ranking_and_prefix_checks_passed':True,'maximum_resource_accounting_error':max_resource_error,
        'maximum_burden_accounting_error':max(errors),'zero_increase_meshes_at_reference_or_one_percent':zero_counts,
        'human_output_acceptance_pending':True}
    (EXP/'construction_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
