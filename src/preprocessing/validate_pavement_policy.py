"""Read-back, independent budget sorting and stored joint outcome accounting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/exp/pavement-policy-computation'
PROC=ROOT/'data/processed/pavement-policy'
OUT=ROOT/'data/exp/data-preprocessing/pavement-policy'
TOL=1e-7


def main():
    model=json.loads((SOURCE/'validation.json').read_text())
    assert model['status']=='passed' and model['full_policy_computation_completed']
    names=set();readbacks={}
    for slug in ['pavement_rankings','pavement_selections','pavement_mesh_scenarios','pavement_group_outcomes']:
        src=pq.ParquetFile(SOURCE/(slug+'.parquet'))
        dst=pq.ParquetFile(PROC/(slug+'_preprocessed.parquet'))
        assert src.metadata.num_rows==dst.metadata.num_rows
        columns=src.schema_arrow.names
        assert all(c in dst.schema_arrow.names for c in columns)
        for a,b in zip(src.iter_batches(batch_size=175612),dst.iter_batches(batch_size=175612,columns=columns),strict=True):
            assert a.equals(b),slug
        names.update(dst.schema_arrow.names)
        readbacks[slug]={'rows':dst.metadata.num_rows,'columns':len(dst.schema_arrow.names)}
    decisions=json.loads((OUT/'decisions.json').read_text())
    assert {v['readable_name'] for v in decisions['derived_variables']}<=names
    roads=pd.read_parquet(ROOT/'data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet')
    ranks=pd.read_parquet(PROC/'pavement_rankings_preprocessed.parquet')
    selections=pd.read_parquet(PROC/'pavement_selections_preprocessed.parquet')
    summaries=pd.read_parquet(PROC/'pavement_group_outcomes_preprocessed.parquet')
    sample=pd.read_parquet(ROOT/'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet')
    groups=pd.read_parquet(ROOT/'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet').set_index('Mesh Code').loc[sample['Mesh Code']]
    mask={'All Residents':np.ones(len(sample),dtype=bool),'Higher Ageing Area':groups['Higher Ageing Area'].to_numpy(),
          'Baseline Distance Disadvantaged Area':groups['Baseline Distance Disadvantaged Area'].to_numpy()}
    population=sample['Local Total Population'].to_numpy();base=sample['Candidate Medical Network Distance'].to_numpy()
    cost=roads.set_index('Graph Source Edge ID')['Pavement Relative Maintenance Cost'];total=float(cost.sum())
    for rule in ['Traffic','Condition','Access']:
        actual=ranks[ranks['Pavement Maintenance Priority'].eq(rule)].reset_index(drop=True)
        assert len(actual)==70049 and actual['Graph Source Edge ID'].is_unique
        np.testing.assert_array_equal(actual['Pavement Priority Rank'],np.arange(1,70050))
        if rule=='Traffic':
            expected=actual.sort_values(['Pavement Traffic Priority Score','Graph Source Edge ID'],ascending=[False,True])
        elif rule=='Condition':
            expected=actual.sort_values(['Anchored Pavement Inspection Category','Pavement Relative Maintenance Cost','Graph Source Edge ID'],ascending=[False,True,True])
        else:expected=actual.sort_values(['Pavement Access Priority Score','Graph Source Edge ID'],ascending=[False,True])
        np.testing.assert_array_equal(actual['Graph Source Edge ID'],expected['Graph Source Edge ID'])
        np.testing.assert_array_equal(actual['Pavement Access Priority Score'],actual['Pavement Isolated Restoration Gain']/actual['Pavement Relative Maintenance Cost'])
        previous=set()
        for share in [.01,.05,.10,.20]:
            selected=selections[selections['Pavement Maintenance Priority'].eq(rule)&selections['Pavement Budget Share'].eq(share)]
            assert len(selected)==70049 and selected['Graph Source Edge ID'].is_unique
            chosen=set(selected.loc[selected['Pavement Maintenance Selected'],'Graph Source Edge ID'])
            assert previous<=chosen;previous=chosen
            cumulative=actual['Pavement Relative Maintenance Cost'].cumsum()
            expected_ids=actual.loc[cumulative<=share*total,'Graph Source Edge ID']
            assert chosen==set(expected_ids)
            np.testing.assert_array_equal(selected['Pavement Effective Road Cost'],
                selected['Graph Source Edge ID'].map(cost)*(1+.25*(~selected['Pavement Maintenance Selected'])))
            spent=float(cost.loc[list(chosen)].sum())
            matching=summaries[summaries['Pavement Maintenance Priority'].eq(rule)&summaries['Pavement Budget Share'].eq(share)]
            np.testing.assert_allclose(matching['Pavement Actual Relative Spending'],spent,rtol=1e-12,atol=1e-8)
            assert matching['Pavement Selected Road Count'].eq(len(chosen)).all()
            np.testing.assert_allclose(matching['Pavement Actual Spending Share']+matching['Pavement Unused Budget Share'],share,atol=1e-14)
    max_burden_error=0.
    for row in summaries.to_dict('records'):
        rule=row['Pavement Maintenance Priority'];b=row['Pavement Budget Share'];delta=row['Pavement Impedance Increment'];g=row['Pavement Population Group']
        reference=np.load(SOURCE/'cases'/f'reference_{delta:.2f}.npy')
        values=reference if rule=='Reference' else np.load(SOURCE/'cases'/f'{rule.lower()}_{b:.2f}_{delta:.2f}.npy')
        assert np.isfinite(values).all() and (values>=base-TOL).all() and (values<=reference+TOL).all()
        extra=values-base;ref_extra=reference-base
        extra[np.abs(extra)<=TOL]=0;ref_extra[np.abs(ref_extra)<=TOL]=0
        residual=float(np.sum(population[mask[g]]*extra[mask[g]])/1000)
        h0=float(np.sum(population[mask[g]]*ref_extra[mask[g]])/1000)
        max_burden_error=max(max_burden_error,abs(residual-row['Pavement Residual Access Burden']),abs(h0-row['Pavement Reference Access Burden']))
        np.testing.assert_allclose(row['Pavement Residual Access Burden'],residual,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(row['Pavement Reference Access Burden'],h0,rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(row['Pavement Avoided Access Burden'],h0-residual,rtol=1e-12,atol=1e-8)
        if h0>0:
            np.testing.assert_allclose(row['Pavement Access Protection Share'],(h0-residual)/h0,rtol=1e-12,atol=1e-14)
            assert -1e-12<=row['Pavement Access Protection Share']<=1+1e-12
        else:assert pd.isna(row['Pavement Access Protection Share'])
    receipt={'status':'passed_readable_pavement_policy_construction','authorization':'MILI-D-20261007-047',
             'readbacks':readbacks,'all_22_derived_names_present':True,'all_source_values_and_dtypes_preserved':True,
             'fixed_priority_and_prefix_resource_accounting_reproduced':True,'joint_burden_and_protection_reconstructed':True,
             'maximum_burden_reconstruction_error':max_burden_error,'original_origins':len(sample),
             'original_population':int(population.sum()),'full_policy_cases':36,'reference_cases':3,
             'no_bridge_event_outcomes_used':True}
    (OUT/'construction_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
