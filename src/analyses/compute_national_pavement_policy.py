"""Exact adopted pavement rankings and joint finite-cost policy comparison.

AnaSOP Sections 5-7: 21,415,800 candidate source roads; three fixed lists; complete
affordable prefixes; all unselected eligible roads deteriorate simultaneously.
Population and fixed area groups remain unchanged. No bridge event sums enter
these outcomes. Source tables are validated before readable preprocessing.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pavement_inputs import ROOT,load_inputs,prefix_indices
from national_pavement_impedance import NationalPavementImpedance

OUT=ROOT/'data/exp/national-pavement-policy-computation'
SCORES=ROOT/'data/exp/national-pavement-access-scoring'
TOL=1e-7  # reference metres; numerical round-off, not empirical impairment.
SHARES=[.01,.05,.10,.20]
DELTAS=[.10,.25,.50]


def digest(path):
    with path.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix('.part');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def difference(upper,lower):
    values=np.asarray(upper)-np.asarray(lower)
    assert (values>=-TOL).all(), float(values.min())
    values[np.abs(values)<=TOL]=0.
    return values


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'cases').mkdir(exist_ok=True)
    lock=OUT/'computation.pid'
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except ProcessLookupError:pass
        else:raise RuntimeError('A pavement policy computation is already running')
    lock.write_text(str(os.getpid()))
    try:run()
    finally:lock.unlink(missing_ok=True)


def run():
    start=time.perf_counter()
    input_receipt=json.loads((ROOT/'data/exp/data-preprocessing/national-pavement-inputs/construction_validation.json').read_text())
    assert input_receipt['status']=='passed_confirmed_national_input_construction'
    score_receipt=json.loads((SCORES/'validation.json').read_text())
    assert score_receipt['status']=='passed_all_road_access_scores'
    kernel_receipt=json.loads((SCORES/'national_kernel_validation.json').read_text())
    assert kernel_receipt['status']=='passed'
    score_configuration=json.loads((SCORES/'configuration.json').read_text())
    for p,h in score_configuration['source_hashes'].items():assert digest(ROOT/p)==h,p
    sources=['data/processed/national-pavement-inputs/national_pavement_inputs_preprocessed.parquet',
             'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet',
             'src/analyses/compute_national_pavement_policy.py',
             'data/exp/national-pavement-access-scoring/isolated_restoration_gain.npy']
    hashes={p:digest(ROOT/p) for p in sources}
    config={'authorizations':['MILI-D-20261008-001'], 'source_hashes':hashes,
            'distance_tolerance_reference_metres':TOL,'budgets':SHARES,'increments':DELTAS,
            'access_rank_increment':.25,'traffic_priority':'scenario flow descending',
            'condition_priority':'III, II, I','common_tie_seed':20261008,
            'selection_storage':'selected-only; every omitted candidate ID is false, not missing'}
    if (OUT/'configuration.json').exists():assert json.loads((OUT/'configuration.json').read_text())==config
    else:atomic_json(OUT/'configuration.json',config)
    cost=np.load(ROOT/'data/processed/national-network-rule-validation/edge_lengths.npy',mmap_mode='r')
    eligible=np.arange(len(cost),dtype=np.int64)
    assert len(eligible)==21415800
    gains=np.load(SCORES/'isolated_restoration_gain.npy',mmap_mode='r')
    assert np.isfinite(gains).all() and (gains>=0).all()
    inputs=pd.read_parquet(ROOT/sources[0],columns=['Scenario Traffic Volume',
        'Scenario Pavement Need Category','Scenario Priority Tie Key'])
    tie=inputs['Scenario Priority Tie Key'].to_numpy()
    orders={};resources={};total=float(cost.sum());selected_rows=0
    rank_writer=selection_writer=None
    rank_path=OUT/'national_pavement_rankings.parquet';select_path=OUT/'national_pavement_selections.parquet'
    try:
        for rule in ['Traffic','Condition','Access']:
            path=OUT/f'{rule.lower()}_order.npy'
            if path.exists():order=np.load(path,mmap_mode='r')
            else:
                score=inputs['Scenario Traffic Volume'].to_numpy() if rule=='Traffic' else (
                    inputs['Scenario Pavement Need Category'].to_numpy(dtype=float) if rule=='Condition' else np.asarray(gains)/cost)
                order=np.lexsort((eligible,tie,-score));np.save(path,order)
                del score
            assert len(order)==len(cost) and np.unique(order).size==len(cost)
            orders[rule]=order
            cumulative=np.cumsum(cost[order],dtype=np.float64)
            previous=0
            for share in SHARES:
                count=int(np.searchsorted(cumulative,share*total,side='right'))
                assert count>=previous;previous=count
                chosen=order[:count];spent=float(cost[chosen].sum());unused=share*total-spent
                assert unused>=-1e-5
                if count<len(cost):assert cumulative[count]>share*total
                np.save(OUT/'cases'/f'{rule.lower()}_{share:.2f}_selected.npy',chosen)
                resources[(rule,share)]={'Pavement Requested Relative Budget':share*total,
                    'Pavement Actual Relative Spending':spent,'Pavement Unused Relative Budget':unused,
                    'Pavement Actual Spending Share':spent/total,'Pavement Unused Budget Share':unused/total,
                    'Pavement Selected Road Count':count}
                for begin in range(0,count,250000):
                    part=chosen[begin:begin+250000]
                    frame=pd.DataFrame({'Graph Source Edge ID':part,'Pavement Maintenance Priority':rule,
                        'Pavement Budget Share':share,'Pavement Maintenance Selected':True})
                    table=pa.Table.from_pandas(frame,preserve_index=False)
                    if selection_writer is None:selection_writer=pq.ParquetWriter(select_path,table.schema,compression='zstd')
                    selection_writer.write_table(table)
                selected_rows+=count
            for begin in range(0,len(cost),250000):
                part=order[begin:begin+250000]
                frame=pd.DataFrame({'Graph Source Edge ID':part,'Pavement Maintenance Priority':rule,
                    'Pavement Impedance Increment':.25,'Pavement Priority Rank':np.arange(begin+1,begin+1+len(part)),
                    'Pavement Traffic Priority Score':inputs['Scenario Traffic Volume'].to_numpy()[part],
                    'Pavement Isolated Restoration Gain':gains[part],'Pavement Access Priority Score':gains[part]/cost[part],
                    'Pavement Relative Maintenance Cost':cost[part]})
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if rank_writer is None:rank_writer=pq.ParquetWriter(rank_path,table.schema,compression='zstd')
                rank_writer.write_table(table)
            print(f'National {rule} list and four budget prefixes saved',flush=True)
            del cumulative
    finally:
        if rank_writer:rank_writer.close()
        if selection_writer:selection_writer.close()
    del inputs,tie
    gc.collect()
    print('Loading validated all-road routing engine',flush=True)
    data=load_inputs()
    engine=NationalPavementImpedance(data['pairs'],data['lengths'],data['origin_edges'],data['origin_positions'],
        data['facility_edges'],data['facility_positions'],graph=data['graph'],cache=SCORES/'array-index',
        progress=lambda msg:print(msg,flush=True))
    population=data['population'];baseline=data['sample']['Candidate Medical Network Distance'].to_numpy()
    masks={'All Residents':np.ones(len(population),dtype=bool),
           'Higher Ageing Area':data['groups']['Higher Ageing Area'].to_numpy(),
           'Baseline Distance Disadvantaged Area':data['groups']['Baseline Distance Disadvantaged Area'].to_numpy()}
    references={};reference_burdens={};case_rows=[];group_rows=[]
    mesh_path=OUT/'national_pavement_mesh_scenarios.parquet';mesh_part=mesh_path.with_suffix('.part')
    writer=None
    try:
        for delta in DELTAS:
            path=OUT/'cases'/f'reference_{delta:.2f}.npy'
            if path.exists():values=np.load(path)
            elif delta==.25:
                values=np.load(SCORES/'all_road_reference_0.25.npy');np.save(path,values)
            else:
                engine.configure(delta);values=engine.distances();np.save(path,values)
            assert np.isfinite(values).all()
            references[delta]=values
            extra=difference(values,baseline)
            reference_burdens[delta]={g:float(np.sum(population[mask]*extra[mask])/1000) for g,mask in masks.items()}
        for delta in DELTAS:
            for rule in ['Reference','Traffic','Condition','Access']:
                previous=references[delta]
                for share in ([0.] if rule=='Reference' else SHARES):
                    chosen=np.empty(0,dtype=np.int64) if rule=='Reference' else np.load(OUT/'cases'/f'{rule.lower()}_{share:.2f}_selected.npy',mmap_mode='r')
                    path=OUT/'cases'/f'{rule.lower()}_{share:.2f}_{delta:.2f}.npy'
                    tick=time.perf_counter()
                    if rule=='Reference':values=references[delta]
                    elif path.exists():values=np.load(path)
                    else:
                        engine.configure(delta,eligible[chosen]);values=engine.distances();np.save(path,values)
                    assert np.isfinite(values).all() and (values<=references[delta]+TOL).all()
                    assert (values<=previous+TOL).all();previous=values
                    extra=difference(values,baseline)
                    resource=resources[(rule,share)] if rule!='Reference' else {
                        'Pavement Requested Relative Budget':0.,'Pavement Actual Relative Spending':0.,
                        'Pavement Unused Relative Budget':0.,'Pavement Actual Spending Share':0.,
                        'Pavement Unused Budget Share':0.,'Pavement Selected Road Count':0}
                    for group,mask in masks.items():
                        h0=reference_burdens[delta][group]
                        residual=float(np.sum(population[mask]*extra[mask])/1000)
                        avoided=h0-residual
                        assert avoided>=-1e-6 and avoided<=h0+1e-6
                        if abs(avoided)<=1e-10:avoided=0.
                        protection=avoided/h0 if h0>0 else np.nan
                        group_rows.append({'Pavement Impedance Increment':delta,'Pavement Maintenance Priority':rule,
                             'Pavement Budget Share':share,'Pavement Population Group':group,
                             'Pavement Reference Access Burden':h0,'Pavement Residual Access Burden':residual,
                             'Pavement Avoided Access Burden':avoided,'Pavement Access Protection Share':protection,**resource})
                    frame=data['sample'][['Mesh Code','Local Total Population','Candidate Medical Network Distance',
                                          'Longitude','Latitude']].copy()
                    frame['Pavement Impedance Increment']=delta;frame['Pavement Maintenance Priority']=rule
                    frame['Pavement Budget Share']=share;frame['Pavement Medical Access Cost']=values
                    frame['Pavement Medical Access Cost Increase']=extra
                    table=pa.Table.from_pandas(frame,preserve_index=False)
                    if writer is None:writer=pq.ParquetWriter(mesh_part,table.schema,compression='zstd')
                    writer.write_table(table)
                    case_rows.append({'rule':rule,'share':share,'delta':delta,'selected_roads':len(chosen),
                                      'seconds_this_pass':time.perf_counter()-tick})
                    atomic_json(OUT/'progress.json',{'stage':'joint_evaluation','completed_cases':len(case_rows),
                                                     'total_cases':39,'elapsed_seconds':time.perf_counter()-start})
                    print(f'Joint policy {rule} budget={share:.0%} increment={delta:.0%}; {len(case_rows)}/39',flush=True)
    finally:
        if writer:writer.close()
    mesh_part.replace(mesh_path)
    summary=pd.DataFrame(group_rows)
    summary.to_parquet(OUT/'national_pavement_group_outcomes.parquet',index=False)
    assert len(summary)==117
    assert pq.ParquetFile(mesh_path).metadata.num_rows==175612*39
    for p,h in hashes.items():assert digest(ROOT/p)==h,p
    result={'status':'passed','full_policy_computation_completed':True,'eligible_roads':len(eligible),
            'origin_meshes':len(population),'original_population':int(population.sum()),
            'rank_rows':len(cost)*3,'selection_rows':selected_rows,'selection_storage':'selected-only; omitted IDs are false','joint_policy_cases':36,
            'reference_cases':3,'mesh_scenario_rows':len(population)*39,'group_outcome_rows':len(summary),
            'source_hashes_unchanged':hashes,'distance_tolerance_reference_metres':TOL,
            'positive_isolated_gain_roads':int(np.count_nonzero(gains>0)),
            'isolated_scoring_reused_from_completed_validated_run':True,
            'nested_selection_and_joint_restoration_verified':True,'case_timings':case_rows,
            'elapsed_seconds':time.perf_counter()-start,'readable_preprocessing_pending':True,
            'publication_figures_generated':False}
    atomic_json(OUT/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['case_timings','source_hashes_unchanged']},indent=2),flush=True)


if __name__=='__main__':main()
