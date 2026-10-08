"""Exact adopted pavement rankings and joint finite-cost policy comparison.

AnaSOP Sections 5-7: 70,049 conditional roads; three fixed lists; complete
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

from pavement_inputs import ROOT,load_inputs,make_engine,fixed_observed_orders,prefix_indices

OUT=ROOT/'data/exp/pavement-policy-computation'
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
    national=json.loads((ROOT/'data/exp/pavement-policy-validation/national/validation.json').read_text())
    assert national['status']=='passed' and national['national_validation_completed']
    assert digest(ROOT/'src/analyses/pavement_impedance.py')==national['kernel_sha256']
    decisions=json.loads((ROOT/'data/exp/data-preprocessing/pavement-policy/decisions.json').read_text())
    assert decisions['authorization']=='MILI-D-20261007-047'
    sources=['data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet',
             'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet',
             'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet',
             'src/analyses/pavement_impedance.py','src/analyses/pavement_inputs.py',
             'src/analyses/compute_pavement_policy.py']
    hashes={p:digest(ROOT/p) for p in sources}
    config={'authorizations':['MILI-D-20261007-042','MILI-D-20261007-045','MILI-D-20261007-047'],
            'source_hashes':hashes,'distance_tolerance_reference_metres':TOL,
            'budgets':SHARES,'increments':DELTAS,'access_rank_increment':.25}
    if (OUT/'configuration.json').exists():assert json.loads((OUT/'configuration.json').read_text())==config
    else:atomic_json(OUT/'configuration.json',config)
    data=load_inputs();engine=make_engine(data)
    roads=data['roads'];eligible=data['eligible'];cost=roads['Pavement Relative Maintenance Cost'].to_numpy()
    population=data['population'];baseline=data['sample']['Candidate Medical Network Distance'].to_numpy()
    engine.configure(.25);reference=engine.distances()
    assert (reference>=baseline-TOL).all()
    score_file=OUT/'isolated_restoration_gain.npy'
    if not score_file.exists():np.save(score_file,np.full(len(eligible),np.nan))
    gains=np.load(score_file,mmap_mode='r+')
    assert len(gains)==len(eligible)
    scoring=time.perf_counter();max_nodes=max_origins=0
    for index,edge in enumerate(eligible):
        if np.isfinite(gains[index]):continue
        indices,values=engine.restore_one(int(edge),return_changes=True)
        improvements=difference(reference[indices],values)
        gains[index]=np.sum(population[indices]*improvements)/1000
        max_nodes=max(max_nodes,engine.last_diagnostics['changed_nodes'])
        max_origins=max(max_origins,engine.last_diagnostics['affected_origins'])
        if (index+1)%5000==0 or index==len(eligible)-1:
            gains.flush()
            atomic_json(OUT/'progress.json',{'stage':'scoring','completed_roads':int(np.isfinite(gains).sum()),
                                             'total_roads':len(eligible),'elapsed_seconds':time.perf_counter()-start})
            print(f'Exact restoration scores {index+1:,}/{len(eligible):,}',flush=True)
    gains.flush();assert np.isfinite(gains).all() and (gains>=0).all()
    orders=fixed_observed_orders(roads)
    orders['Access']=np.lexsort((eligible,-np.asarray(gains)/cost))
    rank_frames=[];selection_frames=[];resources={}
    total=float(cost.sum())
    for rule,order in orders.items():
        rank=roads[['Graph Source Edge ID','Pavement Relative Maintenance Cost','Observed Traffic Volume',
                    'Anchored Pavement Inspection Category']].iloc[order].copy()
        rank['Pavement Impedance Increment']=.25
        rank['Pavement Maintenance Priority']=rule
        rank['Pavement Traffic Priority Score']=roads['Observed Traffic Volume'].to_numpy()[order]/cost[order]
        rank['Pavement Isolated Restoration Gain']=np.asarray(gains)[order]
        rank['Pavement Access Priority Score']=np.asarray(gains)[order]/cost[order]
        rank['Pavement Priority Rank']=np.arange(1,len(roads)+1,dtype=np.int64)
        rank_frames.append(rank)
        previous=set()
        for share in SHARES:
            chosen=prefix_indices(order,cost,share)
            assert previous<=set(chosen);previous=set(chosen)
            requested=share*total;spent=float(cost[chosen].sum());unused=requested-spent
            assert unused>=-1e-8
            resources[(rule,share)]={'Pavement Requested Relative Budget':requested,
                 'Pavement Actual Relative Spending':spent,'Pavement Unused Relative Budget':unused,
                 'Pavement Actual Spending Share':spent/total,'Pavement Unused Budget Share':unused/total,
                 'Pavement Selected Road Count':len(chosen)}
            selected=np.zeros(len(roads),dtype=bool);selected[chosen]=True
            frame=pd.DataFrame({'Graph Source Edge ID':eligible,'Pavement Maintenance Priority':rule,
                                'Pavement Budget Share':share,'Pavement Maintenance Selected':selected})
            selection_frames.append(frame)
    pd.concat(rank_frames,ignore_index=True).to_parquet(OUT/'pavement_rankings.parquet',index=False)
    pd.concat(selection_frames,ignore_index=True).to_parquet(OUT/'pavement_selections.parquet',index=False)
    masks={'All Residents':np.ones(len(population),dtype=bool),
           'Higher Ageing Area':data['groups']['Higher Ageing Area'].to_numpy(),
           'Baseline Distance Disadvantaged Area':data['groups']['Baseline Distance Disadvantaged Area'].to_numpy()}
    references={};reference_burdens={};case_rows=[];group_rows=[]
    mesh_path=OUT/'pavement_mesh_scenarios.parquet';mesh_part=mesh_path.with_suffix('.part')
    writer=None
    try:
        for delta in DELTAS:
            path=OUT/'cases'/f'reference_{delta:.2f}.npy'
            if path.exists():values=np.load(path)
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
                    chosen=np.empty(0,dtype=np.int64) if rule=='Reference' else prefix_indices(orders[rule],cost,share)
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
    summary.to_parquet(OUT/'pavement_group_outcomes.parquet',index=False)
    assert len(summary)==117
    assert pq.ParquetFile(mesh_path).metadata.num_rows==175612*39
    for p,h in hashes.items():assert digest(ROOT/p)==h,p
    result={'status':'passed','full_policy_computation_completed':True,'eligible_roads':len(eligible),
            'origin_meshes':len(population),'original_population':int(population.sum()),
            'rank_rows':len(roads)*3,'selection_rows':len(roads)*12,'joint_policy_cases':36,
            'reference_cases':3,'mesh_scenario_rows':len(population)*39,'group_outcome_rows':len(summary),
            'source_hashes_unchanged':hashes,'distance_tolerance_reference_metres':TOL,
            'positive_isolated_gain_roads':int(np.count_nonzero(gains>0)),
            'score_stage_seconds_including_setup_and_later_cases':time.perf_counter()-scoring,
            'largest_changed_nodes_during_scoring':max_nodes,'largest_affected_origin_count_during_scoring':max_origins,
            'nested_selection_and_joint_restoration_verified':True,'case_timings':case_rows,
            'elapsed_seconds':time.perf_counter()-start,'readable_preprocessing_pending':True,
            'publication_figures_generated':False}
    atomic_json(OUT/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['case_timings','source_hashes_unchanged']},indent=2),flush=True)


if __name__=='__main__':main()
