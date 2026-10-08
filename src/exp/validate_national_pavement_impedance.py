"""National engineering validation and restoration-runtime benchmark.

No publication outcome is generated here. Full recomputation reconstructs all
parallel source minima independently of the incremental finite-cost engine.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from pavement_inputs import load_inputs,make_engine,fixed_observed_orders,prefix_indices
from road_service_loss import RoadServiceLoss

OUT=ROOT/'data/exp/pavement-policy-validation/national'


def full_oracle(data,selected,delta):
    factors=np.ones(len(data['lengths']))
    factors[data['eligible']]+=delta
    factors[selected]=1.
    ref=RoadServiceLoss(data['pairs'],data['lengths']*factors,
                       data['origin_edges'],data['origin_positions']*factors[data['origin_edges']],
                       data['facility_edges'],data['facility_positions']*factors[data['facility_edges']])
    return ref.distances()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((ROOT/'data/exp/pavement-policy-validation/small-network/validation.json').read_text())
    kernel=ROOT/'src/analyses/pavement_impedance.py'
    assert receipt['status']=='passed' and hashlib.sha256(kernel.read_bytes()).hexdigest()==receipt['kernel_sha256']
    start=time.perf_counter();data=load_inputs();print('National inputs loaded.',flush=True)
    engine=make_engine(data);print(f'Engine ready after {time.perf_counter()-start:.2f} seconds.',flush=True)
    baseline=data['sample']['Candidate Medical Network Distance'].to_numpy()
    engine.configure(0);actual=engine.distances()
    np.testing.assert_allclose(actual,baseline,rtol=1e-11,atol=1e-7)
    baseline_error=float(np.max(abs(actual-baseline)))
    orders=fixed_observed_orders(data['roads']);cost=data['roads']['Pavement Relative Maintenance Cost'].to_numpy()
    eligible=data['eligible']
    traffic=eligible[prefix_indices(orders['Traffic'],cost,.01)]
    condition=eligible[prefix_indices(orders['Condition'],cost,.01)]
    checks=[]
    for delta,selected,label in [(.10,[],'unmaintained_10'),(.25,[],'unmaintained_25'),
                                 (.50,[],'unmaintained_50'),(.25,eligible[::2],'half_restored'),
                                 (.25,traffic,'traffic_1pct'),(.25,condition,'condition_1pct')]:
        tick=time.perf_counter();engine.configure(delta,selected);actual=engine.distances()
        expected=full_oracle(data,selected,delta)
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-7)
        assert np.isfinite(actual).all() and (actual>=baseline-1e-7).all()
        checks.append({'label':label,'delta':delta,'selected_count':len(selected),
                       'maximum_error':float(np.max(abs(actual-expected))),
                       'seconds':time.perf_counter()-tick})
        print(f'Independent national check {label} passed: {checks[-1]["seconds"]:.2f} seconds.',flush=True)
        gc.collect()
    engine.configure(.25,eligible);all_maintained=engine.distances()
    np.testing.assert_allclose(all_maintained,baseline,rtol=1e-11,atol=1e-7)
    engine.configure(.25);unmaintained=engine.distances()
    np.save(OUT/'all_unmaintained_origin_costs_25.npy',unmaintained)
    np.save(OUT/'all_unmaintained_node_costs_25.npy',engine.node_distances)
    rng=np.random.default_rng(20261007)
    choices=[int(eligible[np.argmax(cost)]),int(eligible[np.argmax(data['roads']['Observed Traffic Volume'])])]
    for candidates in [np.intersect1d(eligible,data['origin_edges']),np.intersect1d(eligible,data['facility_edges'])]:
        if len(candidates):choices.append(int(candidates[0]))
    multi=np.flatnonzero(np.bincount(engine.eligible_pair_group)>1)
    if len(multi):choices.append(int(eligible[np.flatnonzero(engine.eligible_pair_group==multi[0])[0]]))
    choices=list(dict.fromkeys(choices+[int(x) for x in rng.choice(eligible,3,replace=False)]))
    restored_checks=[]
    for edge in choices:
        fast=engine.restore_one(edge);expected=full_oracle(data,[edge],.25)
        np.testing.assert_allclose(fast,expected,rtol=1e-11,atol=1e-7)
        assert (fast<=unmaintained+1e-7).all() and (fast>=baseline-1e-7).all()
        restored_checks.append({'edge':edge,'maximum_error':float(np.max(abs(fast-expected))),**engine.last_diagnostics})
        print(f'Isolated restoration {edge} independently verified.',flush=True)
        gc.collect()
    benchmark_edges=np.unique(np.r_[choices,rng.choice(eligible,200,replace=False)])
    timings=[];max_nodes=0
    for edge in benchmark_edges:
        indices,values=engine.restore_one(int(edge),return_changes=True)
        assert (values<=unmaintained[indices]+1e-7).all()
        timings.append(engine.last_diagnostics['seconds'])
        max_nodes=max(max_nodes,engine.last_diagnostics['changed_nodes'])
    result={'status':'passed','national_validation_completed':True,'paper_results_generated':False,
            'authorizations':['MILI-D-20261007-042','MILI-D-20261007-045','MILI-D-20261007-046'],
            'source_roads':len(data['lengths']),'eligible_roads':len(eligible),'origins':len(baseline),
            'population':int(data['population'].sum()),'baseline_maximum_error':baseline_error,
            'joint_independent_checks':checks,'isolated_independent_checks':restored_checks,
            'all_maintained_reproduces_baseline':True,'reachability_unchanged':True,
            'partial_legs_and_parallel_alternatives_preserved':True,
            'benchmark':{'roads':len(benchmark_edges),'total_seconds':float(sum(timings)),
                         'mean_seconds':float(np.mean(timings)),'median_seconds':float(np.median(timings)),
                         'maximum_seconds':float(max(timings)),'largest_changed_node_count':max_nodes,
                         'linear_extrapolation_seconds':float(np.mean(timings)*len(eligible)),
                         'extrapolation_is_not_a_completion_guarantee':True},
            'kernel_sha256':hashlib.sha256(kernel.read_bytes()).hexdigest(),
            'elapsed_seconds':time.perf_counter()-start}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
