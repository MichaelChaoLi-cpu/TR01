"""Resumable nationwide evaluation of the adopted source-road loss scenario.

All original graph source roads are represented. A verified unchanged-path
witness screens exact analytic cases; others use exact incremental recomputation.
No traffic-match filter or geographic sampling reduces the road domain.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/analyses'))
from road_service_loss import RoadServiceLoss
from road_service_loss_dynamic import DynamicRoadServiceLoss

OUT = ROOT/'data/exp/national-road-service-loss'
PROC = ROOT/'data/processed/national-network-rule-validation'
CACHE = ROOT/'data/exp/road-service-loss-validation/dynamic'


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def atomic_json(path, value):
    part = path.with_suffix(f'.{os.getpid()}.part')
    part.write_text(json.dumps(value,indent=2)+'\n');part.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,default=10000)
    parser.add_argument('--max-batches',type=int);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'batches').mkdir(exist_ok=True)
    lock=OUT/'computation.pid'
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except ProcessLookupError:pass
        else:raise RuntimeError('A national loss computation already holds this output')
    lock.write_text(str(os.getpid()))
    try:
        run(args)
    finally:
        if lock.exists() and int(lock.read_text())==os.getpid():lock.unlink()


def run(args):
    start=time.perf_counter();validation=json.loads((CACHE/'validation.json').read_text())
    assert validation['status']=='passed' and validation['sparse_updates_reconstruct_full_distances']
    hashes=validation['source_hashes_verified_unchanged']
    for p,h in hashes.items():assert digest(ROOT/p)==h,p
    assert digest(CACHE/'source_removal_replacement_weights.npy')==validation['replacement_weights_sha256']
    config={'scenario_authorization':'MILI-D-20261007-013','source_hashes':hashes,
            'dynamic_kernel_sha256':digest(ROOT/'src/analyses/road_service_loss_dynamic.py'),
            'reference_kernel_sha256':digest(ROOT/'src/analyses/road_service_loss.py'),
            'screen_sha256':digest(CACHE/'critical_source_road_screen.npy'),
            'batch_size':args.batch_size,'source_road_domain':21415800,'tolerance_metres':1e-7}
    config_path=OUT/'configuration.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Inputs changed; preserve the previous run before starting a different configuration'
    else:atomic_json(config_path,config)
    sample=pd.read_parquet(ROOT/'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet')
    points=pd.read_csv(ROOT/'data/exp/national-population-entry/point_attachment_diagnostics.csv',
                       dtype={'point_id':str,'category':str,'population_raw':str,'road_source_id':str},
                       keep_default_na=False,low_memory=False,float_precision='round_trip')
    points['along_m']=pd.to_numeric(points['along_m'],errors='coerce')
    origins=points.set_index('point_id').loc['mesh:'+sample['Mesh Code'].astype(str)]
    facilities=points[points['dataset'].eq('medical')&points['category'].isin(['1','2'])&points['routing_eligible']]
    pairs=np.load(PROC/'edge_nodes_exact.npy',mmap_mode='r');lengths=np.load(PROC/'edge_lengths.npy',mmap_mode='r')
    population=sample['Local Total Population'].to_numpy(dtype=np.int64)
    ref=RoadServiceLoss(pairs,lengths,origins['edge_id'].to_numpy(),origins['along_m'].to_numpy(),
                        facilities['edge_id'].to_numpy(),facilities['along_m'].to_numpy(),graph=load_npz(PROC/'graph_exact.npz'))
    del points,facilities;gc.collect()
    fast=DynamicRoadServiceLoss(ref,replacements=np.load(CACHE/'source_removal_replacement_weights.npy',mmap_mode='r'))
    np.testing.assert_allclose(fast.baseline,sample['Candidate Medical Network Distance'],rtol=1e-11,atol=1e-7)
    critical=np.load(CACHE/'critical_source_road_screen.npy',mmap_mode='r')
    assert len(critical)==len(lengths) and int(critical.sum())==validation['exact_witness_screen_recompute_roads']
    roads=np.flatnonzero(critical);n=len(lengths)
    # Source entry population uses integer accumulation, never a floating census sum.
    entry_population=np.zeros(n,dtype=np.int64);np.add.at(entry_population,ref.origin_edges,population)
    lost_path=OUT/'lost_medical_path_population.npy';extra_path=OUT/'additional_medical_distance_burden.npy'
    if not lost_path.exists():
        lost=np.lib.format.open_memmap(lost_path,mode='w+',dtype=np.int64,shape=(n,));lost[:]=entry_population;lost[critical]=-1;lost.flush()
        extra=np.lib.format.open_memmap(extra_path,mode='w+',dtype=np.float64,shape=(n,));extra[:]=0.;extra[critical]=np.nan;extra.flush()
    lost=np.load(lost_path,mmap_mode='r+');extra=np.load(extra_path,mmap_mode='r+')
    completed=0;computed_this_run=0;timings=[]
    for batch_start in range(0,len(roads),args.batch_size):
        edges=roads[batch_start:batch_start+args.batch_size];path=OUT/'batches'/f'{batch_start:08d}.npz'
        if path.exists():
            with np.load(path) as cache:
                np.testing.assert_array_equal(cache['edges'],edges)
                lost[edges]=cache['lost'];extra[edges]=cache['extra'];timings.append(float(cache['seconds']))
            completed+=len(edges);continue
        batch_time=time.perf_counter();losses=[];burdens=[];max_nodes=0
        for edge in edges:
            indices,distances=fast.distances(int(edge),return_changes=True)
            finite=np.isfinite(distances)
            delta=distances[finite]-fast.baseline[indices[finite]]
            assert (delta>=-config['tolerance_metres']).all()
            delta[np.abs(delta)<=config['tolerance_metres']]=0
            loss=int(population[indices[~finite]].sum())
            assert entry_population[edge]<=loss<=int(population.sum())
            burden=float(np.sum(population[indices[finite]]*delta)/1000)
            losses.append(loss);burdens.append(burden)
            max_nodes=max(max_nodes,fast.last_diagnostics['Invalidated Nodes'])
        seconds=time.perf_counter()-batch_time
        losses=np.asarray(losses,dtype=np.int64);burdens=np.asarray(burdens,dtype=np.float64)
        part=path.with_suffix('.part')
        with part.open('wb') as handle:np.savez_compressed(handle,edges=edges,lost=losses,extra=burdens,seconds=seconds)
        part.replace(path);lost[edges]=losses;extra[edges]=burdens;lost.flush();extra.flush()
        completed+=len(edges);computed_this_run+=1;timings.append(seconds)
        atomic_json(OUT/'progress.json',{'execution_status':'running','all_source_roads':n,
                    'analytic_witness_cases':int((~critical).sum()),'conditional_recompute_roads':len(roads),
                    'completed_recompute_roads':completed,'last_batch_seconds':seconds,
                    'last_batch_roads':len(edges),'largest_invalidated_subtree_last_batch':max_nodes,
                    'elapsed_this_run_seconds':time.perf_counter()-start,'national_complete':False})
        print(f'Exact removals {completed:,}/{len(roads):,}; batch {seconds:.2f} s; max subtree {max_nodes:,}',flush=True)
        if args.max_batches is not None and computed_this_run>=args.max_batches:break
    complete=completed==len(roads)
    if complete:
        assert (lost>=0).all() and np.isfinite(extra).all() and (extra>=0).all()
        for p,h in hashes.items():assert digest(ROOT/p)==h,p
        import pyarrow as pa
        import pyarrow.parquet as pq
        output=OUT/'national_road_service_loss.parquet'
        writer=None
        try:
            for first in range(0,n,500000):
                last=min(first+500000,n)
                table=pa.table({'Graph Source Edge ID':np.arange(first,last,dtype=np.int32),
                                'Lost Medical Path Population':np.asarray(lost[first:last]),
                                'Additional Medical Distance Burden':np.asarray(extra[first:last])})
                if writer is None:writer=pq.ParquetWriter(output,table.schema,compression='zstd')
                writer.write_table(table)
        finally:
            if writer:writer.close()
        assert pq.ParquetFile(output).metadata.num_rows==n
    atomic_json(OUT/'validation.json',{'status':'passed' if complete else 'checkpointed',
                'national_complete':complete,'source_road_rows':n,'analytic_witness_cases':int((~critical).sum()),
                'conditional_recompute_roads':len(roads),'completed_recompute_roads':completed,
                'original_meshes':len(sample),'original_population':int(population.sum()),
                'all_source_roads_preserved':True,'traffic_filter_applied':False,
                'total_recorded_batch_seconds':sum(timings),'source_hashes_unchanged':hashes,
                'scenario_authorization':config['scenario_authorization']})
    atomic_json(OUT/'progress.json',{'execution_status':'complete' if complete else 'checkpointed',
                'all_source_roads':n,'analytic_witness_cases':int((~critical).sum()),
                'conditional_recompute_roads':len(roads),'completed_recompute_roads':completed,
                'national_complete':complete,'total_recorded_batch_seconds':sum(timings)})


if __name__=='__main__':main()
