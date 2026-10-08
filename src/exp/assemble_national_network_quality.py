"""Assemble national diagnostic graphs, fixed point attachments and anomaly ledgers.

Strict level-coded and shared-coordinate endpoint graphs are diagnostic variants,
not accepted final vehicle networks. No flagged short gap or interior crossing is joined.
"""
import csv,gc,hashlib,json,math
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix,save_npz
from scipy.sparse.csgraph import connected_components
from screen_national_network_tiles import ROOT,OUT,CACHE,band
from build_vehicle_network_pilot import csvout

PROC=ROOT/'data/processed/national-network-quality'
KEY=np.dtype([('x','<f8'),('y','<f8'),('level','<i2')])


def graph(pairs,lengths,n,path):
    # Deduplicate undirected node pairs by minimum length, never sum parallel lengths.
    a=np.minimum(pairs[:,0],pairs[:,1]).astype(np.uint64);b=np.maximum(pairs[:,0],pairs[:,1]).astype(np.uint64)
    keys=a*np.uint64(n)+b;del a,b
    order=np.argsort(keys);sortedkeys=keys[order];del keys
    starts=np.r_[0,np.flatnonzero(sortedkeys[1:]!=sortedkeys[:-1])+1]
    unique=sortedkeys[starts];weights=np.minimum.reduceat(np.asarray(lengths)[order],starts)
    del sortedkeys,order,starts
    u=(unique//np.uint64(n)).astype(np.int32);v=(unique%np.uint64(n)).astype(np.int32);del unique
    keep=u!=v;u=u[keep];v=v[keep];weights=weights[keep]
    csr=coo_matrix((np.r_[weights,weights],(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()
    save_npz(path,csr,compressed=False)
    nc,labels=connected_components(csr,directed=False)
    edgecomp=labels[pairs[:,0]];componentlength=np.bincount(edgecomp,weights=lengths,minlength=nc)
    result=dict(nodes=n,edges=len(pairs),components=int(nc),total_length_km=float(np.sum(lengths)/1000),largest_component_length_share=float(componentlength.max()/componentlength.sum()),isolated_nodes=int(np.sum(np.diff(csr.indptr)==0)))
    return labels,edgecomp,componentlength,result


def main():
    summaries=json.loads((OUT/'tile_screen_summary.json').read_text());assert len(summaries)==160
    n=sum(r['vehicle_edges'] for r in summaries);offsets={};pos=0
    for r in summaries:offsets[r['tile']]=pos;pos+=r['vehicle_edges']
    np.save(PROC/'tile_offsets.npy',np.array([(int(r['tile']),offsets[r['tile']],r['vehicle_edges']) for r in summaries],dtype=np.int64))
    keys=np.lib.format.open_memmap(PROC/'endpoint_keys.npy',mode='w+',dtype=KEY,shape=(2*n,))
    lengths=np.lib.format.open_memmap(PROC/'edge_lengths.npy',mode='w+',dtype=np.float64,shape=(n,))
    fids=np.lib.format.open_memmap(PROC/'edge_source_ordinals.npy',mode='w+',dtype=np.int32,shape=(n,))
    xy=np.load(OUT/'point_coordinates.npy',mmap_mode='r');best=np.full(len(xy),math.inf);edge=np.full(len(xy),-1,dtype=np.int32);along=np.full(len(xy),np.nan)
    for r in summaries:
        tile=r['tile'];o=offsets[tile];count=r['vehicle_edges'];z=np.load(CACHE/(tile+'.npz'))
        sl=slice(2*o,2*(o+count));ends=z['ends'].reshape(-1,2)
        keys['x'][sl]=ends[:,0];keys['y'][sl]=ends[:,1];keys['level'][sl]=np.repeat(z['levels'],2)
        lengths[o:o+count]=z['lengths'];fids[o:o+count]=z['fids']
        att=np.load(CACHE/(tile+'_attachments.npz'));pi=att['point_indices'];dist=att['distances'];ei=att['edge_indices']+o
        update=(dist<best[pi]-1e-9)|((np.abs(dist-best[pi])<=1e-9)&(ei<edge[pi]))
        pi=pi[update];best[pi]=dist[update];edge[pi]=ei[update];along[pi]=att['along'][update]
    keys.flush();lengths.flush();fids.flush()
    np.savez(PROC/'point_attachment_arrays.npz',edge_id=edge,offset_m=best,along_m=along)
    print(f'Assembled {n:,} edges; sorting exact level-coded endpoints',flush=True)
    unique,inverse=np.unique(keys,return_inverse=True)
    np.save(PROC/'nodes_exact.npy',unique)
    pairs=inverse.reshape(-1,2).astype(np.int32);del inverse,keys
    np.save(PROC/'edge_nodes_exact.npy',pairs)
    labels,ec,cl,strict=graph(pairs,lengths,len(unique),PROC/'graph_exact.npz')
    np.save(PROC/'node_components_exact.npy',labels);np.save(PROC/'edge_components_exact.npy',ec)
    csvout(OUT/'components_exact.csv',(dict(component=i,edges=int(c),length_km=float(cl[i]/1000)) for i,c in enumerate(np.bincount(ec,minlength=len(cl)))),['component','edges','length_km'])
    print('Strict graph:',strict,flush=True)
    del labels,cl;gc.collect()
    newsite=np.r_[True,(unique['x'][1:]!=unique['x'][:-1])|(unique['y'][1:]!=unique['y'][:-1])]
    remap=np.cumsum(newsite,dtype=np.int32)-1
    shared=remap[pairs];np.save(PROC/'edge_nodes_shared.npy',shared)
    gradeidx=np.flatnonzero(~newsite)
    csvout(OUT/'global_multilevel_endpoint_sites.csv',(dict(longitude=float(unique[j]['x']),latitude=float(unique[j]['y']),level_a=int(unique[j-1]['level']),level_b=int(unique[j]['level'])) for j in gradeidx),['longitude','latitude','level_a','level_b'])
    ns=int(remap[-1]+1);del unique,newsite,remap,pairs;gc.collect()
    labels,shared_ec,cl,permissive=graph(shared,lengths,ns,PROC/'graph_shared.npz')
    np.save(PROC/'node_components_shared.npy',labels);np.save(PROC/'edge_components_shared.npy',shared_ec)
    del labels,cl,shared;gc.collect()
    print('Shared-coordinate diagnostic graph:',permissive,flush=True)
    points=pd.read_csv(OUT/'points.csv',dtype=str,keep_default_na=False)
    points['edge_id']=edge;points['offset_m']=np.where(np.isfinite(best),best,np.nan);points['along_m']=along
    points['attached_100m']=best<=100;points['attached_500m']=best<=500
    tiles=np.array([int(r['tile']) for r in summaries]);starts=np.array([offsets[r['tile']] for r in summaries])
    valid=edge>=0;ti=np.searchsorted(starts,edge[valid],side='right')-1
    points['road_source_id']='';points.loc[valid,'road_source_id']=[f'{tiles[i]:04}:{fids[e]}' for i,e in zip(ti,edge[valid])]
    points['component_exact']=-1;points['component_shared']=-1
    points.loc[valid,'component_exact']=ec[edge[valid]];points.loc[valid,'component_shared']=shared_ec[edge[valid]]
    # Reachability by component does not assert a traversable off-road connector.
    for target in ['medical','ems_base']:
        selector=(points.dataset.eq('medical')&points.category.isin(['1','2']) if target=='medical' else points.dataset.eq('fire_stations')&points.category.isin(['2','3']))
        eligible=selector&points.attached_500m
        for variant in ['exact','shared']:
            components=set(points.loc[eligible,'component_'+variant])
            points[target+'_component_reachable_'+variant]=points.attached_500m&points['component_'+variant].isin(components)
    points.to_csv(OUT/'point_attachment_diagnostics.csv',index=False)
    points.loc[~points.attached_500m].to_csv(OUT/'point_attachment_anomalies.csv',index=False)
    pop=points[points.dataset=='population'].copy();pop['population_numeric']=pd.to_numeric(pop.population_raw,errors='coerce')
    questionable=pop.attached_500m&((~pop.medical_component_reachable_exact)|(~pop.ems_base_component_reachable_exact)|pop.medical_component_reachable_exact.ne(pop.medical_component_reachable_shared)|pop.ems_base_component_reachable_exact.ne(pop.ems_base_component_reachable_shared))
    pop.loc[questionable].to_csv(OUT/'population_connectivity_candidates.csv',index=False)
    points['geographic_band']=[band(float(x),float(y)) for x,y in zip(points.longitude,points.latitude)]
    summary=[]
    for dataset,t in points.groupby('dataset',sort=True):
        ds=pd.to_numeric(t.offset_m,errors='coerce')
        summary.append(dict(dataset=dataset,points=len(t),within_100m=int(t.attached_100m.sum()),within_500m=int(t.attached_500m.sum()),no_eligible_line_within_5km=int(t.edge_id.eq(-1).sum()),median_offset_m=float(ds.median()),p95_offset_m=float(ds.quantile(.95))))
    csvout(OUT/'attachment_summary.csv',summary)
    pop['geographic_band']=[band(float(x),float(y)) for x,y in zip(pop.longitude,pop.latitude)]
    regional=[]
    for area,t in pop.groupby('geographic_band',sort=True):
        regional.append(dict(geographic_band=area,meshes=len(t),inhabited_published_meshes=int(t.population_numeric.gt(0).sum()),attached_500m=int(t.attached_500m.sum()),medical_reachable_exact=int(t.medical_component_reachable_exact.sum()),medical_reachable_shared=int(t.medical_component_reachable_shared.sum()),ems_reachable_exact=int(t.ems_base_component_reachable_exact.sum()),ems_reachable_shared=int(t.ems_base_component_reachable_shared.sum())))
    csvout(OUT/'regional_population_screen.csv',regional)
    for suffix,dest in [('grade','multilevel_endpoint_candidates.csv'),('gaps','near_dangling_endpoint_candidates.csv'),('crossings','sampled_interior_crossing_candidates.csv')]:
        files=[CACHE/(r['tile']+'_'+suffix+'.csv') for r in summaries]
        with (OUT/dest).open('w',newline='') as output:
            writer=None
            for p in files:
                reader=csv.DictReader(p.open())
                if writer is None:writer=csv.DictWriter(output,fieldnames=reader.fieldnames);writer.writeheader()
                writer.writerows(reader)
    samplepool=[]
    for r in summaries:samplepool.extend(json.loads((CACHE/(r['tile']+'_samples.json')).read_text()))
    chosen=[]
    for group in sorted({(r['band'],r['kind']) for r in samplepool}):
        options=[r for r in samplepool if (r['band'],r['kind'])==group]
        chosen.extend(sorted(options,key=lambda r:hashlib.sha256((r['source_ids']+str(r['longitude'])).encode()).digest())[:1])
    # One large-offset point per geographic band, with nearest source trace.
    for area,t in points[points.offset_m.notna()&~points.attached_100m].groupby('geographic_band'):
        p=t.loc[pd.to_numeric(t.offset_m).idxmax()]
        chosen.append(dict(kind='large_point_offset',longitude=float(p.longitude),latitude=float(p.latitude),tile=p.road_source_id.split(':')[0],source_ids=p.road_source_id,levels='',road_states='',band=area,geometries=[],point_id=p.point_id,offset_m=float(p.offset_m)))
    for i,r in enumerate(chosen,1):r['review_id']=f'N{i:03}'
    (OUT/'map_review_samples.json').write_text(json.dumps(chosen,indent=2)+'\n')
    csvout(OUT/'map_review_queue.csv',[{k:r.get(k,'') for k in ['review_id','kind','band','longitude','latitude','tile','source_ids','levels','road_states','point_id','offset_m']} for r in chosen])
    result=dict(archives=160,vehicle_edges=n,exact_graph=strict,shared_coordinate_diagnostic_graph=permissive,
                attachments=summary,total_points=len(points),population_meshes=len(pop),
                component_reachability={target:{variant:int(pop[f'{target}_component_reachable_{variant}'].sum()) for variant in ['exact','shared']} for target in ['medical','ems_base']},
                global_multilevel_adjacent_level_records=len(gradeidx),
                local_gap_candidates=sum(r['local_nearest_other_endpoint_candidates'] for r in summaries),
                structure_lines_sampled_for_crossings=sum(r['bridge_tunnel_lines_sampled_for_crossings'] for r in summaries),
                sampled_crossing_records=sum(r['sampled_crossing_records'] for r in summaries),map_review_records=len(chosen),
                rules='Ordinary centerlines, positive length; ground-level ordinary-state non-expressway attachment; 5 km search cap; diagnostic 100/500 m thresholds; no repairs; AEQD metric per source tile')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
