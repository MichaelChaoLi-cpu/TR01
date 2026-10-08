"""Build an auditable conservative structure-continuity candidate graph.

Only global degree-two exact-coordinate transitions with matching road classes,
levels {0,1} or {-1,0}, and states {1,2} or {1,3} are candidates. Known unresolved
visual sites are excluded. Source edges, lengths and point anchors are preserved.
"""
from collections import Counter
import csv,gc,hashlib,json,os
import numpy as np
import pandas as pd
from build_vehicle_network_pilot import ROOT,Union,csvout
from assemble_national_network_quality import PROC as BASE,OUT as QC,graph

OUT=ROOT/'data/exp/national-network-rule-validation'
PROC=ROOT/'data/processed/national-network-rule-validation'


def sitekey(x,y):return round(float(x),8),round(float(y),8)


def main():
    OUT.mkdir(parents=True,exist_ok=True);PROC.mkdir(parents=True,exist_ok=True)
    nodes=np.load(BASE/'nodes_exact.npy',mmap_mode='r');pairs=np.load(BASE/'edge_nodes_exact.npy',mmap_mode='r')
    lengths=np.load(BASE/'edge_lengths.npy',mmap_mode='r');offsets=np.load(BASE/'tile_offsets.npy')
    newsite=np.r_[True,(nodes['x'][1:]!=nodes['x'][:-1])|(nodes['y'][1:]!=nodes['y'][:-1])]
    site=np.cumsum(newsite,dtype=np.int32)-1;starts=np.flatnonzero(newsite)
    degree=np.bincount(pairs.ravel(),minlength=len(nodes));global_incidence=np.add.reduceat(degree,starts)
    del degree,starts,newsite;gc.collect()
    held={};known=[]
    for folder,name in [(QC,'map_review_results.csv'),(ROOT/'data/exp/network-map-review','review_results.csv'),(OUT,'independent_map_review_results.csv')]:
        if not (folder/name).exists():continue
        with (folder/name).open() as f:
            for r in csv.DictReader(f):
                if r['kind'] not in ['bridge_endpoint','tunnel_endpoint','multilevel_endpoint']:continue
                k=sitekey(r['longitude'],r['latitude']);known.append(r)
                if r['status'] in ['unresolved','separation_supported','no_connect_supported']:held[k]=r['review_id']
    metadata={};baseoffset={str(t):int(o) for t,o,n in offsets}
    for t,o,n in offsets:
        z=np.load(BASE/'tiles'/f'{t}.npz');metadata[str(t)]={key:z[key] for key in ['fids','ends','levels','states','classes','widths']}
    rows=[];actions=[];parent=np.arange(len(nodes),dtype=np.int32)
    frame=pd.read_csv(QC/'multilevel_endpoint_candidates.csv',dtype={'tile':str,'source_ids':str,'levels':str,'road_states':str},float_precision='round_trip')
    seen=set()
    for r in frame.itertuples():
        endpoints=[];properties=[];edges=[]
        for source in r.source_ids.split(';'):
            tile,fid=source.split(':');m=metadata[tile];i=int(np.searchsorted(m['fids'],int(fid)))
            assert m['fids'][i]==int(fid)
            eid=baseoffset[tile]+i;edges.append(eid)
            for j in [0,1]:
                if m['ends'][i,j,0]==r.longitude and m['ends'][i,j,1]==r.latitude:endpoints.append(int(pairs[eid,j]))
            properties.append((int(m['levels'][i]),int(m['states'][i]),int(m['classes'][i]),int(m['widths'][i])))
        assert endpoints
        nk=tuple(sorted(set(endpoints)))
        if nk in seen:continue
        seen.add(nk);inc=int(global_incidence[site[endpoints[0]]])
        levels={p[0] for p in properties};states={p[1] for p in properties};classes={p[2] for p in properties}
        reason='eligible_conservative_structure_transition'
        if inc!=2 or len(edges)!=2 or len(nk)!=2:reason='not_global_two_endpoint_transition'
        elif levels not in [{0,1},{-1,0}]:reason='outside_ground_adjacent_level_pair'
        elif states not in [{1,2},{1,3}]:reason='outside_ordinary_to_single_structure_pair'
        elif len(classes)!=1:reason='road_class_change_requires_review'
        elif sitekey(r.longitude,r.latitude) in held:reason='withheld_prior_unresolved_visual_review'
        apply=reason=='eligible_conservative_structure_transition'
        row=dict(longitude=r.longitude,latitude=r.latitude,source_ids=r.source_ids,global_endpoint_incidence=inc,
                 levels=';'.join(map(str,sorted(levels))),road_states=';'.join(map(str,sorted(states))),
                 road_classes=';'.join(map(str,sorted(classes))),width_classes=';'.join(map(str,sorted({p[3] for p in properties}))),
                 baseline_node_ids=';'.join(map(str,nk)),applied=apply,reason=reason)
        rows.append(row)
        if apply:
            a,b=nk;assert parent[a]==a and parent[b]==b
            parent[b]=a;actions.append(row)
    keep=parent==np.arange(len(nodes));lookup=np.cumsum(keep,dtype=np.int32)-1
    remap=lookup[parent];corrected=remap[pairs]
    np.save(PROC/'node_remap_from_baseline.npy',remap);np.save(PROC/'nodes_exact.npy',nodes[keep]);np.save(PROC/'edge_nodes_exact.npy',corrected)
    labels,ec,cl,report=graph(corrected,lengths,int(keep.sum()),PROC/'graph_exact.npz')
    np.save(PROC/'node_components_exact.npy',labels);np.save(PROC/'edge_components_exact.npy',ec)
    for name in ['edge_lengths.npy','edge_source_ordinals.npy','tile_offsets.npy','tiles']:
        path=PROC/name
        if not path.exists():path.symlink_to(os.path.relpath(BASE/name,PROC),target_is_directory=name=='tiles')
    points=pd.read_csv(QC/'point_attachment_diagnostics.csv',dtype={'point_id':str,'category':str,'population_raw':str,'road_source_id':str},keep_default_na=False,low_memory=False,float_precision='round_trip')
    valid=points.edge_id.ge(0);points.loc[valid,'component_exact']=ec[points.loc[valid,'edge_id'].to_numpy(dtype=int)]
    before={}
    for target in ['medical','ems_base']:
        before[target]=points[target+'_component_reachable_exact'].to_numpy(dtype=bool).copy()
        selector=(points.dataset.eq('medical')&points.category.isin(['1','2']) if target=='medical' else points.dataset.eq('fire_stations')&points.category.isin(['2','3']))&points.attached_500m
        components=set(points.loc[selector,'component_exact'])
        points[target+'_component_reachable_exact']=points.attached_500m&points.component_exact.isin(components)
        assert (~before[target]|points[target+'_component_reachable_exact']).all()
        assert (~points[target+'_component_reachable_exact']|points[target+'_component_reachable_shared']).all()
    points.to_csv(OUT/'point_attachment_diagnostics.csv',index=False)
    csvout(OUT/'continuity_rule_audit.csv',rows);csvout(OUT/'applied_structure_transitions.csv',actions)
    bysite={sitekey(r['longitude'],r['latitude']):r for r in rows};evidence=[]
    for r in known:
        candidate=bysite.get(sitekey(r['longitude'],r['latitude']))
        evidence.append(dict(review_id=r['review_id'],visual_status=r['status'],longitude=r['longitude'],latitude=r['latitude'],rule_applied=bool(candidate and candidate['applied']),rule_reason=candidate['reason'] if candidate else 'not_in_vehicle_endpoint_candidate_ledger'))
    assert not any(r['rule_applied'] and r['visual_status'] in ['unresolved','separation_supported','no_connect_supported'] for r in evidence)
    csvout(OUT/'prior_visual_evidence_rule_checks.csv',evidence)
    popmask=points.dataset.eq('population').to_numpy();reach={target:dict(before=int(before[target][popmask].sum()),after=int(points.loc[popmask,target+'_component_reachable_exact'].sum())) for target in before}
    summary=dict(status='passed',baseline_components=80862,candidate_graph=report,applied_transitions=len(actions),withheld_reason_counts=dict(Counter(r['reason'] for r in rows)),population_component_reachability=reach,prior_visual_records_checked=len(evidence),known_unresolved_sites_not_applied=True,
                 source_edges_unchanged=True,total_length_unchanged=True,
                 rule='Global degree two; exact coordinate; levels {0,1} or {-1,0}; states {1,2} or {1,3}; matching road class; prior unresolved evidence withheld',
                 status_limit='Candidate rule validation, not nationwide ground-truth confirmation; interior crossings and short gaps are not repaired')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
