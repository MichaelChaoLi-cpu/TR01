"""Independent accounting, source integrity and known-route checks for the pilot."""
from collections import defaultdict,Counter
from pathlib import Path
import csv,json,math
import numpy as np
import pandas as pd
from shapely import LineString
from pyproj import Transformer,network
from build_vehicle_network_pilot import graph,multisource,ROOT,OUT,PROC,csvout

def known_routes():
    def edge(i,xy,level=0):return dict(edge_id=i,geometry=LineString(xy),level=level)
    def point(pid,kind,eid,pos):return dict(point_id=pid,dataset=kind,edge_id=eid,along_m=pos,attached_500m=True)
    selector=lambda p:p['dataset']=='medical'
    es=[edge(0,[(0,0),(10,0)])]
    points=[point('clinic','medical',0,2),point('mesh','population',0,8)]
    g=graph(es,False,False,{})
    assert multisource(g,es,points,selector)['mesh']==6
    es.append(edge(1,[(10,0),(20,0)],1))
    points[1]=point('mesh','population',1,8)
    assert multisource(graph(es,True,False,{}),es,points,selector)['mesh'] is None
    assert multisource(graph(es,False,False,{}),es,points,selector)['mesh']==16
    crossed=[edge(0,[(-5,0),(5,0)]),edge(1,[(0,-5),(0,5)])]
    assert graph(crossed,False,False,{})['report']['components']==2
    # Parallel endpoint pairs must retain the shortest line, not sum lengths.
    parallel=[edge(0,[(0,0),(10,0)]),edge(1,[(0,0),(5,5),(10,0)])]
    g=graph(parallel,False,False,{})
    assert g['adjacency'][0][0][1]==10
    return ['same-edge partial route is 6 m','different-level shared endpoint remains disconnected in layer-aware graph','shared-endpoint route is 16 m','interior crossing does not create a node','parallel edges retain shortest weight']

def main():
    tests=known_routes();s=json.loads((OUT/'summary.json').read_text())
    a=pd.read_csv(PROC/'point_attachments.csv',dtype={'point_id':str,'category':str,'mesh_code':str})
    edges=json.loads((PROC/'edges_layer_aware.geojson').read_text())['features'];nodes=pd.read_csv(PROC/'nodes_layer_aware.csv')
    assert len(edges)==s['stats']['candidate_edges'];assert a.point_id.is_unique
    assert len(nodes)==s['network_variants'][1]['nodes']
    assert all(e['properties']['length_m']>0 for e in edges)
    for edge in edges:
        p=edge['properties'];coords=edge['geometry']['coordinates']
        for node_id,xy in [(p['from_node'],coords[0]),(p['to_node'],coords[-1])]:
            node=nodes.iloc[node_id]
            assert math.hypot(node.easting-xy[0],node.northing-xy[1])<=.001+1e-8
    assert a.edge_id.between(0,len(edges)-1).all()
    assert (a.offset_m>=0).all()
    assert (a.along_m>=0).all()
    lengths=np.array([e['properties']['length_m'] for e in edges])
    assert np.all(a.along_m.to_numpy()<=lengths[a.edge_id.to_numpy()]+1e-7)
    for row in s['attachment_summary']:
        sub=a[a.dataset==row['dataset']];assert len(sub)==row['points'];assert int(sub.attached_500m.sum())==row['within_500m']
    hashes={}
    for name in ['n13_2024_download_manifest.csv','facility_download_manifest.csv','additional_facility_download_manifest.csv','population_2020_download_manifest.csv']:
        for r in csv.DictReader((ROOT/'data/exp/data-acquisition'/name).open()):hashes[r['filename']]=r['sha256']
    src=json.loads((OUT/'source_hashes.json').read_text())
    assert all(hashes[Path(r['path']).name]==r['sha256'] for r in src)
    routes=pd.read_csv(OUT/'mesh_route_diagnostics.csv',dtype={'mesh_code':str})
    assert len(routes)==int((a.dataset=='population').sum())*4*2
    rs=routes.groupby(['variant','target']).agg(meshes=('mesh_code','size'),attached_500m=('attached_500m','sum'),reachable_meshes=('network_distance_m','count'),median_distance_m=('network_distance_m','median')).reset_index()
    rs.to_csv(OUT/'route_summary.csv',index=False)
    changed=[]
    for target in ['medical','ems_base']:
        wide=routes[routes.target==target].pivot(index='mesh_code',columns='variant',values='network_distance_m')
        aa=wide['layer_aware_submm_seam'];bb=wide['shared_endpoint_submm_seam']
        assert bb[aa.notna()].notna().all()
        assert (bb[aa.notna()]<=aa[aa.notna()]+1e-6).all()
        for mesh in wide.index:
            x,y=aa.loc[mesh],bb.loc[mesh]
            if (pd.isna(x)!=pd.isna(y)) or (pd.notna(x) and pd.notna(y) and abs(x-y)>1e-6):changed.append(dict(mesh_code=mesh,target=target,layer_aware_distance_m=None if pd.isna(x) else x,shared_endpoint_distance_m=None if pd.isna(y) else y))
    csvout(OUT/'variant_sensitive_meshes.csv',changed,['mesh_code','target','layer_aware_distance_m','shared_endpoint_distance_m'])
    # Exhaustive endpoint-level transition ledger within this pilot.
    sites=defaultdict(list)
    inverse=Transformer.from_crs(6670,6668,always_xy=True)
    for e in edges:
        props=e['properties'];coords=e['geometry']['coordinates']
        for xy in [coords[0],coords[-1]]:sites[tuple(xy)].append(props)
    transitions=[]
    for (x,y),es in sites.items():
        levels={e['level'] for e in es}
        if len(levels)<2:continue
        lon,lat=inverse.transform(x,y)
        states=sorted(set(e['road_state'] for e in es))
        transitions.append(dict(longitude=lon,latitude=lat,endpoint_incidence=len(es),levels=';'.join(map(str,sorted(levels))),road_states=';'.join(states),source_ids=';'.join(sorted(set(e['source_id'] for e in es))),candidate_class='two-edge_bridge_or_tunnel_transition' if len(es)==2 and any(st in ['2','3'] for st in states) else 'other_multilevel_shared_endpoint'))
    csvout(OUT/'grade_transition_candidates.csv',transitions)
    transforms=[]
    for source in ['EPSG:4612','EPSG:6668']:
        for target in ['EPSG:6668','EPSG:6670']:
            tr=Transformer.from_crs(source,target,always_xy=True);tr.transform(130.25,32.7)
            transforms.append(dict(source_crs=source,target_crs=target,description=tr.description,reported_accuracy_m=tr.accuracy,network_grids_enabled=network.is_network_enabled()))
    csvout(OUT/'coordinate_transform_audit.csv',transforms)
    result=dict(status='passed',known_route_checks=tests,source_hashes_matching_acquisition=len(src),attachment_accounting='passed',variant_monotonicity='passed',grade_transition_sites=len(transitions),variant_sensitive_mesh_target_rows=len(changed))
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
