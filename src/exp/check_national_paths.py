"""Diagnostic national shortest distances and reproducible representative route checks.

Uses the strict exact-endpoint graph only. These are quality diagnostics, not
accepted national accessibility estimates. Roads are never modified.
"""
import csv,gc,io,json,math,zipfile
from collections import defaultdict
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape,mapping,Point
from shapely.ops import substring
from pyproj import Transformer,Geod
from scipy.sparse import load_npz,coo_matrix,bmat
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from assemble_national_network_quality import ROOT,OUT,PROC,KEY,band,csvout


def main(proc=PROC,out=OUT,attachment_column='attached_500m'):
    PROC=proc;OUT=out
    g=load_npz(PROC/'graph_exact.npz');n=g.shape[0]
    pairs=np.load(PROC/'edge_nodes_exact.npy',mmap_mode='r');lengths=np.load(PROC/'edge_lengths.npy',mmap_mode='r')
    nodes=np.load(PROC/'nodes_exact.npy',mmap_mode='r');fids=np.load(PROC/'edge_source_ordinals.npy',mmap_mode='r')
    offsets=np.load(PROC/'tile_offsets.npy');starts=offsets[:,1]
    points=pd.read_csv(OUT/'point_attachment_diagnostics.csv',dtype={'point_id':str,'category':str,'source_id':str,'road_source_id':str,'population_raw':str},keep_default_na=False,low_memory=False,float_precision='round_trip')
    points['along_m']=pd.to_numeric(points.along_m,errors='coerce');points['offset_m']=pd.to_numeric(points.offset_m,errors='coerce')
    pop=points[points.dataset=='population'].copy();attached=pop[attachment_column].to_numpy(dtype=bool);edges=pop.edge_id.to_numpy(dtype=int)
    safe=np.maximum(0,edges);pos=pop.along_m.to_numpy(dtype=float)
    pop['geographic_band']=[band(x,y) for x,y in zip(pop.longitude,pop.latitude)]
    transformer=Transformer.from_crs(6668,4978,always_xy=True)
    px,py,pz=transformer.transform(pop.longitude.to_numpy(),pop.latitude.to_numpy(),np.zeros(len(pop)))
    path_specs=[];diagnostics=[]
    for target in ['medical','ems_base']:
        selected=(points.dataset.eq('medical')&points.category.isin(['1','2']) if target=='medical' else points.dataset.eq('fire_stations')&points.category.isin(['2','3']))&points[attachment_column]
        facilities=points[selected].copy();seeds={};positions=defaultdict(list)
        for p in facilities.itertuples():
            eid=int(p.edge_id);s=max(0.,min(float(lengths[eid]),float(p.along_m)));u,v=pairs[eid]
            positions[eid].append((s,p.point_id))
            for node,cost in [(int(u),s),(int(v),float(lengths[eid])-s)]:
                if node not in seeds or cost<seeds[node][0]:seeds[node]=(cost,p.point_id)
        seednodes=np.array(list(seeds),dtype=np.int32);costs=np.array([seeds[int(node)][0] for node in seednodes])
        column=coo_matrix((costs,(seednodes,np.zeros(len(seednodes),dtype=int))),shape=(n,1)).tocsr()
        augmented=bmat([[g,column],[column.T,None]],format='csr');del column
        assert np.all(augmented.data>=0)
        print(f'{target}: {len(facilities):,} target points; computing national diagnostic distances',flush=True)
        distances,predecessors=dijkstra(augmented,directed=False,indices=n,return_predecessors=True)
        del augmented;gc.collect()
        values=np.minimum(distances[pairs[safe,0]]+pos,distances[pairs[safe,1]]+lengths[safe]-pos)
        direct_target={}
        for i,(eid,s) in enumerate(zip(edges,pos)):
            if attached[i] and eid in positions:
                dd,dest=min((abs(s-q),pid) for q,pid in positions[eid])
                if dd<=values[i]:values[i]=dd;direct_target[i]=dest
        values[~attached]=np.inf
        fx,fy,fz=transformer.transform(facilities.longitude.to_numpy(),facilities.latitude.to_numpy(),np.zeros(len(facilities)))
        straight,_=cKDTree(np.column_stack((fx,fy,fz))).query(np.column_stack((px,py,pz)))
        ratio=np.divide(values,straight,out=np.full(len(values),np.nan),where=straight>=1000)
        finite=np.isfinite(values)
        component=pop[target+'_component_reachable_exact'].to_numpy(dtype=bool)
        assert np.array_equal(finite,component)
        for i,p in enumerate(pop.itertuples()):
            diagnostics.append(dict(point_id=p.point_id,mesh_code=p.point_id.split(':')[1],geographic_band=p.geographic_band,target=target,attached_500m=bool(p.attached_500m),routing_eligible=bool(attached[i]),network_distance_m=float(values[i]) if finite[i] else None,nearest_target_chord_distance_m=float(straight[i]),detour_screen_ratio=float(ratio[i]) if math.isfinite(ratio[i]) else None,road_source_id=p.road_source_id))
        for area in sorted(pop.geographic_band.unique()):
            candidates=np.flatnonzero(pop.geographic_band.eq(area).to_numpy()&finite&(values>5000)&np.isfinite(ratio))
            if not len(candidates):candidates=np.flatnonzero(pop.geographic_band.eq(area).to_numpy()&finite&(values>0))
            if not len(candidates):continue
            i=int(max(candidates,key=lambda j:(ratio[j] if math.isfinite(ratio[j]) else 0,values[j])))
            p=pop.iloc[i];eid=int(edges[i]);s=float(pos[i]);pieces=[];nn=[]
            if i in direct_target:
                destination=direct_target[i];q=next(q for q,pid in positions[eid] if pid==destination)
                pieces.append((eid,s,q))
            else:
                u,v=map(int,pairs[eid]);node=u if distances[u]+s<=distances[v]+lengths[eid]-s else v
                pieces.append((eid,s,0. if node==u else float(lengths[eid])))
                while predecessors[node]!=n:
                    parent=int(predecessors[node]);assert parent>=0
                    nn.append((node,parent));node=parent
                    assert len(nn)<n
                destination=seeds[node][1]
                targetpoint=points.loc[points.point_id.eq(destination)].iloc[0];teid=int(targetpoint.edge_id);tu,tv=map(int,pairs[teid]);ts=float(targetpoint.along_m)
                final=(teid,0. if node==tu else float(lengths[teid]),ts)
                pieces.append(final)
            path_specs.append(dict(path_id=f'{target}:{area}',mesh_point_id=p.point_id,geographic_band=area,target=target,destination_point_id=destination,network_distance_m=float(values[i]),nearest_target_chord_distance_m=float(straight[i]),detour_screen_ratio=float(ratio[i]) if math.isfinite(ratio[i]) else None,pieces=pieces,node_links=nn))
        del distances,predecessors,seeds;gc.collect()
    print('Resolve representative node itineraries to source road edges',flush=True)
    wanted=sorted({min(u,v)*n+max(u,v) for spec in path_specs for u,v in spec['node_links']})
    wanted=np.array(wanted,dtype=np.uint64)
    edgekey=np.minimum(pairs[:,0],pairs[:,1]).astype(np.uint64)*np.uint64(n)+np.maximum(pairs[:,0],pairs[:,1]).astype(np.uint64)
    inds=np.searchsorted(wanted,edgekey);mask=inds<len(wanted)
    safeinds=np.minimum(inds,max(0,len(wanted)-1))
    mask&=(wanted[safeinds]==edgekey) if len(wanted) else False
    found={}
    for eid in np.flatnonzero(mask):
        k=int(edgekey[eid])
        if k not in found or lengths[eid]<lengths[found[k]]:found[k]=int(eid)
    assert len(found)==len(wanted)
    del edgekey,inds,mask,safeinds;gc.collect()
    for spec in path_specs:
        middle=[]
        for u,v in spec.pop('node_links'):
            eid=found[min(u,v)*n+max(u,v)];a,b=map(int,pairs[eid]);length=float(lengths[eid])
            middle.append((eid,0. if u==a else length,0. if v==a else length))
        pieces=spec['pieces'];spec['pieces']=([pieces[0],*middle,pieces[-1]] if len(pieces)>1 else pieces)
        assert abs(sum(abs(a-b) for eid,a,b in spec['pieces'])-spec['network_distance_m'])<1e-4
    wanted_sources=defaultdict(dict)
    for spec in path_specs:
        for eid,a,b in spec['pieces']:
            ti=np.searchsorted(starts,eid,side='right')-1;tile=str(int(offsets[ti,0]));wanted_sources[tile][int(fids[eid])]=eid
    geometries={};transforms={}
    for tile,ordinals in wanted_sources.items():
        info=json.loads((PROC/'tiles'/(tile+'.json')).read_text());tr=Transformer.from_crs(6668,info['local_crs'],always_xy=True)
        transforms[tile]=(tr,Transformer.from_crs(tr.target_crs,6668,always_xy=True))
        path=ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip'
        with zipfile.ZipFile(path) as z:
            name=next(v for v in z.namelist() if v.endswith('.geojson'));j=0
            with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as stream:
                for line in stream:
                    if not line.lstrip().startswith('{ "type": "Feature",'):continue
                    j+=1
                    if j in ordinals:
                        f=json.loads(line.strip().rstrip(','));eid=ordinals[j]
                        geom=shapely.transform(shape(f['geometry']),tr.transform,interleaved=False)
                        assert abs(geom.length-lengths[eid])<1e-7
                        geometries[eid]=(tile,geom)
                    if j>=max(ordinals):break
        print(f'Route source geometry {tile}: {len(ordinals)} edges',flush=True)
    routefeatures=[];summaries=[];maximumgap=0.;geod=Geod(ellps='GRS80')
    for spec in path_specs:
        previous=None;total=0.;used=[]
        for order,(eid,start,end) in enumerate(spec.pop('pieces')):
            if abs(start-end)<1e-8:continue
            tile,line=geometries[eid];geom=substring(line,start,end);total+=geom.length
            lonlat=shapely.transform(geom,transforms[tile][1].transform,interleaved=False)
            if previous is not None:
                gap=abs(geod.inv(*previous,*lonlat.coords[0])[2]);maximumgap=max(maximumgap,gap);assert gap<.001
            previous=lonlat.coords[-1]
            source=f'{tile}:{fids[eid]}';used.append(source)
            routefeatures.append(dict(type='Feature',properties=dict(path_id=spec['path_id'],segment_order=order,source_id=source,length_m=geom.length,edge_id=eid),geometry=mapping(lonlat)))
        assert abs(total-spec['network_distance_m'])<1e-4
        summaries.append(dict(spec,route_edges=len(used),source_ids=';'.join(used),geometry_length_m=total))
    csvout(OUT/'population_path_diagnostics.csv',diagnostics)
    csvout(OUT/'representative_route_checks.csv',summaries)
    (OUT/'representative_routes.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=routefeatures)))
    result=dict(status='passed',mesh_target_distance_records=len(diagnostics),representative_routes=len(summaries),maximum_route_join_gap_m=maximumgap,component_reachability_matches_distances=True,metric='Per-tile AEQD lengths; diagnostic approximation, no speeds or attachment connector distances',sample_selection='One high detour-screen route per geographic band and target; purposive, not an error-rate sample')
    (OUT/'path_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
