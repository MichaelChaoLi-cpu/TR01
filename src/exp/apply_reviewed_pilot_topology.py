"""Apply only reviewed pilot connections and recalculate fixed-attachment distances.

Inputs are immutable prior pilot artifacts. All outputs use separate directories.
"""
from collections import defaultdict, Counter
import csv, hashlib, heapq, json, math
from pathlib import Path
import numpy as np
from shapely.geometry import shape, Point, mapping
from shapely.ops import substring
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from build_vehicle_network_pilot import ROOT, PROJECTOR, Union, multisource, csvout

BASE = ROOT / 'data/processed/vehicle-network-pilot'
REVIEW = ROOT / 'data/exp/network-map-review'
OUT = ROOT / 'data/exp/reviewed-network-pilot'
PROC = ROOT / 'data/processed/reviewed-network-pilot'


def readcsv(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def make_graph(nodes, pairs, records, union=None):
    roots = [union.find(i) if union else i for i in range(len(nodes))]
    unique = sorted(set(roots)); lookup = {r:i for i,r in enumerate(unique)}
    remap = [lookup[r] for r in roots]
    pairs = np.array([(remap[u],remap[v]) for u,v in pairs],dtype=int)
    lengths = np.array([e['geometry'].length for e in records])
    weights = {}; chosen = {}
    for eid, ((u,v),w) in enumerate(zip(pairs,lengths)):
        if u == v: continue
        for a,b in [(int(u),int(v)),(int(v),int(u))]:
            if w < weights.get((a,b),math.inf): weights[a,b]=w; chosen[a,b]=eid
    adj = [[] for _ in unique]
    for (u,v),w in weights.items(): adj[u].append((v,w))
    sparse = coo_matrix((list(weights.values()),([u for u,v in weights],[v for u,v in weights])),shape=(len(unique),len(unique))).tocsr()
    nc,labels = connected_components(sparse,directed=False)
    ec = labels[pairs[:,0]]; cl = np.bincount(ec,weights=lengths,minlength=nc)
    return dict(nodes=[nodes[r] for r in unique],pairs=pairs,lengths=lengths,
                adjacency=adj,labels=labels,edge_comp=ec,component_length=cl,
                chosen_edges=chosen,node_remap=remap,
                report=dict(nodes=len(unique),edges=len(records),components=int(nc),
                    total_length_m=float(lengths.sum()),largest_component_length_share=float(cl.max()/cl.sum())))


def load_inputs():
    features = json.loads((BASE/'edges_layer_aware.geojson').read_text())['features']
    records = [dict(f['properties'],geometry=shape(f['geometry'])) for f in features]
    nodes = [(float(r['easting']),float(r['northing']),int(r['level'])) for r in readcsv(BASE/'nodes_layer_aware.csv')]
    points = readcsv(BASE/'point_attachments.csv')
    for p in points:
        p['edge_id']=int(p['edge_id']); p['along_m']=float(p['along_m'])
        for key in ['attached_100m','attached_500m']: p[key]=p[key]=='True'
    review = readcsv(REVIEW/'review_results.csv')
    return records,nodes,points,review


def correct(records,nodes,review):
    nodes=list(nodes); breaks=defaultdict(list); grade_actions=[]; crosses=[]
    for row in review:
        if row['status']!='connect_supported': continue
        xy=Point(*PROJECTOR.transform(float(row['longitude']),float(row['latitude'])))
        ids=list(map(int,row['edge_ids'].split(';')))
        if row['kind']=='multilevel_endpoint':
            participating=set()
            for eid in ids:
                e=records[eid]
                for end,key in [(0,'from_node'),(-1,'to_node')]:
                    if xy.distance(Point(e['geometry'].coords[end])) < .001:
                        participating.add(e[key])
            assert len(participating)>=2, row['review_id']
            grade_actions.append((row['review_id'],sorted(participating)))
        else:
            assert len(ids)==2
            intersection=records[ids[0]]['geometry'].intersection(records[ids[1]]['geometry'])
            candidates=list(intersection.geoms) if intersection.geom_type=='MultiPoint' else [intersection]
            point=min(candidates,key=lambda p:p.distance(xy))
            assert point.geom_type=='Point' and point.distance(xy)<.001
            participating=[]
            for eid in ids:
                e=records[eid]; pos=e['geometry'].project(point); length=e['geometry'].length
                if pos<1e-7: node=e['from_node']; pos=0.
                elif length-pos<1e-7: node=e['to_node']; pos=length
                else:
                    matches=[(s,n) for s,n in breaks[eid] if abs(s-pos)<1e-7]
                    if matches: node=matches[0][1]
                    else:
                        node=len(nodes); q=e['geometry'].interpolate(pos)
                        nodes.append((q.x,q.y,e['level'])); breaks[eid].append((pos,node))
                participating.append(node)
            crosses.append((row['review_id'],participating))
    union=Union(len(nodes)); actions=grade_actions+crosses
    for rid,nn in actions:
        for n in nn[1:]: union.join(nn[0],n)
    segments=[]; pairs=[]; parent_segments=defaultdict(list)
    for eid,e in enumerate(records):
        bp=sorted([(0.,e['from_node']),*breaks[eid],(e['geometry'].length,e['to_node'])])
        for part,((start,u),(end,v)) in enumerate(zip(bp,bp[1:])):
            line=substring(e['geometry'],start,end)
            assert line.geom_type=='LineString' and line.length>0
            sid=len(segments)
            segments.append(dict(e,edge_id=sid,parent_edge_id=eid,split_part=part,
                                 parent_start_m=start,parent_end_m=end,geometry=line))
            pairs.append((u,v)); parent_segments[eid].append(sid)
        assert abs(sum(segments[i]['geometry'].length for i in parent_segments[eid])-e['geometry'].length)<1e-6
    graph=make_graph(nodes,pairs,segments,union)
    # Check direct topology treatment, independently of reachability through other roads.
    for row in review:
        if row['kind']!='multilevel_endpoint' or row['status']!='unresolved': continue
        xy=Point(*PROJECTOR.transform(float(row['longitude']),float(row['latitude'])))
        old=set()
        for eid in map(int,row['edge_ids'].split(';')):
            for end,key in [(0,'from_node'),(-1,'to_node')]:
                if xy.distance(Point(records[eid]['geometry'].coords[end]))<.001: old.add(records[eid][key])
        assert len({union.find(n) for n in old})==len(old)
    for row in review:
        if row['status']!='connect_supported' and row['kind']!='multilevel_endpoint':
            assert row['review_id'] not in {rid for rid,nn in actions}
    audit=[dict(review_id=rid,operation='endpoint_join' if rid.startswith('G') else 'surface_junction_split_join',
                premerge_node_ids=';'.join(map(str,nn)),corrected_node_id=graph['node_remap'][nn[0]]) for rid,nn in actions]
    return segments,graph,parent_segments,audit


def remap_points(points,segments,parent_segments):
    result=[]
    for p in points:
        pos=p['along_m']; children=parent_segments[p['edge_id']]
        sid=next(i for i in children if segments[i]['parent_start_m']-1e-7<=pos<=segments[i]['parent_end_m']+1e-7)
        e=segments[sid]; q=dict(p,parent_edge_id=p['edge_id'],edge_id=sid,
                               along_m=max(0.,min(e['geometry'].length,pos-e['parent_start_m'])))
        result.append(q)
    return result


def route_example(g,records,points,selector,pop):
    """Reconstruct one shortest route with explicit partial source/destination edges."""
    dist=np.full(len(g['nodes']),math.inf); prev={}; origin={}; queue=[]; direct=[]
    for p in points:
        if not p['attached_500m'] or not selector(p): continue
        eid=p['edge_id'];pos=p['along_m'];u,v=g['pairs'][eid];length=records[eid]['geometry'].length
        if eid==pop['edge_id']: direct.append((abs(pos-pop['along_m']),p))
        for node,d in [(u,pos),(v,length-pos)]:
            if d<dist[node]: dist[node]=d;origin[int(node)]=p;prev[int(node)]=None;heapq.heappush(queue,(d,int(node)))
    while queue:
        d,u=heapq.heappop(queue)
        if d!=dist[u]: continue
        for v,w in g['adjacency'][u]:
            if d+w<dist[v]: dist[v]=d+w;prev[v]=(u,g['chosen_edges'][u,v]);origin[v]=origin[u];heapq.heappush(queue,(d+w,v))
    eid=pop['edge_id'];pos=pop['along_m'];length=records[eid]['geometry'].length;u,v=g['pairs'][eid]
    value,node=min((dist[u]+pos,int(u)),(dist[v]+length-pos,int(v)))
    pieces=[]; route_nodes=[]
    if direct and min(d for d,p in direct)<=value:
        value,target=min(direct,key=lambda item:item[0]);pieces=[(eid,pos,target['along_m'])]
    else:
        target=origin[node];pieces.append((eid,pos,0. if node==u else length))
        while True:
            route_nodes.append(node)
            if prev[node] is None: break
            parent,edge=prev[node];eu,ev=g['pairs'][edge];el=records[edge]['geometry'].length
            pieces.append((edge,0. if node==eu else el,0. if parent==eu else el));node=parent
        teid=target['edge_id'];tu,tv=g['pairs'][teid];tl=records[teid]['geometry'].length
        pieces.append((teid,0. if node==tu else tl,target['along_m']))
    features=[];total=0.
    for edge,start,end in pieces:
        if abs(start-end)<1e-8: continue
        geom=substring(records[edge]['geometry'],start,end);total+=geom.length
        features.append(dict(type='Feature',geometry=mapping(geom),properties=dict(edge_id=edge,source_id=records[edge]['source_id'],length_m=geom.length)))
    assert abs(total-value)<1e-5
    return value,target['point_id'],route_nodes,features


def main():
    OUT.mkdir(parents=True,exist_ok=True);PROC.mkdir(parents=True,exist_ok=True)
    inputpaths=[BASE/'edges_layer_aware.geojson',BASE/'nodes_layer_aware.csv',BASE/'point_attachments.csv',REVIEW/'review_results.csv',ROOT/'data/exp/vehicle-network-pilot/mesh_route_diagnostics.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputpaths}
    records,nodes,points,review=load_inputs()
    base=make_graph(nodes,[(e['from_node'],e['to_node']) for e in records],records)
    segments,g,parent_segments,audit=correct(records,nodes,review)
    attached=remap_points(points,segments,parent_segments)
    assert abs(base['report']['total_length_m']-g['report']['total_length_m'])<1e-6
    assert Counter(r['operation'] for r in audit)=={'endpoint_join':139,'surface_junction_split_join':3}
    selectors=dict(medical=lambda p:p['dataset']=='medical' and p['category'] in ('1','2'),
                   ems_base=lambda p:p['dataset']=='fire_stations' and p['category'] in ('2','3'))
    oldrows=readcsv(ROOT/'data/exp/vehicle-network-pilot/mesh_route_diagnostics.csv')
    original={(p['mesh_code'],p['target']):float(p['network_distance_m']) if p['network_distance_m'] else None for p in oldrows if p['variant']=='layer_aware_submm_seam'}
    rows=[]; examples=[]; routefeatures=[]; counts={}
    for target,selector in selectors.items():
        before=multisource(base,records,points,selector);after=multisource(g,segments,attached,selector)
        targetrows=[]
        for p in attached:
            if p['dataset']!='population': continue
            a=before.get(p['point_id']);b=after.get(p['point_id']);old=original[p['mesh_code'],target]
            assert (a is None)==(old is None)
            if a is not None: assert abs(a-old)<1e-6 and b is not None and b<=a+1e-6
            change='unattached' if not p['attached_500m'] else 'still_disconnected' if b is None else 'newly_reachable' if a is None else 'shorter' if a-b>1e-6 else 'unchanged'
            row=dict(mesh_code=p['mesh_code'],target=target,attached_500m=p['attached_500m'],before_distance_m=a,after_distance_m=b,
                     reduction_m=None if a is None or b is None else a-b,change=change,population_raw=p['population_raw'],suppression_code=p['suppression_code'])
            rows.append(row);targetrows.append(row)
        counts[target]=dict(Counter(r['change'] for r in targetrows))
        candidates=[]
        new=[r for r in targetrows if r['change']=='newly_reachable']
        shorter=[r for r in targetrows if r['change']=='shorter']
        if new: candidates.append(min(new,key=lambda r:r['after_distance_m']))
        if shorter: candidates.append(max(shorter,key=lambda r:r['reduction_m']))
        for row in candidates:
            pop=next(p for p in attached if p['point_id']=='mesh:'+row['mesh_code'])
            value,destination,nn,ff=route_example(g,segments,attached,selector,pop)
            assert abs(value-row['after_distance_m'])<1e-5
            used=sorted({r['review_id'] for r in audit if r['corrected_node_id'] in nn})
            assert used, 'A changed route must traverse a reviewed correction'
            examples.append(dict(row,destination_point_id=destination,review_ids_on_path=';'.join(used),reconstructed_length_m=value))
            for f in ff: f['properties'].update(mesh_code=row['mesh_code'],target=target,change=row['change']);routefeatures.append(f)
    csvout(OUT/'mesh_distance_comparison.csv',rows)
    csvout(OUT/'changed_meshes.csv',[r for r in rows if r['change'] in ('newly_reachable','shorter')])
    csvout(OUT/'topology_actions.csv',audit)
    csvout(OUT/'route_examples.csv',examples)
    csvout(PROC/'point_attachments.csv',attached)
    csvout(PROC/'nodes.csv',[dict(node_id=i,easting=x,northing=y,level=l,component=int(g['labels'][i])) for i,(x,y,l) in enumerate(g['nodes'])])
    features=[]
    for e,(u,v) in zip(segments,g['pairs']):
        props={k:v for k,v in e.items() if k!='geometry'}
        props.update(from_node=int(u),to_node=int(v),length_m=e['geometry'].length,component=int(g['edge_comp'][e['edge_id']]))
        features.append(dict(type='Feature',properties=props,geometry=mapping(e['geometry'])))
    crs=dict(type='name',properties=dict(name='urn:ogc:def:crs:EPSG::6670'))
    for path,ff in [(PROC/'edges.geojson',features),(OUT/'route_examples.geojson',routefeatures)]:
        path.write_text(json.dumps(dict(type='FeatureCollection',crs=crs,features=ff)))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    result=dict(baseline_graph=base['report'],reviewed_graph=g['report'],applied_records=142,
                endpoint_candidates_applied=139,surface_junction_candidates_applied=3,
                parent_edges_split=len([v for v in parent_segments.values() if len(v)>1]),
                separated_crossings_retained=163,unresolved_records_without_forced_join=55,
                distances=counts,reconstructed_route_examples=len(examples),input_hashes=hashes,
                validation='passed',rules='Fixed attachments and destination eligibility; bidirectional distance weights; no speeds; no national extrapolation')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
