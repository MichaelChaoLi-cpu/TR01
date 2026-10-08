"""Provisional vehicle-network engineering pilot; preserves all raw sources.
Run .venv/bin/python src/exp/build_vehicle_network_pilot.py
No speed assignment, national estimate, or final variable selection is performed.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,heapq,io,json,math,os,sys,zipfile
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tr01-matplotlib')
import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS,Transformer
from shapely import LineString,Point,box,STRtree
from shapely.ops import transform
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from check_national_roads import features
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from correct_municipal_office_code import apply_corrections
OUT=ROOT/'data/exp/vehicle-network-pilot'; PROC=ROOT/'data/processed/vehicle-network-pilot'
BBOX=(130.15,32.50,130.60,32.85); REGION=box(*BBOX)
PROJECT=6670
PROJECTOR=Transformer.from_crs(6668,PROJECT,always_xy=True)
INVERSE=Transformer.from_crs(PROJECT,6668,always_xy=True)
SOURCE_FILES=set(); stats=Counter()

def csvout(path,rows,fields=None):
    rows=list(rows)
    if fields is None: fields=list(rows[0]) if rows else []
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def mesh_center(code):
    return (100+int(code[2:4])+int(code[5])/8+int(code[7])/80+1/160,
            int(code[:2])/1.5+int(code[4])/12+int(code[6])/120+1/240)

def roads():
    records=[];seen={}; counts=Counter(); excluded=Counter(); actions=[]
    for tile in ['4830','4930']:
        path=ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip';SOURCE_FILES.add(path)
        with zipfile.ZipFile(path) as z:
            name=next(n for n in z.namelist() if n.endswith('.geojson'))
            with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as f:
                for index,obj in enumerate(features(f,{}),1):
                    c=obj['geometry']['coordinates']; p=obj['properties']
                    xs=[v[0] for v in c];ys=[v[1] for v in c]
                    if max(xs)<BBOX[0] or min(xs)>BBOX[2] or max(ys)<BBOX[1] or min(ys)>BBOX[3]: continue
                    original=LineString(c)
                    if not original.intersects(REGION):continue
                    counts[str(p['N13_002'])]+=1
                    if str(p['N13_002'])!='1':excluded['non_ordinary_centerline']+=1;continue
                    if str(p['N13_004']) in ('5','6','7'):excluded['construction_other_unknown_state']+=1;continue
                    clean=[c[0]]
                    for pt in c[1:]:
                        if pt!=clean[-1]:clean.append(pt)
                    fid=f'{tile}:{index}'
                    if len(clean)<2:
                        excluded['zero_length']+=1;actions.append(dict(source_id=fid,action='remove_zero_length_in_pilot'));continue
                    if len(clean)!=len(c):actions.append(dict(source_id=fid,action='remove_repeated_consecutive_vertices_in_pilot'))
                    # Only collapse exact/reversed geometry with identical properties.
                    key=(min(tuple(map(tuple,clean)),tuple(map(tuple,reversed(clean)))),json.dumps(p,sort_keys=True))
                    if key in seen:
                        excluded['duplicate_identical_geometry_and_attributes']+=1;actions.append(dict(source_id=fid,action='remove_exact_duplicate_in_pilot',retained_source_id=seen[key]));continue
                    seen[key]=fid
                    clipped=LineString(clean).intersection(REGION)
                    pieces=[clipped] if clipped.geom_type=='LineString' else list(clipped.geoms) if clipped.geom_type in ('MultiLineString','GeometryCollection') else []
                    for part,g in enumerate(pieces):
                        if g.geom_type!='LineString' or g.length==0:continue
                        projected=transform(PROJECTOR.transform,g)
                        records.append(dict(edge_id=len(records),source_id=fid,part=part,tile=tile,level=int(p['N13_005']),road_class=str(p['N13_003']),road_state=str(p['N13_004']),width_class=str(p['N13_006']),geometry=projected,lonlat=g))
        print(f'{tile}: selected cumulative {len(records):,} candidate vehicle edges',flush=True)
    csvout(OUT/'road_type_counts.csv',[dict(centerline_type=k,features_in_bbox=v) for k,v in counts.items()])
    csvout(OUT/'geometry_actions.csv',actions,['source_id','action','retained_source_id'])
    stats.update({f'excluded_{k}':v for k,v in excluded.items()});stats['candidate_edges']=len(records)
    return records

class Union:
    def __init__(self,n): self.p=list(range(n))
    def find(self,a):
        while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def seam_repairs(records):
    endpoints=[];seen={};incidence=Counter()
    for e in records:
        for j in (0,-1):
            x,y=e['geometry'].coords[j];key=(x,y,e['level']);incidence[key]+=1
            if key not in seen:seen[key]=len(endpoints);endpoints.append((x,y,e['level'],e['tile'],tuple(e['lonlat'].coords[j])))
    union=Union(len(endpoints));repairs=[]
    tree=cKDTree(np.array([e[:2] for e in endpoints]))
    for a,b in sorted(tree.query_pairs(.001)):
        x,y,l,t,geo=endpoints[a];xx,yy,ll,tt,geob=endpoints[b]
        if l!=ll or t==tt:continue
        # Only the shared national tile latitude boundary; no general short-gap repair.
        if abs(geo[1]-32-2/3)>1e-8 or abs(geob[1]-32-2/3)>1e-8:continue
        if incidence[(x,y,l)]!=1 and incidence[(xx,yy,ll)]!=1:continue
        d=math.hypot(x-xx,y-yy)
        if d==0:continue
        union.join(a,b);repairs.append(dict(longitude_a=geo[0],latitude_a=geo[1],longitude_b=geob[0],latitude_b=geob[1],distance_m=d,level=l,rule='same_level_cross_tile_below_1mm'))
    mapping={key:endpoints[union.find(i)][:2] for key,i in seen.items()}
    csvout(OUT/'submillimetre_seam_repairs.csv',repairs,['longitude_a','latitude_a','longitude_b','latitude_b','distance_m','level','rule'])
    stats['submillimetre_seam_pairs_merged']=len(repairs)
    return mapping

def graph(records,layered,repaired,mapping):
    lookup={};nodes=[];pairs=[];lengths=[];geo_sites=defaultdict(set)
    for e in records:
        ends=[]
        for x,y in (e['geometry'].coords[0],e['geometry'].coords[-1]):
            if repaired:x,y=mapping[(x,y,e['level'])]
            key=(x,y,e['level']) if layered else (x,y)
            if key not in lookup:lookup[key]=len(nodes);nodes.append((x,y,e['level'] if layered else None))
            ends.append(lookup[key]);geo_sites[(x,y)].add(e['level'])
        pairs.append(ends);lengths.append(e['geometry'].length)
    pairs=np.asarray(pairs,dtype=int);lengths=np.asarray(lengths)
    # Duplicate/parallel endpoint pairs retain minimum physical line length.
    weights={}
    for (u,v),w in zip(pairs,lengths):
        if u==v:continue
        for a,b in [(int(u),int(v)),(int(v),int(u))]:weights[(a,b)]=min(w,weights.get((a,b),math.inf))
    rr=[a for a,b in weights];cc=[b for a,b in weights]
    sparse=coo_matrix((list(weights.values()),(rr,cc)),shape=(len(nodes),len(nodes))).tocsr()
    ncomp,labels=connected_components(sparse,directed=False)
    edge_comp=labels[pairs[:,0]]; component_length=np.bincount(edge_comp,weights=lengths,minlength=ncomp)
    adjacency=[[] for _ in nodes]
    for (u,v),w in weights.items():adjacency[u].append((v,w))
    report=dict(variant=('layer_aware' if layered else 'shared_endpoint')+('_submm_seam' if repaired else '_exact'),nodes=len(nodes),edges=len(records),components=int(ncomp),largest_component_length_share=float(component_length.max()/lengths.sum()),same_coordinate_multilevel_sites=sum(len(s)>1 for s in geo_sites.values()),self_loop_edges=int((pairs[:,0]==pairs[:,1]).sum()))
    return dict(nodes=nodes,pairs=pairs,lengths=lengths,labels=labels,edge_comp=edge_comp,component_length=component_length,adjacency=adjacency,report=report)

def intersections(records):
    geoms=[e['geometry'] for e in records];tree=STRtree(geoms);counts=Counter();examples=[]
    pairs=tree.query(geoms,predicate='intersects').T
    for a,b in pairs:
        if a>=b:continue
        ga,gb=geoms[a],geoms[b]
        shared=set(ga.coords[i] for i in (0,-1))&set(gb.coords[i] for i in (0,-1))
        if shared:continue
        inter=ga.intersection(gb)
        typ='interior_crossing' if ga.crosses(gb) else 'non_endpoint_touch_or_overlap'
        same=records[a]['level']==records[b]['level'];counts[typ+('_same_level' if same else '_different_level')]+=1
        if len(examples)<200:
            pt=inter.representative_point();lon,lat=INVERSE.transform(pt.x,pt.y)
            examples.append(dict(source_a=records[a]['source_id'],source_b=records[b]['source_id'],issue=typ,same_level=same,longitude=lon,latitude=lat,state_a=records[a]['road_state'],state_b=records[b]['road_state']))
    csvout(OUT/'intersection_candidates.csv',examples,['source_a','source_b','issue','same_level','longitude','latitude','state_a','state_b'])
    return dict(counts)

def population():
    rows=[]
    for tile in ['4830','4930']:
        path=ROOT/f'data/raw/population/census_2020_1km_jgd2011/T001140/T001140_{tile}.zip';SOURCE_FILES.add(path)
        with zipfile.ZipFile(path) as z:
            with io.TextIOWrapper(z.open(z.namelist()[0]),encoding='cp932') as f:
                r=csv.reader(f); cols=next(r);next(r)
                for values in r:
                    p=dict(zip(cols,values));code=p['KEY_CODE'];x,y=mesh_center(code)
                    if REGION.covers(Point(x,y)):
                        rows.append(dict(point_id='mesh:'+code,dataset='population',longitude=x,latitude=y,source_crs=6668,category='',name='',mesh_code=code,population_raw=p['T001140001'],elderly_raw=p['T001140019'],suppression_code=p['HTKSYORI'],source_id=path.name+':'+code))
    return rows

def layers(z,provenance):
    for name in z.namelist():
        if name.endswith('.zip'):
            if not any(f'_{pref}_' in name for pref in ['42','43']):continue
            with zipfile.ZipFile(io.BytesIO(z.read(name))) as child:yield from layers(child,provenance+'!'+name)
        elif name.endswith('.shp'):yield z,name,provenance+'!'+name

def pois():
    rows=[];crs_audit=[]
    classes=dict(fire_stations='P17_003',medical='P04_001',welfare='P14_006',municipal_offices='P34_002',post_offices='P30_004',roadside_stations='P35_011')
    for folder,field in classes.items():
        for path in sorted((ROOT/'data/raw/poi'/folder).glob('*.zip')):
            if folder in ['fire_stations','municipal_offices'] and not any(f'_{pref}_' in path.name for pref in ['42','43']):continue
            SOURCE_FILES.add(path)
            with zipfile.ZipFile(path) as z:
                for child,name,prov in layers(z,path.name):
                    stem=name[:-4];reader=shapefile.Reader(shp=io.BytesIO(child.read(name)),dbf=io.BytesIO(child.read(stem+'.dbf')),encoding='utf-8' if folder=='welfare' else 'cp932')
                    if reader.shapeType!=1:continue
                    prj=stem+'.prj'
                    if prj in child.namelist():source=CRS.from_wkt(child.read(prj).decode());evidence='source .prj'
                    else:source=CRS.from_epsg(4612 if folder=='fire_stations' else 6668);evidence='official cached source specification'
                    crs_audit.append(dict(dataset=folder,layer=prov,source_crs=source.to_string(),evidence=evidence))
                    tr=Transformer.from_crs(source,6668,always_xy=True)
                    for i,sr in enumerate(reader.iterShapeRecords(),1):
                        if not sr.shape.points:continue
                        x,y=tr.transform(*sr.shape.points[0])
                        if not REGION.covers(Point(x,y)):continue
                        p=sr.record.as_dict()
                        if folder=='municipal_offices':p=apply_corrections(p,path.name)[0]
                        namefield={'fire_stations':'P17_001','medical':'P04_002','welfare':'P14_001','municipal_offices':'P34_003','post_offices':'P30_005','roadside_stations':'P35_006'}[folder]
                        rows.append(dict(point_id=f'{folder}:{len(rows)}',dataset=folder,longitude=x,latitude=y,source_crs=source.to_string(),category=str(p.get(field,'')),name=str(p.get(namefield,'')),mesh_code='',population_raw='',elderly_raw='',suppression_code='',source_id=prov+':'+str(i)))
        print(f'{folder}: selected {sum(r["dataset"]==folder for r in rows)} POI',flush=True)
    csvout(OUT/'crs_audit.csv',crs_audit)
    return rows

def attach(points,records):
    # Point entry is limited to ground-level, ordinary-state non-expressway edges.
    eligible=[i for i,e in enumerate(records) if e['level']==0 and e['road_state']=='1' and e['road_class']!='4']
    tree=STRtree([records[i]['geometry'] for i in eligible]);rows=[]
    for p in points:
        x,y=PROJECTOR.transform(p['longitude'],p['latitude']);pt=Point(x,y)
        inds,distances=tree.query_nearest(pt,return_distance=True,all_matches=True)
        # Deterministic lowest edge ID among exact-distance ties; retain ambiguity count.
        idx=min(int(i) for i in inds);eid=eligible[idx];line=records[eid]['geometry'];d=float(distances[list(inds).index(idx)])
        along=line.project(pt);q=line.interpolate(along);lon,lat=INVERSE.transform(q.x,q.y)
        rows.append(dict(**p,edge_id=eid,offset_m=d,along_m=along,snap_longitude=lon,snap_latitude=lat,nearest_ties=len(inds),attached_100m=d<=100,attached_500m=d<=500))
    return rows

def multisource(graph,records,attached,selector):
    distance=np.full(len(graph['nodes']),math.inf);queue=[];positions=defaultdict(list)
    for p in attached:
        if not p['attached_500m'] or not selector(p):continue
        eid=p['edge_id'];pos=p['along_m'];positions[eid].append(pos);u,v=graph['pairs'][eid];length=records[eid]['geometry'].length
        for node,d in [(u,pos),(v,length-pos)]:
            if d<distance[node]:distance[node]=d;heapq.heappush(queue,(d,int(node)))
    while queue:
        d,u=heapq.heappop(queue)
        if d!=distance[u]:continue
        for v,w in graph['adjacency'][u]:
            nd=d+w
            if nd<distance[v]:distance[v]=nd;heapq.heappush(queue,(nd,v))
    result={}
    for p in attached:
        if p['dataset']!='population' or not p['attached_500m']:continue
        eid=p['edge_id'];u,v=graph['pairs'][eid];pos=p['along_m'];length=records[eid]['geometry'].length
        d=min(distance[u]+pos,distance[v]+length-pos)
        if positions[eid]:d=min(d,min(abs(pos-q) for q in positions[eid]))
        result[p['point_id']]=float(d) if math.isfinite(d) else None
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);PROC.mkdir(parents=True,exist_ok=True)
    records=roads();mapping=seam_repairs(records);variants=[];graphs={}
    for layered in (True,False):
        for repaired in (False,True):
            g=graph(records,layered,repaired,mapping);graphs[g['report']['variant']]=g;variants.append(g['report'])
    print('Graph variants constructed',flush=True)
    crossing=intersections(records);print('Intersection screening',crossing,flush=True)
    points=population()+pois();attached=attach(points,records)
    summary=[]
    for dataset in sorted(set(p['dataset'] for p in attached)):
        subset=[p for p in attached if p['dataset']==dataset];ds=np.array([p['offset_m'] for p in subset])
        summary.append(dict(dataset=dataset,points=len(subset),within_100m=int((ds<=100).sum()),within_500m=int((ds<=500).sum()),median_offset_m=float(np.median(ds)),p95_offset_m=float(np.quantile(ds,.95)),max_offset_m=float(ds.max())))
    csvout(OUT/'attachment_summary.csv',summary)
    selectors={'medical':lambda p:p['dataset']=='medical' and p['category'] in ('1','2'), 'ems_base':lambda p:p['dataset']=='fire_stations' and p['category'] in ('2','3')}
    routes=[]
    for variant,g in graphs.items():
        for target,selector in selectors.items():
            values=multisource(g,records,attached,selector)
            for p in attached:
                if p['dataset']=='population':routes.append(dict(mesh_code=p['mesh_code'],variant=variant,target=target,attached_500m=p['attached_500m'],network_distance_m=values.get(p['point_id']),population_raw=p['population_raw'],suppression_code=p['suppression_code']))
    csvout(OUT/'mesh_route_diagnostics.csv',routes)
    csvout(OUT/'network_variant_summary.csv',variants)
    csvout(PROC/'point_attachments.csv',attached)
    chosen=graphs['layer_aware_submm_seam']
    csvout(PROC/'nodes_layer_aware.csv',[dict(node_id=i,easting=x,northing=y,level=l,component=int(chosen['labels'][i])) for i,(x,y,l) in enumerate(chosen['nodes'])])
    feature_list=[]
    for e,(u,v) in zip(records,chosen['pairs']):
        props={k:v for k,v in e.items() if k not in ('geometry','lonlat')};props.update(from_node=int(u),to_node=int(v),length_m=e['geometry'].length,component=int(chosen['edge_comp'][e['edge_id']]))
        feature_list.append(dict(type='Feature',properties=props,geometry=dict(type='LineString',coordinates=list(map(list,e['geometry'].coords)))))
    (PROC/'edges_layer_aware.geojson').write_text(json.dumps(dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='urn:ogc:def:crs:EPSG::6670')),features=feature_list)))
    # Diagnostic map, not a manuscript figure.
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    segs=[np.array(e['geometry'].coords)/1000 for e in records]
    for ax,key in zip(axes,['layer_aware_submm_seam','shared_endpoint_submm_seam']):
        g=graphs[key];largest=int(g['component_length'].argmax());colors=['#39769c' if c==largest else '#c89660' for c in g['edge_comp']]
        ax.add_collection(LineCollection(segs,colors=colors,linewidths=.25));ax.autoscale();ax.set_aspect('equal');ax.set_title(key.replace('_',' ')+'\n'+str(g['report']['components'])+' components');ax.set_xlabel('Easting (km)');ax.set_ylabel('Northing (km)')
    fig.suptitle('Vehicle network pilot: largest component (blue), others (orange)');fig.tight_layout();fig.savefig(OUT/'network_components.png',dpi=160);plt.close(fig)
    integrity=[]
    for path in sorted(SOURCE_FILES):
        integrity.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
    (OUT/'source_hashes.json').write_text(json.dumps(integrity,indent=2)+'\n')
    result=dict(bbox=BBOX,projected_crs=PROJECT,stats=dict(stats),network_variants=variants,intersection_candidates=crossing,attachment_summary=summary,source_files_read=len(SOURCE_FILES),rules=dict(vehicle_centerline_type='1 only; provisional',attachment_eligible='level 0, normal road state, non-expressway',attachment_thresholds_m=[100,500],seam_merge_threshold_m=.001,all_edges_bidirectional=True,interior_crossings_split=False,speeds_assigned=False))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
