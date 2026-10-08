"""Nationwide bounded-memory vehicle-centerline and point-attachment screening.

Exact endpoint topology is retained. AEQD per tile supplies diagnostic metric
lengths and nearest-line distances; candidate gaps are never repaired.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import csv,hashlib,io,json,math,zipfile
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Point
from pyproj import Transformer
from scipy.spatial import cKDTree
from build_vehicle_network_pilot import ROOT,csvout

OUT=ROOT/'data/exp/national-network-quality'
CACHE=ROOT/'data/processed/national-network-quality/tiles'
DT=np.dtype([('x','<f8'),('y','<f8')])


def band(x,y):
    if y>=41:return '01_north'
    if y>=37:return '02_northeast'
    if x>=138:return '03_east'
    if x>=135:return '04_central'
    if x>=132 and y>=34:return '05_west_north'
    if x>=132:return '06_west_south'
    if y>=30:return '07_southwest'
    return '08_southern_islands'


def work(tile):
    path=ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip'
    done=CACHE/(tile+'.json')
    if done.exists():return json.loads(done.read_text())
    xy=np.load(OUT/'point_coordinates.npy',mmap_mode='r')
    x0=100+int(tile[2:]);y0=int(tile[:2])/1.5
    tr=Transformer.from_crs(6668,f'+proj=aeqd +lat_0={y0+1/3} +lon_0={x0+.5} +datum=WGS84 +units=m',always_xy=True)
    inv=Transformer.from_crs(tr.target_crs,6668,always_xy=True)
    coords=[];offsets=[0];meta=[];total=0;excluded=Counter()
    with zipfile.ZipFile(path) as z:
        name=next(n for n in z.namelist() if n.endswith('.geojson'))
        with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.lstrip().startswith('{ "type": "Feature",'):continue
                f=json.loads(line.strip().rstrip(','));total+=1;p=f['properties'];c=f['geometry']['coordinates']
                if p['N13_002']!='1':excluded['nonordinary_centerline']+=1;continue
                if str(p['N13_004']) in ('5','6','7'):excluded['unusable_source_state']+=1;continue
                if all(pt==c[0] for pt in c):excluded['zero_length']+=1;continue
                coords.extend(c);offsets.append(len(coords))
                meta.append((total,int(p['N13_005']),int(p['N13_004']),int(p['N13_003']),int(p['N13_006'])))
    counts={r['archive']:int(r['features']) for r in csv.DictReader((ROOT/'data/exp/road-quality-check/archive_summary.csv').open())}
    assert total==counts[path.name],(tile,total,counts[path.name])
    if not meta:
        np.savez(CACHE/(tile+'.npz'),ends=np.empty((0,2,2)),fids=np.array([],dtype=int),levels=np.array([],dtype=np.int16),states=np.array([],dtype=np.uint8),classes=np.array([],dtype=np.uint8),widths=np.array([],dtype=np.uint8),lengths=np.array([]))
        np.savez(CACHE/(tile+'_attachments.npz'),point_indices=np.array([],dtype=int),edge_indices=np.array([],dtype=int),distances=np.array([]),along=np.array([]))
        for suffix,fields in [('grade',['kind','longitude','latitude','tile','incidence','source_ids','levels','road_states']),('crossings',['kind','longitude','latitude','tile','incidence','source_ids','levels','road_states']),('gaps',['kind','longitude','latitude','tile','source_ids','levels','road_states','other_longitude','other_latitude','distance_m'])]:csvout(CACHE/(tile+'_'+suffix+'.csv'),[],fields)
        (CACHE/(tile+'_samples.json')).write_text('[]')
        result=dict(tile=tile,source_features=total,vehicle_edges=0,eligible_attachment_edges=0,excluded=dict(excluded),multilevel_endpoint_sites=0,degree_one_endpoint_sites=0,local_nearest_other_endpoint_candidates=0,bridge_tunnel_lines_sampled_for_crossings=0,sampled_crossing_records=0,local_crs=tr.target_crs.to_string(),validation='passed')
        done.write_text(json.dumps(result,indent=2)+'\n');return result
    a=np.array(coords,dtype=np.float64);del coords
    assert np.isfinite(a).all()
    offsets=np.array(offsets,dtype=np.int64);m=np.array(meta,dtype=np.int64);del meta
    lines=shapely.from_ragged_array(shapely.GeometryType.LINESTRING,a,(offsets,))
    ends=np.stack((a[offsets[:-1]],a[offsets[1:]-1]),axis=1)
    projected=shapely.transform(lines,tr.transform,interleaved=False)
    lengths=shapely.length(projected)
    assert np.all(lengths>0)
    np.savez(CACHE/(tile+'.npz'),ends=ends,fids=m[:,0],levels=m[:,1].astype(np.int16),states=m[:,2].astype(np.uint8),classes=m[:,3].astype(np.uint8),widths=m[:,4].astype(np.uint8),lengths=lengths)
    eligible=np.flatnonzero((m[:,1]==0)&(m[:,2]==1)&(m[:,3]!=4))
    pi=np.flatnonzero((xy[:,0]>=x0-.1)&(xy[:,0]<=x0+1.1)&(xy[:,1]>=y0-.1)&(xy[:,1]<=y0+2/3+.1))
    if len(pi) and len(eligible):
        pts=shapely.points(*tr.transform(xy[pi,0],xy[pi,1]))
        ix,dist=shapely.STRtree(projected[eligible]).query_nearest(pts,all_matches=True,return_distance=True,max_distance=5000)
        # Deterministic lowest source feature ordinal among equal-distance ties.
        order=np.lexsort((m[eligible[ix[1]],0],dist,ix[0]));ix=ix[:,order];dist=dist[order]
        keep=np.r_[True,ix[0,1:]!=ix[0,:-1]] if ix.shape[1] else np.array([],dtype=bool)
        ii=ix[0,keep];ee=eligible[ix[1,keep]];dd=dist[keep]
        along=shapely.line_locate_point(projected[ee],pts[ii])
        np.savez(CACHE/(tile+'_attachments.npz'),point_indices=pi[ii],edge_indices=ee,distances=dd,along=along)
    else:np.savez(CACHE/(tile+'_attachments.npz'),point_indices=np.array([],dtype=int),edge_indices=np.array([],dtype=int),distances=np.array([]),along=np.array([]))
    flat=ends.reshape(-1,2);keys=np.empty(len(flat),dtype=DT);keys['x']=flat[:,0];keys['y']=flat[:,1]
    unique,inverse=np.unique(keys,return_inverse=True);inc=np.bincount(inverse)
    lo=np.full(len(unique),32767,dtype=np.int16);hi=np.full(len(unique),-32768,dtype=np.int16)
    lev=np.repeat(m[:,1],2).astype(np.int16);np.minimum.at(lo,inverse,lev);np.maximum.at(hi,inverse,lev)
    order=np.argsort(inverse,kind='stable');start=np.r_[0,np.cumsum(inc)];grade=[];sample=[]
    for j in np.flatnonzero(lo!=hi):
        ei=np.unique(order[start[j]:start[j+1]]//2);x=float(unique[j]['x']);y=float(unique[j]['y'])
        grade.append(dict(kind='multilevel_endpoint',longitude=x,latitude=y,tile=tile,incidence=int(inc[j]),source_ids=';'.join(f'{tile}:{m[i,0]}' for i in ei),levels=';'.join(map(str,sorted(set(m[ei,1])))),road_states=';'.join(map(str,sorted(set(m[ei,2]))))))
    csvout(CACHE/(tile+'_grade.csv'),grade,['kind','longitude','latitude','tile','incidence','source_ids','levels','road_states'])
    # Sample endpoint transitions separately for bridge and tunnel context.
    for state,kind in [(2,'bridge_endpoint'),(3,'tunnel_endpoint')]:
        available=[r for r in grade if str(state) in r['road_states'].split(';')]
        selected=sorted(available,key=lambda r:hashlib.sha256(r['source_ids'].encode()).digest())[:2]
        for r in selected:
            ids=[int(v.split(':')[1]) for v in r['source_ids'].split(';')]
            ei=[int(np.flatnonzero(m[:,0]==fid)[0]) for fid in ids]
            sample.append(dict(r,kind=kind,band=band(r['longitude'],r['latitude']),geometries=[shapely.get_coordinates(lines[i]).tolist() for i in ei]))
    # Inspect a deterministic subset of bridge/tunnel lines for interior crossings.
    tree=shapely.STRtree(projected);tested=0;crosses=[]
    for state,kind in [(2,'bridge_crossing'),(3,'tunnel_crossing')]:
        inds=np.flatnonzero(m[:,2]==state)
        selected=sorted(inds,key=lambda i:hashlib.sha256(f'{tile}:{m[i,0]}'.encode()).digest())[:100]
        found=[]
        for i in selected:
            tested+=1
            for j in tree.query(projected[i],predicate='intersects'):
                if j==i:continue
                if any(np.array_equal(u,v) for u in ends[i] for v in ends[j]):continue
                inter=shapely.intersection(projected[i],projected[j])
                if inter.geom_type not in ('Point','MultiPoint'):continue
                for point in (list(inter.geoms) if inter.geom_type=='MultiPoint' else [inter]):
                    x,y=inv.transform(point.x,point.y)
                    found.append(dict(kind=kind,longitude=x,latitude=y,tile=tile,incidence='',source_ids=f'{tile}:{m[i,0]};{tile}:{m[j,0]}',levels=f'{m[i,1]};{m[j,1]}',road_states=f'{m[i,2]};{m[j,2]}',band=band(x,y),geometries=[shapely.get_coordinates(lines[k]).tolist() for k in [i,j]]))
        crosses.extend(found)
        sample.extend(sorted(found,key=lambda r:hashlib.sha256((r['source_ids']+str(r['longitude'])).encode()).digest())[:2])
    csvout(CACHE/(tile+'_crossings.csv'),[{k:v for k,v in r.items() if k not in ('geometries','band')} for r in crosses],['kind','longitude','latitude','tile','incidence','source_ids','levels','road_states'])
    # Exhaustive closest-other-endpoint screen for local degree-one endpoints.
    xx,yy=tr.transform(unique['x'],unique['y']);ep=np.column_stack((xx,yy));degree1=np.flatnonzero(inc==1)
    gaps=[]
    if len(ep)>1 and len(degree1):
        ds,js=cKDTree(ep).query(ep[degree1],k=2,distance_upper_bound=5)
        seen=set()
        for j,dist,k in zip(degree1,ds[:,1],js[:,1]):
            if not math.isfinite(dist) or dist<=0:continue
            pair=tuple(sorted((int(j),int(k))))
            if pair in seen:continue
            seen.add(pair)
            i1=int(order[start[j]]//2);i2=int(order[start[k]]//2)
            if i1==i2:continue
            gaps.append(dict(kind='near_dangling_endpoint',longitude=float(unique[j]['x']),latitude=float(unique[j]['y']),tile=tile,source_ids=f'{tile}:{m[i1,0]};{tile}:{m[i2,0]}',levels=f'{lo[j]};{hi[j]};{lo[k]};{hi[k]}',road_states=f'{m[i1,2]};{m[i2,2]}',other_longitude=float(unique[k]['x']),other_latitude=float(unique[k]['y']),distance_m=float(dist)))
    csvout(CACHE/(tile+'_gaps.csv'),gaps,['kind','longitude','latitude','tile','source_ids','levels','road_states','other_longitude','other_latitude','distance_m'])
    for r in sorted(gaps,key=lambda r:hashlib.sha256(r['source_ids'].encode()).digest())[:2]:
        ei=[int(np.flatnonzero(m[:,0]==int(fid.split(':')[1]))[0]) for fid in r['source_ids'].split(';')]
        sample.append(dict(r,band=band(r['longitude'],r['latitude']),geometries=[shapely.get_coordinates(lines[i]).tolist() for i in ei]))
    (CACHE/(tile+'_samples.json')).write_text(json.dumps(sample))
    result=dict(tile=tile,source_features=total,vehicle_edges=len(lines),eligible_attachment_edges=len(eligible),excluded=dict(excluded),multilevel_endpoint_sites=len(grade),degree_one_endpoint_sites=len(degree1),local_nearest_other_endpoint_candidates=len(gaps),bridge_tunnel_lines_sampled_for_crossings=tested,sampled_crossing_records=len(crosses),local_crs=tr.target_crs.to_string(),validation='passed')
    done.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    tiles=[p.name.split('_')[1] for p in sorted((ROOT/'data/raw/roads/n13_2024_geojson').glob('*.zip'))]
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(work,tiles),1):
            results.append(r);print(f"{i}/160 {r['tile']} {r['vehicle_edges']:,} vehicle edges; {r['local_nearest_other_endpoint_candidates']} local gap candidates",flush=True)
    (OUT/'tile_screen_summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
