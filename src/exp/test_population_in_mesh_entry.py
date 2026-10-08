"""Test in-mesh eligible-road proxies for large-offset census mesh centers.

The proxy is a road representative, never an observed residential location.
Search all eligible source lines in relevant tiles. No source records are altered.
"""
from concurrent.futures import ProcessPoolExecutor
import io,json,zipfile
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point,box,shape
from shapely.ops import nearest_points
from pyproj import Transformer
from build_vehicle_network_pilot import ROOT,csvout
from assemble_national_network_quality import PROC as BASE,OUT as QC

OUT=ROOT/'data/exp/national-population-entry'


def work(tile):
    rows=pd.read_csv(OUT/'flagged_meshes.csv',dtype={'point_id':str},keep_default_na=False,float_precision='round_trip')
    # 1 km census meshes nest within the first-mesh road tile; include a halo
    # for source-line overflow at archive boundaries.
    x0=100+int(tile[2:]);y0=int(tile[:2])/1.5
    rows=rows[(rows.longitude>=x0-.01)&(rows.longitude<=x0+1.01)&(rows.latitude>=y0-.01)&(rows.latitude<=y0+2/3+.01)]
    if rows.empty:return []
    boxes=[box(p.longitude-1/160,p.latitude-1/240,p.longitude+1/160,p.latitude+1/240) for p in rows.itertuples()]
    tree=shapely.STRtree(boxes);info=json.loads((BASE/'tiles'/(tile+'.json')).read_text())
    tr=Transformer.from_crs(6668,info['local_crs'],always_xy=True);inv=Transformer.from_crs(tr.target_crs,6668,always_xy=True)
    polygons=[shapely.transform(poly,tr.transform,interleaved=False).buffer(-.5) for poly in boxes]
    centers=[Point(*tr.transform(p.longitude,p.latitude)) for p in rows.itertuples()]
    m=np.load(BASE/'tiles'/(tile+'.npz'))
    eligible=set(m['fids'][(m['levels']==0)&(m['states']==1)&(m['classes']!=4)].tolist())
    best={};ordinal=0
    with zipfile.ZipFile(ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip') as z:
        name=next(v for v in z.namelist() if v.endswith('.geojson'))
        with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as stream:
            for text in stream:
                if not text.lstrip().startswith('{ "type": "Feature",'):continue
                ordinal+=1
                if ordinal not in eligible:continue
                f=json.loads(text.strip().rstrip(','));line=shape(f['geometry'])
                indices=tree.query(line,predicate='intersects')
                if not len(indices):continue
                projected=shapely.transform(line,tr.transform,interleaved=False)
                for i in indices:
                    clip=projected.intersection(polygons[i])
                    if clip.is_empty or clip.length<1:continue
                    q=nearest_points(centers[i],clip)[1];x,y=inv.transform(q.x,q.y)
                    if not boxes[i].covers(Point(x,y)):continue
                    distance=q.distance(centers[i]);p=rows.iloc[i];source=f'{tile}:{ordinal}'
                    if p.point_id in best and (distance,source)>=(best[p.point_id]['proxy_center_offset_m'],best[p.point_id]['proxy_road_source_id']):continue
                    edge=int(np.searchsorted(m['fids'],ordinal));assert m['fids'][edge]==ordinal
                    along=float(projected.project(q));along=max(0.,min(float(m['lengths'][edge]),along))
                    assert projected.interpolate(along).distance(q)<1e-6
                    best[p.point_id]=dict(point_id=p.point_id,proxy_longitude=x,proxy_latitude=y,proxy_center_offset_m=distance,
                                         proxy_road_source_id=source,proxy_tile_edge_id=edge,proxy_along_m=along,
                                         source_segment_length_m=float(m['lengths'][edge]),source_tile=tile,
                                         proxy_within_own_mesh=True,proxy_is_observed_residential_location=False)
    assert ordinal==info['source_features']
    return list(best.values())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    columns=['point_id','dataset','longitude','latitude','population_raw','edge_id','along_m','offset_m','attached_500m','road_source_id']
    points=pd.read_csv(QC/'point_attachment_diagnostics.csv',usecols=columns,dtype={'point_id':str,'population_raw':str,'road_source_id':str},keep_default_na=False,low_memory=False,float_precision='round_trip')
    flagged=points[points.dataset.eq('population')&~points.attached_500m].copy();assert len(flagged)==1722
    flagged.to_csv(OUT/'flagged_meshes.csv',index=False)
    tiles=[]
    for p in (BASE/'tiles').glob('*.json'):
        tile=p.stem
        if not tile.isdigit():continue
        x0=100+int(tile[2:]);y0=int(tile[:2])/1.5
        if ((flagged.longitude>=x0-.01)&(flagged.longitude<=x0+1.01)&(flagged.latitude>=y0-.01)&(flagged.latitude<=y0+2/3+.01)).any():tiles.append(tile)
    options=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(work,sorted(tiles)),1):
            options.extend(result)
            if i%10==0:print(f'{i}/{len(tiles)} tiles; {len(options)} local proxy candidates',flush=True)
    best={}
    for r in options:
        if r['point_id'] not in best or (r['proxy_center_offset_m'],r['proxy_road_source_id'])<(best[r['point_id']]['proxy_center_offset_m'],best[r['point_id']]['proxy_road_source_id']):best[r['point_id']]=r
    rows=[]
    for p in flagged.itertuples():
        candidate=best.get(p.point_id)
        base=dict(point_id=p.point_id,longitude=p.longitude,latitude=p.latitude,population_raw=p.population_raw,
                  baseline_offset_m=p.offset_m,baseline_road_source_id=p.road_source_id,
                  result='in_mesh_road_proxy_available' if candidate else 'no_in_mesh_eligible_line_found',
                  original_record_retained=True)
        fields=['proxy_longitude','proxy_latitude','proxy_center_offset_m','proxy_road_source_id','proxy_tile_edge_id','proxy_along_m','source_segment_length_m','source_tile','proxy_within_own_mesh','proxy_is_observed_residential_location']
        base.update({key:candidate.get(key,'') if candidate else '' for key in fields});rows.append(base)
    csvout(OUT/'population_entry_candidates.csv',rows)
    summary=dict(flagged_meshes=len(flagged),source_tiles_scanned=len(tiles),proxies_available=len(best),unresolved_meshes=len(flagged)-len(best),
                 proxy_definition='Closest eligible source-road point inside the same mesh, using a conservative 0.5 m interior margin and at least 1 m of clipped line; no inferred residence or off-road traversable connector',
                 scope='Only the 1722 centers failing the existing 500 m threshold; all other population and all POI attachments held fixed',raw_data_modified=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
