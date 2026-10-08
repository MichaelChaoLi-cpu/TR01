"""Read-only source-road audit of the 200 unresolved in-mesh entries."""
from concurrent.futures import ProcessPoolExecutor
import io,json,zipfile
from collections import Counter
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box,shape
from pyproj import Transformer
from build_vehicle_network_pilot import ROOT,csvout
from assemble_national_network_quality import PROC as BASE
from test_population_in_mesh_entry import OUT as ENTRY
OUT=ROOT/'data/exp/data-briefing/population-sample-audit'


def work(tile):
    rows=pd.read_csv(OUT/'pending_meshes_input.csv',dtype={'point_id':str},float_precision='round_trip')
    x0=100+int(tile[2:]);y0=int(tile[:2])/1.5
    rows=rows[(rows.longitude>=x0-.01)&(rows.longitude<=x0+1.01)&(rows.latitude>=y0-.01)&(rows.latitude<=y0+2/3+.01)]
    if rows.empty:return [],{}
    boxes=[box(r.longitude-1/160,r.latitude-1/240,r.longitude+1/160,r.latitude+1/240) for r in rows.itertuples()]
    tree=shapely.STRtree(boxes);tr=Transformer.from_crs(6668,f'+proj=aeqd +lat_0={y0+1/3} +lon_0={x0+.5} +datum=WGS84 +units=m',always_xy=True)
    polygons=[shapely.transform(b,tr.transform,interleaved=False) for b in boxes];guard=[b.buffer(-.5) for b in polygons]
    evidence=[];ordinal=0
    with zipfile.ZipFile(ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip') as z:
        name=next(n for n in z.namelist() if n.endswith('.geojson'))
        with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as stream:
            for text in stream:
                if not text.lstrip().startswith('{ "type": "Feature",'):continue
                ordinal+=1;f=json.loads(text.strip().rstrip(','));line=shape(f['geometry']);indices=tree.query(line,predicate='intersects')
                if not len(indices):continue
                p=f['properties'];normal=p['N13_002']=='1';eligible=normal and str(p['N13_005'])=='0' and p['N13_004']=='1' and p['N13_003']!='4'
                proj=shapely.transform(line,tr.transform,interleaved=False)
                for i in indices:
                    length=proj.intersection(polygons[i]).length;inside=proj.intersection(guard[i]).length
                    evidence.append(dict(point_id=rows.iloc[i].point_id,source_id=f'{tile}:{ordinal}',road_type=p['N13_002'],road_class=p['N13_003'],road_state=p['N13_004'],level=p['N13_005'],length_in_mesh_m=length,length_inside_guard_m=inside,existing_entry_attributes_eligible=eligible,normal_centerline=normal))
    expected=json.loads((BASE/'tiles'/f'{tile}.json').read_text())['source_features'];assert ordinal==expected
    return evidence,dict(tile=tile,features_scanned=ordinal,intersecting_records=len(evidence))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv(ENTRY/'population_entry_candidates.csv',dtype={'point_id':str,'population_raw':str},keep_default_na=False,float_precision='round_trip')
    pending=rows[rows.result.eq('no_in_mesh_eligible_line_found')].copy();assert len(pending)==200
    pending.to_csv(OUT/'pending_meshes_input.csv',index=False)
    tiles=[]
    for path in (BASE/'tiles').glob('*.json'):
        t=path.stem
        if not t.isdigit():continue
        x0=100+int(t[2:]);y0=int(t[:2])/1.5
        if ((pending.longitude>=x0-.01)&(pending.longitude<=x0+1.01)&(pending.latitude>=y0-.01)&(pending.latitude<=y0+2/3+.01)).any():tiles.append(t)
    evidence=[];scans=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,(records,scan) in enumerate(pool.map(work,sorted(tiles)),1):
            evidence.extend(records);scans.append(scan);print(f'{i}/{len(tiles)} source tiles scanned',flush=True)
    cols=['point_id','source_id','road_type','road_class','road_state','level','length_in_mesh_m','length_inside_guard_m','existing_entry_attributes_eligible','normal_centerline']
    df=pd.DataFrame(evidence,columns=cols);df.to_csv(OUT/'pending_mesh_road_evidence.csv',index=False)
    answers=[]
    for p in pending.itertuples():
        q=df[df.point_id.eq(p.point_id)];positive=q[q.length_in_mesh_m.ge(1)]
        strict=q[q.existing_entry_attributes_eligible&q.length_inside_guard_m.ge(1)]
        assert strict.empty,('previous search contradiction',p.point_id)
        if (positive.existing_entry_attributes_eligible).any():reason='eligible_line_only_without_interior_guard'
        elif positive.normal_centerline.any():reason='normal_centerline_present_but_entry_attributes_excluded'
        elif len(positive):reason='only_nonordinary_centerline_present'
        elif len(q):reason='only_boundary_touch_or_submetre_line_present'
        else:reason='no_intersecting_N13_source_line_found'
        answers.append(dict(point_id=p.point_id,mesh_code=p.point_id.split(':')[1],longitude=p.longitude,latitude=p.latitude,population_raw=p.population_raw,published_local_population=int(p.population_raw),baseline_offset_m=p.baseline_offset_m,reason=reason,source_intersection_records=len(q),positive_length_records=len(positive),source_ids=';'.join(q.source_id),primary_road_tile_downloaded=p.point_id.split(':')[1][:4] in {x.stem for x in (BASE/'tiles').glob('*.json')},interpretation='Source/entry diagnostic only; island status and actual residence not established; retain original mesh'))
    csvout(OUT/'pending_mesh_audit.csv',answers);csvout(OUT/'pending_tile_scan.csv',scans)
    groups=pd.DataFrame(answers).groupby('reason',as_index=False).agg(meshes=('point_id','size'),population=('published_local_population','sum'))
    groups.to_csv(OUT/'pending_reason_summary.csv',index=False)
    result=dict(status='passed',pending_meshes=200,pending_population=sum(r['published_local_population'] for r in answers),source_tiles_scanned=len(scans),source_features_scanned=sum(r['features_scanned'] for r in scans),classification=groups.to_dict('records'),original_records_retained=True,island_or_land_barrier_claims_not_inferred=True)
    (OUT/'pending_entry_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
