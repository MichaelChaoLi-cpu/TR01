"""Check public coordinate-bearing bridge records and mountain traffic geometry.
Local only; no accepted network assignment, no raw or AnaSOP writes.
"""
import csv,hashlib,io,json,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from pyproj import Transformer
from shapely import LineString,Point,STRtree,box
from shapely.geometry import shape
from shapely.ops import transform,unary_union
from check_national_roads import features
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/feasibility-check/bridge-access-rural-sample'
SRC=OUT/'sources'
def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
    sop=ROOT/'docs/AnaSOP.md';sop_hash=hashlib.sha256(sop.read_bytes()).hexdigest()
    manifest=json.loads((SRC/'manifest.json').read_text())
    for item in manifest:assert hashlib.sha256((SRC/item['file']).read_bytes()).hexdigest()==item['sha256']
    traffic=list(csv.reader((SRC/'traffic_tochigi_2021.csv').read_text(encoding='cp932').splitlines()))
    assert all(len(r)==159 for r in traffic)
    by_id={r[0].zfill(11):r for r in traffic[1:]};assert len(by_id)==len(traffic)-1
    proj=Transformer.from_crs(6668,6677,always_xy=True).transform
    map_proj=Transformer.from_crs(4326,6677,always_xy=True).transform
    lines=[]
    for file in sorted(SRC.glob('*.geojson')):
        obj=json.loads(file.read_text());assert all(f['properties']['R03cen']==file.stem for f in obj['features'])
        g=unary_union([shape(f['geometry']) for f in obj['features']]);lines.append((file.stem,transform(map_proj,g)))
    latest=json.loads((SRC/'bridge_nikko_latest.json').read_text());bridges=latest['result']
    assert not latest['resultset']['is_error'] and len(bridges)==10
    assert len({r['shisetsu_id'] for r in bridges})==10
    bridge_points=[Point(*proj(r['syogen']['ichi']['keido'],r['syogen']['ichi']['ido'])) for r in bridges]
    interest=unary_union([g.buffer(100) for _,g in lines]+[p.buffer(250) for p in bridge_points])
    # Coverage spans primary meshes 5439 and 5539. These windows are diagnostics.
    coords=[]
    for _,g in lines:coords.extend([g.bounds[:2],g.bounds[2:]])
    # Geographic bounding box derived from the projected interest, with margin.
    inv=Transformer.from_crs(6677,6668,always_xy=True).transform
    geographic_interest=transform(inv,interest);bounds=geographic_interest.bounds
    roads=[];ids=[];properties=[];scanned=0;raw_hashes={}
    for tile in ['5439','5539']:
        path=ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip'
        raw_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as z:
            name=next(n for n in z.namelist() if n.endswith('.geojson'))
            with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as f:
                for i,feat in enumerate(features(f,{}),1):
                    scanned+=1;p=feat['properties'];c=feat['geometry']['coordinates']
                    if str(p['N13_002'])!='1' or str(p['N13_004']) in ['5','6','7']:continue
                    xs=[v[0] for v in c];ys=[v[1] for v in c]
                    if max(xs)<bounds[0] or min(xs)>bounds[2] or max(ys)<bounds[1] or min(ys)>bounds[3]:continue
                    g=transform(proj,LineString(c))
                    if g.length<=0 or not interest.intersects(g):continue
                    roads.append(g);ids.append(f'{tile}:{i}');properties.append(p)
    tree=STRtree(roads)
    bridge_indices=[i for i,p in enumerate(properties) if str(p['N13_004'])=='2']
    structure_tree=STRtree([roads[i] for i in bridge_indices])
    br=[]
    for record,point in zip(bridges,bridge_points):
        syogen=record['syogen'];i=int(tree.nearest(point));j=bridge_indices[int(structure_tree.nearest(point))]
        candidates=[int(k) for k in tree.query(point.buffer(50)) if point.distance(roads[int(k)])<=50]
        br.append({'facility_id':record['shisetsu_id'],'bridge_name':syogen['shisetsu']['meisyou'],'route_name':syogen['rosen'].get('meisyou',''),'latitude':syogen['ichi']['ido'],'longitude':syogen['ichi']['keido'],'municipality_code':syogen['gyousei_kuiki']['shikuchouson_code'],'inspection_year':record['tenken']['nendo'],'health_code':record['tenken']['kiroku']['hantei_kubun'],'maintenance_action_code':record['tenken']['syuzen']['sochi_joukyou'],'nearest_n13_id':ids[i],'nearest_distance_m':point.distance(roads[i]),'nearest_source_state':properties[i]['N13_004'],'nearest_bridge_elevated_id':ids[j],'bridge_elevated_distance_m':point.distance(roads[j]),'ordinary_candidates_within_50m':len(candidates),'bridge_elevated_candidates_within_50m':sum(str(properties[k]['N13_004'])=='2' for k in candidates),'accepted_match':False})
    csvout('bridge_coordinate_candidates.csv',br)
    tr=[];point_rows=[]
    road_15m=unary_union([g.buffer(15) for g in roads])
    for key,g in lines:
        ds=[];ambiguous=0;pieces=list(g.geoms) if g.geom_type=='MultiLineString' else [g]
        for part,line in enumerate(pieces):
            for pos in np.linspace(0,line.length,max(2,int(np.ceil(line.length/25))+1)):
                point=line.interpolate(pos);i=int(tree.nearest(point));d=point.distance(roads[i]);ds.append(d)
                nearby=[int(k) for k in tree.query(point.buffer(max(15,d+5))) if point.distance(roads[int(k)])<=d+5]
                ambiguous+=len(nearby)>1
                point_rows.append({'survey_id':key,'part':part,'position_m':float(pos),'nearest_distance_m':d,'nearest_n13_id':ids[i],'nearest_source_class':properties[i]['N13_003'],'candidates_within_nearest_plus_5m':len(nearby),'accepted_match':False})
        source=by_id[key]
        tr.append({'survey_id':key,'municipality_code_5':source[18].zfill(5),'road_type':source[3],'route_name':source[5],'csv_length_km':source[23],'geometry_length_km':g.length/1000,'up_observation_code':source[34],'down_observation_code':source[46],'total_24h_vehicles_source':source[61],'point_count':len(ds),'distance_p50_m':float(np.median(ds)),'distance_p95_m':float(np.percentile(ds,95)),'distance_max_m':max(ds),'point_share_within_15m':sum(d<=15 for d in ds)/len(ds),'point_share_within_30m':sum(d<=30 for d in ds)/len(ds),'line_length_share_within_15m':g.intersection(road_15m).length/g.length,'ambiguity_share':ambiguous/len(ds),'accepted_match':False})
    csvout('mountain_traffic_candidates.csv',tr);csvout('mountain_traffic_point_candidates.csv',point_rows)
    probe_summaries=[]
    for name in ['bridge_api_city5','bridge_api_pref','bridge_api_area','bridge_nikko_latest']:
        o=json.loads((SRC/(name+'.json')).read_text());rs=o['resultset'];probe_summaries.append({'file':name+'.json','is_error':rs['is_error'],'query_count':rs['count'],'returned_rows':len(o['result']),'limit':rs['limit']})
    csvout('api_probe_results.csv',probe_summaries)
    ds=[r['nearest_distance_m'] for r in point_rows]
    summary={'traffic_csv_rows':len(traffic)-1,'traffic_csv_columns':159,'traffic_sections':len(lines),'traffic_road_type_counts':dict(Counter(r['road_type'] for r in tr)),'traffic_sample_points':len(ds),'traffic_distance_p50_m':float(np.median(ds)),'traffic_distance_p95_m':float(np.percentile(ds,95)),'traffic_share_within_15m':sum(d<=15 for d in ds)/len(ds),'traffic_ambiguous_point_share':sum(r['candidates_within_nearest_plus_5m']>1 for r in point_rows)/len(ds),'bridge_rows':len(br),'bridge_unique_ids':len({r['facility_id'] for r in br}),'bridge_inspection_year_counts':dict(Counter(r['inspection_year'] for r in br)),'bridge_points_with_ordinary_within_50m':sum(r['nearest_distance_m']<=50 for r in br),'bridge_points_with_bridge_elevated_within_50m':sum(r['bridge_elevated_distance_m']<=50 for r in br),'bridge_points_without_bridge_elevated_within_50m':[r['facility_id'] for r in br if r['bridge_elevated_distance_m']>50],'n13_scanned_features':scanned,'n13_local_candidate_features':len(roads),'accepted_traffic_assignments':0,'accepted_bridge_assignments':0,'api_probes':probe_summaries}
    for file,h in raw_hashes.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h
    assert hashlib.sha256(sop.read_bytes()).hexdigest()==sop_hash
    validation={'status':'passed','download_hashes_verified':len(manifest),'traffic_map_ids_verified':len(lines),'latest_bridge_ids_unique':True,'municipality_codes_5_and_6_distinguished':True,'raw_hashes_unchanged':True,'anasop_unchanged':True,'anasop_sha256':sop_hash,'raw_sha256':raw_hashes,'national_acquisition_performed':False,'accepted_policy_input_assignments':0}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
