"""Build local bridge-to-N13 candidate evidence without assigning assets to edges."""
import csv,hashlib,io,json,math,zipfile
from pathlib import Path
from collections import Counter
from pyproj import Transformer
from shapely import LineString,Point,STRtree,box
from check_national_roads import features
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/feasibility-check/bridge-crosswalk-review'
SOURCE=ROOT/'data/exp/feasibility-check/bridge-access-rural-sample/sources/bridge_nikko_latest.json'
def csvout(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assets=json.loads(SOURCE.read_text())['result'];assert len(assets)==10
    proj=Transformer.from_crs(6668,6677,always_xy=True).transform
    points=[Point(*proj(r['syogen']['ichi']['keido'],r['syogen']['ichi']['ido'])) for r in assets]
    # Fixed 200-m buffers define candidate evidence, not asset-matching acceptance.
    road_path=ROOT/'data/raw/roads/n13_2024_geojson/N13-24_5439_GEOJSON.zip'
    road_hash=hashlib.sha256(road_path.read_bytes()).hexdigest()
    boxes=[box(r['syogen']['ichi']['keido']-.003,r['syogen']['ichi']['ido']-.003,r['syogen']['ichi']['keido']+.003,r['syogen']['ichi']['ido']+.003) for r in assets]
    bounds=(min(b.bounds[0] for b in boxes),min(b.bounds[1] for b in boxes),max(b.bounds[2] for b in boxes),max(b.bounds[3] for b in boxes))
    roads=[];scanned=0
    with zipfile.ZipFile(road_path) as z:
        name=next(n for n in z.namelist() if n.endswith('.geojson'))
        with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as f:
            for i,feat in enumerate(features(f,{}),1):
                scanned+=1;p=feat['properties'];c=feat['geometry']['coordinates']
                if str(p['N13_002'])!='1' or str(p['N13_004']) in ['5','6','7']:continue
                xs=[x[0] for x in c];ys=[x[1] for x in c]
                if max(xs)<bounds[0] or min(xs)>bounds[2] or max(ys)<bounds[1] or min(ys)>bounds[3]:continue
                if not any(LineString(c).intersects(b) for b in boxes):continue
                g=LineString([proj(*coord) for coord in c])
                if g.length>0:roads.append({'source_id':f'5439:{i}','coordinates':c,'properties':p,'geometry':g})
    tree=STRtree([r['geometry'] for r in roads]);cases=[];candidates=[]
    for i,(asset,point) in enumerate(zip(assets,points),1):
        review_id=f'B{i:02}';syogen=asset['syogen'];length=float(syogen['kyouchou'])
        matches=[]
        for j in tree.query(point.buffer(200)):
            road=roads[int(j)];g=road['geometry'];distance=g.distance(point)
            if distance>200:continue
            position=g.project(point);endpoint_distance=min(point.distance(Point(g.coords[0])),point.distance(Point(g.coords[-1])))
            row={'review_id':review_id,'facility_id':asset['shisetsu_id'],'source_id':road['source_id'],'point_distance_m':distance,'source_feature_length_m':g.length,'asset_length_m':length,'feature_to_asset_length_ratio':g.length/length,'projected_position_m':position,'distance_to_nearest_feature_endpoint_m':endpoint_distance,'source_class':road['properties']['N13_003'],'source_state':road['properties']['N13_004'],'source_level':road['properties']['N13_005'],'source_width_class':road['properties']['N13_006'],'accepted_match':False}
            candidates.append(row);matches.append((row,road))
        matches.sort(key=lambda x:x[0]['point_distance_m']);assert matches
        nearest=matches[0][0]
        structures=[m[0] for m in matches if str(m[0]['source_state'])=='2']
        cases.append({'review_id':review_id,'facility_id':asset['shisetsu_id'],'bridge_name':syogen['shisetsu']['meisyou'],'route_name':syogen['rosen']['meisyou'],'latitude':syogen['ichi']['ido'],'longitude':syogen['ichi']['keido'],'asset_length_m':length,'asset_width_m':syogen['fukuin'],'inspection_year':asset['tenken']['nendo'],'health_code':asset['tenken']['kiroku']['hantei_kubun'],'nearest_source_id':nearest['source_id'],'nearest_distance_m':nearest['point_distance_m'],'nearest_feature_length_m':nearest['source_feature_length_m'],'nearest_state':nearest['source_state'],'nearest_class':nearest['source_class'],'nearest_feature_to_asset_ratio':nearest['feature_to_asset_length_ratio'],'bridge_state_candidates_within_50m':sum(r['point_distance_m']<=50 for r in structures),'candidate_geometries':[{'source_id':r['source_id'],'distance_m':r['point_distance_m'],'properties':road['properties'],'coordinates':road['coordinates']} for r,road in matches]})
    counts=Counter(c['nearest_source_id'] for c in cases)
    for case in cases:case['assets_sharing_nearest_feature']=counts[case['nearest_source_id']]
    csvout('candidate_attributes.csv',candidates)
    csvout('case_attributes.csv',[{k:v for k,v in c.items() if k!='candidate_geometries'} for c in cases])
    (OUT/'review_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    assert hashlib.sha256(road_path.read_bytes()).hexdigest()==road_hash
    validation={'status':'passed','asset_records':10,'candidate_rows':len(candidates),'n13_scanned_features':scanned,'n13_local_features':len(roads),'shared_nearest_features':{k:v for k,v in counts.items() if v>1},'raw_n13_sha256':road_hash,'raw_unchanged':True,'asset_source_sha256':source_hash,'accepted_assignments':0}
    (OUT/'preparation_validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
