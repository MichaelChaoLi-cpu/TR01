"""Build the exhaustive pilot topology review queue, without changing the graph."""
from pathlib import Path
from collections import defaultdict
import csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tr01-matplotlib')
import pandas as pd
from shapely import STRtree
from shapely.geometry import shape
from pyproj import Transformer
from build_vehicle_network_pilot import ROOT,OUT as PILOT,PROC,csvout
OUT=ROOT/'data/exp/network-map-review'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    features=json.loads((PROC/'edges_layer_aware.geojson').read_text())['features']
    geoms=[shape(f['geometry']) for f in features];props=[f['properties'] for f in features]
    bysource=defaultdict(list)
    for i,p in enumerate(props):bysource[p['source_id']].append(i)
    rows=[]
    grades=pd.read_csv(PILOT/'grade_transition_candidates.csv')
    for i,p in grades.iterrows():
        ids=p.source_ids.split(';');edges=[j for sid in ids for j in bysource[sid]]
        rows.append(dict(review_id=f'G{i+1:03}',kind='multilevel_endpoint',longitude=p.longitude,latitude=p.latitude,source_ids=p.source_ids,levels=p.levels,road_states=p.road_states,endpoint_incidence=p.endpoint_incidence,candidate_class=p.candidate_class,edge_ids=';'.join(map(str,edges)),point_index=1,pair_id='',map_url=f'https://maps.gsi.go.jp/#18/{p.latitude:.8f}/{p.longitude:.8f}/&base=std&ls=std&disp=1'))
    inverse=Transformer.from_crs(6670,6668,always_xy=True);tree=STRtree(geoms)
    pairs=tree.query(geoms,predicate='intersects').T;counts=defaultdict(int);n=0
    for a,b in pairs:
        if a>=b:continue
        ga,gb=geoms[a],geoms[b]
        if set(ga.coords[i] for i in (0,-1))&set(gb.coords[i] for i in (0,-1)):continue
        same=props[a]['level']==props[b]['level'];cross=ga.crosses(gb)
        typ='interior_crossing' if cross else 'non_endpoint_touch_or_overlap'
        counts[typ+('_same_level' if same else '_different_level')]+=1
        if not same:continue
        inter=ga.intersection(gb)
        points=list(inter.geoms) if inter.geom_type=='MultiPoint' else [inter] if inter.geom_type=='Point' else [inter.representative_point()]
        n+=1
        for j,pt in enumerate(points,1):
            lon,lat=inverse.transform(pt.x,pt.y)
            rows.append(dict(review_id=f'C{n:03}-{j}',kind=typ,longitude=lon,latitude=lat,source_ids=props[a]['source_id']+';'+props[b]['source_id'],levels=str(props[a]['level']),road_states=props[a]['road_state']+';'+props[b]['road_state'],endpoint_incidence='',candidate_class='same_level_geometry_candidate',edge_ids=f'{a};{b}',point_index=j,pair_id=f'C{n:03}',map_url=f'https://maps.gsi.go.jp/#18/{lat:.8f}/{lon:.8f}/&base=std&ls=std&disp=1'))
    csvout(OUT/'review_queue.csv',rows)
    expected=json.loads((PILOT/'summary.json').read_text())['intersection_candidates']
    assert dict(counts)==expected
    assert len(grades)==151 and counts['interior_crossing_same_level']==204
    result=dict(multilevel_endpoints=len(grades),same_level_crossing_pairs=counts['interior_crossing_same_level'],same_level_touch_overlap_pairs=counts['non_endpoint_touch_or_overlap_same_level'],review_locations=len(rows),intersection_locations=len(rows)-len(grades),notes='Previous 204 count refers to intersecting line pairs, not unique crossing coordinates. MultiPoint intersections now get separate locations; original 200-example cap has been removed for this review queue.')
    (OUT/'queue_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
