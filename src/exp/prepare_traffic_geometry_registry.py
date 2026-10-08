"""Source-preserving geometry registry, grouped by measured observation unit.

Final mode requires the complete national acquisition pass. Snapshot mode is
explicitly diagnostic and cannot supply an accepted national correspondence.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import numpy as np
import pandas as pd
import shapely

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/exp/national-traffic-geometry'
CLASS_MAP={1:4,2:4,3:1,4:2,5:3,6:2,7:3}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',action='store_true');args=parser.parse_args()
    data=(BASE/'manifest.csv').read_bytes()
    manifest=list(csv.DictReader(io.StringIO(data.decode())))
    assert all(len(r['Traffic Geometry Section'])==11 and r['Traffic Geometry Section'].isdigit() for r in manifest)
    digest=hashlib.sha256(data).hexdigest()
    receipt=json.loads((BASE/'validation.json').read_text())
    if not args.snapshot:assert receipt['acquisition_pass_complete'] and len(manifest)==99417
    out=BASE/('registry-snapshots'/Path(digest[:12]) if args.snapshot else Path('registry'))
    out.mkdir(parents=True,exist_ok=True)
    (out/'source_manifest.csv').write_bytes(data)
    status={r['Traffic Geometry Section']:r for r in manifest}
    traffic=pd.read_parquet(ROOT/'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet')
    selected=traffic[traffic['Traffic Measurement Eligible']].copy()
    selected['Target N13 Road Class']=selected['Traffic Road Class'].astype(int).map(CLASS_MAP)
    inventories=[];geometries=[];bad=[]
    for unit,group in selected.groupby('Traffic Observation Key',sort=True):
        shapes={};available=0
        for _,row in group.iterrows():
            section=row['Traffic Geometry Section'];record=status.get(section)
            if record is None or record['Status']!='available':continue
            path=ROOT/record['Path'];raw=path.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==record['SHA256']
            parsed=json.loads(raw)
            try:
                lines=[shapely.from_geojson(json.dumps(f['geometry'])) for f in parsed['features']]
                shape=shapely.union_all(lines)
                coords=shapely.get_coordinates(shape)
                assert len(coords)>1 and np.isfinite(coords).all() and not shapely.is_empty(shape)
                assert shapely.is_valid(shape) and float(shapely.length(shape))>0
                assert coords[:,0].min()>=122 and coords[:,0].max()<=154 and coords[:,1].min()>=20 and coords[:,1].max()<=46
            except Exception as error:
                bad.append({'Traffic Geometry Section':section,'Error':repr(error)})
                continue
            available+=1
            shapes.setdefault(int(row['Target N13 Road Class']),[]).append(shape)
        volumes=group['Observed Traffic Volume'].unique();assert len(volumes)==1
        complete=available==len(group)
        inventories.append({'Traffic Observation Key':unit,'Observed Traffic Volume':float(volumes[0]),
                            'Source Sections':len(group),'Valid Available Geometry Sections':available,
                            'Complete Observation Unit Geometry':complete,
                            'Traffic Road Classes':';'.join(sorted(group['Traffic Road Class'].unique()))})
        for road_class,parts in shapes.items():
            shape=shapely.union_all(parts)
            bounds=shapely.bounds(shape)
            geometries.append({'Traffic Observation Key':unit,'Target N13 Road Class':road_class,
                               'Observed Traffic Volume':float(volumes[0]),'Complete Observation Unit Geometry':complete,
                               'Geometry WKB':shapely.to_wkb(shape),
                               'Min Longitude':bounds[0],'Min Latitude':bounds[1],
                               'Max Longitude':bounds[2],'Max Latitude':bounds[3]})
    inventory=pd.DataFrame(inventories);frame=pd.DataFrame(geometries)
    inventory.to_parquet(out/'observation_unit_inventory.parquet',index=False)
    inventory.to_csv(out/'observation_unit_inventory.csv',index=False)
    frame.to_parquet(out/'geometry_registry.parquet',index=False)
    pd.DataFrame(bad,columns=['Traffic Geometry Section','Error']).to_csv(out/'invalid_geometry.csv',index=False)
    receipt={'status':'passed','mode':'diagnostic_snapshot' if args.snapshot else 'complete_acquisition_registry',
             'source_manifest_rows':len(manifest),'source_manifest_sha256':digest,
             'measured_observation_units':len(inventory),'available_geometry_groups':len(frame),
             'complete_observation_units':int(inventory['Complete Observation Unit Geometry'].sum()),
             'incomplete_observation_units':int((~inventory['Complete Observation Unit Geometry']).sum()),
             'invalid_source_geometries':len(bad),'all_measured_units_retained':True,
             'road_class_source':'https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadCategory.html',
             'class_mapping':CLASS_MAP,'accepted_road_correspondences':0,
             'coordinate_limit':'Viewer coordinates interpreted geographically; explicit datum correspondence is not independently verified.',
             'result_directory':str(out.relative_to(ROOT))}
    (out/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
