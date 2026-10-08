"""Audit public traffic/inspection samples against N13; no accepted crosswalk.
Run with the existing project environment. Inputs are exploratory downloads.
"""
import csv, hashlib, io, json, zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
from pyproj import Transformer
from shapely import LineString, STRtree, box
from shapely.geometry import shape
from shapely.ops import transform, unary_union
from check_national_roads import features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/source-matching-sample'
SRC = OUT / 'sources'
NS = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def csvout(name, rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def xlsx_rows(path):
    """Read cached cell values, excluding ruby phonetic annotations."""
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS):
                strings.append(''.join(t.text or '' for t in si.findall('s:t',NS)+si.findall('s:r/s:t',NS)))
        for name in sorted(n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')):
            with z.open(name) as f:
                for _,row in ET.iterparse(f,events=['end']):
                    if row.tag != '{'+NS['s']+'}row': continue
                    cells={}
                    for c in row:
                        v=c.find('s:v',NS);text=v.text or '' if v is not None else ''
                        if c.get('t')=='s':text=strings[int(text)]
                        if c.get('t')=='inlineStr':text=''.join(t.text or '' for t in c.findall('s:is/s:t',NS)+c.findall('s:is/s:r/s:t',NS))
                        cells[''.join(x for x in c.get('r','') if x.isalpha())]=text
                    yield name,int(row.get('r')),cells
                    row.clear()

def main():
    sop=ROOT/'docs/AnaSOP.md';sop_hash=hashlib.sha256(sop.read_bytes()).hexdigest()
    manifest=json.loads((SRC/'manifest.json').read_text())
    for item in manifest:
        assert hashlib.sha256((SRC/item['file']).read_bytes()).hexdigest()==item['sha256']
    traffic=list(csv.reader((SRC/'traffic_tokyo_2021.csv').read_text(encoding='cp932').splitlines()))
    header=traffic[0];records=traffic[1:]
    assert len(header)==159 and all(len(r)==159 for r in records)
    assert len(header)!=len(set(header)) # original duplicated trailing identifiers retained
    csvout('traffic_field_inventory.csv',[{'position':i,'source_label':v,'coordinate_candidate':any(k in v for k in ['緯度','経度','座標'])} for i,v in enumerate(header)])
    by_id={r[0]:r for r in records};assert len(by_id)==len(records)
    projector=Transformer.from_crs(6668,6677,always_xy=True).transform
    map_projector=Transformer.from_crs(4326,6677,always_xy=True).transform
    line_sources=[]
    for p in sorted(SRC.glob('*.geojson')):
        obj=json.loads(p.read_text());key=p.stem
        assert key in by_id
        for feat in obj['features']:
            assert feat['properties']['R03cen']==key
        geom=unary_union([shape(f['geometry']) for f in obj['features']])
        line_sources.append((key,geom,transform(map_projector,geom)))
    assert len(line_sources)==12
    bounds=unary_union([g for _,g,_ in line_sources]).bounds
    region=box(bounds[0]-.001,bounds[1]-.001,bounds[2]+.001,bounds[3]+.001)
    road_file=ROOT/'data/raw/roads/n13_2024_geojson/N13-24_5339_GEOJSON.zip'
    road_before=hashlib.sha256(road_file.read_bytes()).hexdigest()
    roads=[];ids=[];props=[];scanned=0
    with zipfile.ZipFile(road_file) as z:
        n=next(n for n in z.namelist() if n.endswith('.geojson'))
        with io.TextIOWrapper(z.open(n),encoding='utf-8-sig') as f:
            for i,feat in enumerate(features(f,{}),1):
                scanned+=1;p=feat['properties'];c=feat['geometry']['coordinates']
                if str(p['N13_002'])!='1' or str(p['N13_004']) in ['5','6','7']:continue
                xs=[pt[0] for pt in c];ys=[pt[1] for pt in c]
                if max(xs)<region.bounds[0] or min(xs)>region.bounds[2] or max(ys)<region.bounds[1] or min(ys)>region.bounds[3]:continue
                g=LineString(c)
                if not g.intersects(region):continue
                g=transform(projector,g)
                if g.length<=0:continue
                roads.append(g);ids.append(f'5339:{i}');props.append(p)
    tree=STRtree(roads);results=[];points=[]
    for key,source,g in line_sources:
        pieces=list(g.geoms) if g.geom_type=='MultiLineString' else [g]
        distances=[];ambiguity=0;candidate_ids=set()
        for part_no,line in enumerate(pieces):
            for position in np.linspace(0,line.length,max(2,int(np.ceil(line.length/25))+1)):
                p=line.interpolate(position);idx=int(tree.nearest(p));d=p.distance(roads[idx]);distances.append(d);candidate_ids.add(ids[idx])
                near=tree.query(p.buffer(max(15,d+5)))
                close=[int(j) for j in near if p.distance(roads[int(j)])<=d+5]
                ambiguous=len(close)>1;ambiguity+=ambiguous
                points.append({'survey_id':key,'part':part_no,'position_m':float(position),'nearest_n13_id':ids[idx],'nearest_distance_m':d,'candidates_within_nearest_plus_5m':len(close),'accepted_match':False})
        row=by_id[key]
        results.append({'survey_id':key,'route_name':row[5],'csv_section_length_km':row[23],'map_geometry_length_km':g.length/1000,'up_observation_code':row[34],'down_observation_code':row[46],'total_24h_vehicles_source':row[61],'sample_points':len(distances),'distance_p50_m':float(np.median(distances)),'distance_p95_m':float(np.percentile(distances,95)),'distance_max_m':max(distances),'share_within_5m':sum(d<=5 for d in distances)/len(distances),'share_within_15m':sum(d<=15 for d in distances)/len(distances),'share_within_30m':sum(d<=30 for d in distances)/len(distances),'ambiguous_point_share':ambiguity/len(distances),'nearest_unique_n13_features':len(candidate_ids),'accepted_match':False})
    csvout('traffic_geometry_diagnostics.csv',results);csvout('traffic_point_candidates.csv',points)
    bridge_summary=[];bridge_samples=[];bridge_headers=[]
    for path in sorted(SRC.glob('*.xlsx')):
        count=0;tokyo=0;status=Counter();fields=set()
        for sheet,r,c in xlsx_rows(path):
            if r<=3:
                bridge_headers.extend({'file':path.name,'sheet':sheet,'row':r,'column':k,'label':v} for k,v in c.items() if v)
                continue
            if not c.get('A') or c.get('J') not in ['Ⅰ','Ⅱ','Ⅲ','Ⅳ']:continue
            count+=1;status[c['J']]+=1;fields.update(k for k,v in c.items() if v)
            if c.get('H')=='東京都':
                tokyo+=1
                if sum(x['file']==path.name for x in bridge_samples)<20:
                    bridge_samples.append({'file':path.name,'sheet':sheet,'row':r,'bridge_name':c.get('A'),'route_name':c.get('C'),'construction_year':c.get('D'),'length_m':c.get('E'),'width_m':c.get('F'),'manager':c.get('G'),'prefecture':c.get('H'),'municipality':c.get('I'),'health_category':c.get('J'),'spatial_match_status':'not-attemptable: no coordinates or shared N13 asset identifier in this table'})
        bridge_summary.append({'file':path.name,'health_coded_records':count,'tokyo_records':tokyo,'health_counts':dict(status),'populated_data_columns':sorted(fields)})
    csvout('bridge_source_headers.csv',bridge_headers);csvout('bridge_tokyo_sample.csv',bridge_samples)
    distances=[r['nearest_distance_m'] for r in points]
    summary={'traffic_csv_rows':len(records),'traffic_csv_columns':159,'traffic_unique_header_labels':len(set(header)),'traffic_coordinate_fields':sum(any(k in h for k in ['緯度','経度','座標']) for h in header),'traffic_geometry_samples':12,'traffic_sample_design':'First 12 road-type-3 rows in Tokyo CSV, all National Route 1; convenience format sample, not nationally representative.','n13_scanned_features':scanned,'n13_aoi_candidate_features':len(roads),'n13_properties':sorted(props[0]),'traffic_points':len(points),'pooled_distance_p50_m':float(np.median(distances)),'pooled_distance_p95_m':float(np.percentile(distances,95)),'pooled_share_within_15m':sum(d<=15 for d in distances)/len(distances),'pooled_ambiguous_point_share':sum(r['candidates_within_nearest_plus_5m']>1 for r in points)/len(points),'bridges':bridge_summary,'accepted_traffic_matches':0,'accepted_bridge_matches':0}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    assert hashlib.sha256(road_file.read_bytes()).hexdigest()==road_before
    assert hashlib.sha256(sop.read_bytes()).hexdigest()==sop_hash
    validation={'status':'passed','manifest_files_hash_verified':len(manifest),'traffic_map_csv_ids_verified':12,'candidate_only_no_traffic_assigned':True,'bridge_geolocation_unavailable_in_downloaded_tables':True,'raw_n13_hash_unchanged':True,'anasop_hash_unchanged':True,'anasop_sha256':sop_hash}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    report=f'''# Public maintenance-input source matching sample

Review date: 2026-10-06. Input contract: docs/AnaSOP.md, five applied research questions. National research scope remains unchanged. This is a format and candidate-matching check, not an approved empirical crosswalk.

## Findings

The Tokyo traffic CSV has {len(records):,} rows and 159 columns (153 unique header labels, with six repeated identifiers at the end). It contains traffic counts and observation codes but no coordinate fields. Original duplicated columns are preserved through positional reading. The official map supplies per-section GeoJSON with an R03cen identifier that agrees with all 12 selected CSV section identifiers. This supplies a viable geometry-to-attribute link absent from the CSV alone.

The sample is the first 12 national-road rows in the Tokyo table, all on National Route 1. It is a convenience format sample, not a nationwide or rural validation sample. From the existing N13 tile 5339, {scanned:,} features were scanned and {len(roads):,} positive-length ordinary-centerline candidates retained in the sample envelope. Construction/other/unknown source-state codes were excluded only from this diagnostic candidate set.

At approximately 25-metre spacing including endpoints, {len(points)} sampled traffic-geometry points had median nearest-N13 distance {np.median(distances):.2f} m and 95th percentile {np.percentile(distances,95):.2f} m. {100*summary['pooled_share_within_15m']:.1f}% fell within 15 m of a candidate. {100*summary['pooled_ambiguous_point_share']:.1f}% had multiple candidates within nearest distance plus 5 m. This ambiguity measure includes adjacent fragments at junctions and endpoints as well as parallel alignments; it is not an estimated wrong-match rate. Thresholds of 5/15/30 m are diagnostic tolerances, not accepted correspondence rules.

Map GeoJSON coordinates were interpreted as WGS84 geographic coordinates and N13 as JGD2011, both projected to EPSG:6677 for metre distances. The map's explicit datum metadata was not independently established, so these distances remain exploratory. Pointwise proximity cannot check direction, level, full-section overlap or temporal changes. N13's eight attributes do not supply the traffic survey section identifier, bridge asset identifier, or route/bridge names. Zero traffic-to-N13 matches have been accepted and no traffic values have been assigned to network edges.

## Bridge inspection format

The downloaded FY2024 national-manager table contains {bridge_summary[1]['health_coded_records']:,} rows coded I-IV, including {bridge_summary[1]['tokyo_records']} Tokyo records. The local-manager table contains {bridge_summary[0]['health_coded_records']:,}, including {bridge_summary[0]['tokyo_records']:,} Tokyo records. These are rows in annual inspection-result lists, not a national all-bridge stock count; health-coded rows are explicitly the counting unit. Highway-company lists were not downloaded in this format sample.

Populated data fields are bridge name and reading, route name, construction year, bridge length, width, manager, prefecture, municipality and health category. Neither downloaded table provides coordinates or a shared N13 asset identifier. Twenty Tokyo examples from each manager class are retained in bridge_tokyo_sample.csv. Spatial bridge matching cannot be attempted from these tables alone; a zero accepted-match count must not be interpreted as a measured zero match rate. Annual I-IV inspection classification does not establish cost or the accessibility effect of repair.

## Source and access evidence

- [Traffic CSV and documentation](https://www.mlit.go.jp/road/census/r3/); [official geometry viewer](https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/webmap.html). Its public JavaScript constructs census/13/SECTION.geojson links. Twelve actual files, URLs and SHA256 hashes are retained. Nationwide automated reuse or publication permissions for the map geometry were not established by this format check.
- [FY2024 bridge inspection lists](https://www.mlit.go.jp/road/sisaku/yobohozen/yobohozen_maint_r06.html). Actual Excel files were read directly through ZIP/XML using the Python standard library; no Excel dependency was installed. Phonetic ruby annotations are excluded from parsed text.
- [Current inspection database operator notices](https://rirs.or.jp/tenken-db/) state that provision of basic facility data to non-road-manager users resumed on 2026-07-28. Login/API acquisition and secondary use have application procedures. No account, purchase, application or external message was submitted.
- [Public damage-map entry](https://road-structures-map.mlit.go.jp/) identifies its facility-data snapshot as 2025-11-04 and presents use terms before the map. Its terms are distinct from a documented bulk-download interface. No map-entry agreement was submitted and no map facility records were extracted.

## Decision and next step

Traffic attribute-to-survey-geometry linkage passed this limited sample. Traffic geometry-to-N13 correspondence remains a candidate requiring whole-line overlap, road-class/direction/level checks and review of ambiguous junctions. Before nationwide acquisition, extend this diagnostic to a small rural sample and establish reuse terms for the geometry.

Bridge condition fields are readable, but their spatial linkage is blocked by missing coordinates/common identity in the annual lists. The immediate bridge task is to verify a coordinate-bearing acquisition route and its terms through the documented database or a road manager's public asset inventory. Cost and intervention-to-function inputs remain separate unresolved requirements. The evidence does not authorize narrowing the nationwide road research scope to bridges or adopting equal costs or universal closure.

## Reproduction and validation

Download: .venv/bin/python src/exp/fetch_maintenance_source_sample.py (network required).
Analyze locally: .venv/bin/python src/exp/check_maintenance_source_sample.py.

The download writes only sources/ under this exploratory folder. The analysis verifies downloaded hashes, map/CSV identity, the read N13 archive hash and unchanged AnaSOP hash. See summary.json, validation.json, traffic_geometry_diagnostics.csv, traffic_point_candidates.csv, bridge_source_headers.csv and bridge_tokyo_sample.csv. Raw data and AnaSOP were preserved; no packages, Git or DVC operations were performed. The five-question feasibility classifications in the parent report are retained because costs, intervention effects and complete national traffic/condition coverage are still unresolved.
'''
    (OUT/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
