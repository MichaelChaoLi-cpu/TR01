"""Create a source-traceable nationwide diagnostic point inventory, preserving raw data."""
import csv, io, json, sys, zipfile
from pathlib import Path
import numpy as np
import shapefile
from pyproj import CRS, Transformer
from build_vehicle_network_pilot import ROOT, mesh_center, csvout
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from correct_municipal_office_code import apply_corrections
OUT=ROOT/'data/exp/national-network-quality'


def layers(z,provenance):
    for name in z.namelist():
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(z.read(name))) as child:yield from layers(child,provenance+'!'+name)
        elif name.endswith('.shp'):yield z,name,provenance+'!'+name


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audit=[]
    for path in sorted((ROOT/'data/raw/population/census_2020_1km_jgd2011/T001140').glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            with io.TextIOWrapper(z.open(z.namelist()[0]),encoding='cp932') as f:
                r=csv.reader(f);cols=next(r);next(r)
                for values in r:
                    p=dict(zip(cols,values));code=p['KEY_CODE'];x,y=mesh_center(code)
                    rows.append(dict(point_id='mesh:'+code,dataset='population',longitude=x,latitude=y,category='',name='',population_raw=p['T001140001'],suppression_code=p['HTKSYORI'],source_id=str(path.relative_to(ROOT))+':'+code))
    fields=dict(fire_stations=('P17_003','P17_001'),medical=('P04_001','P04_002'),welfare=('P14_006','P14_008'),municipal_offices=('P34_002','P34_003'),post_offices=('P30_004','P30_005'),roadside_stations=('P35_011','P35_006'))
    for folder,(category,namefield) in fields.items():
        count=0
        for path in sorted((ROOT/'data/raw/poi'/folder).glob('*.zip')):
            with zipfile.ZipFile(path) as z:
                for child,name,prov in layers(z,str(path.relative_to(ROOT))):
                    stem=name[:-4];reader=shapefile.Reader(shp=io.BytesIO(child.read(name)),dbf=io.BytesIO(child.read(stem+'.dbf')),encoding='utf-8' if folder=='welfare' else 'cp932')
                    if reader.shapeType!=1:continue
                    if stem+'.prj' in child.namelist():crs=CRS.from_wkt(child.read(stem+'.prj').decode());evidence='source .prj'
                    else:crs=CRS.from_epsg(4612 if folder=='fire_stations' else 6668);evidence='official source specification'
                    tr=Transformer.from_crs(crs,6668,always_xy=True)
                    audit.append(dict(dataset=folder,layer=prov,source_crs=crs.to_string(),evidence=evidence))
                    for i,sr in enumerate(reader.iterShapeRecords(),1):
                        assert sr.shape.points
                        x,y=tr.transform(*sr.shape.points[0]);p=sr.record.as_dict()
                        if folder=='municipal_offices':p=apply_corrections(p,path.name)[0]
                        rows.append(dict(point_id=f'{folder}:{count}',dataset=folder,longitude=x,latitude=y,category=str(p.get(category,'')),name=str(p.get(namefield,'')),population_raw='',suppression_code='',source_id=prov+':'+str(i)));count+=1
        print(folder,count,flush=True)
    assert len({p['point_id'] for p in rows})==len(rows)
    facts=json.loads((ROOT/'data/exp/data-briefing/inspection_facts.json').read_text())
    counts={k:sum(r['dataset']==k for r in rows) for k in ['population',*fields]}
    assert counts['population']==facts['T001140']['rows']
    assert all(counts[k]==facts[k]['rows'] for k in fields)
    csvout(OUT/'points.csv',rows);csvout(OUT/'point_crs_audit.csv',audit)
    np.save(OUT/'point_coordinates.npy',np.array([(p['longitude'],p['latitude']) for p in rows]))
    (OUT/'point_inventory.json').write_text(json.dumps(dict(points=len(rows),datasets=counts,validation='passed'),indent=2)+'\n')
    print(counts,flush=True)


if __name__=='__main__':main()
