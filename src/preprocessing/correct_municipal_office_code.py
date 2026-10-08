"""Apply the verified one-record source correction without altering raw files.

Run: .venv/bin/python src/preprocessing/correct_municipal_office_code.py
Preserves source property names and point coordinates. This is source repair,
not final variable selection or analysis-ready research preprocessing.
"""
import io
import json
import zipfile
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / 'data/processed/municipal_office_code_corrections.json'


def apply_corrections(properties, archive_name):
    rules = json.loads(RULES.read_text())['corrections']
    result = dict(properties)
    applied = []
    for rule in rules:
        if archive_name != rule['source_archive']:
            continue
        if all(str(properties.get(k)) == v for k,v in rule['match'].items()):
            result[rule['field']] = rule['new_value']
            applied.append(rule)
    return result, applied


def main():
    features, changes = [], []
    for path in sorted((ROOT/'data/raw/poi/municipal_offices').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith('.shp'):
                    continue
                reader = shapefile.Reader(shp=io.BytesIO(archive.read(member)),
                    dbf=io.BytesIO(archive.read(member[:-4]+'.dbf')), encoding='cp932')
                fields = [f[0] for f in reader.fields[1:]]
                for record in reader.iterShapeRecords():
                    original = dict(zip(fields,list(record.record)))
                    corrected, applied = apply_corrections(original,path.name)
                    changed = [key for key in original if original[key]!=corrected[key]]
                    if changed:
                        if changed != ['P34_001']:
                            raise ValueError('Unexpected property change')
                        changes.append(dict(source_archive=path.name,original=original,
                                            corrected=corrected,evidence=applied[0]['evidence']))
                    features.append(dict(type='Feature',properties=corrected,
                        geometry=dict(type='Point',coordinates=list(record.shape.points[0]))))
    if len(changes)!=1 or len(features)!=5774:
        raise ValueError('Unexpected number of records or corrections')
    invalid=[f for f in features if not (str(f['properties']['P34_001']).isdigit()
                                       and len(str(f['properties']['P34_001']))==5)]
    if invalid:
        raise ValueError('Invalid administrative codes remain')
    dest = ROOT/'data/processed/poi/municipal_offices_code_corrected.geojson'
    dest.parent.mkdir(parents=True,exist_ok=True)
    # Retain original JGD2000 coordinates; no CRS transformation is implied.
    payload=dict(type='FeatureCollection',name='Municipal offices: verified code correction',
                 crs={'type':'name','properties':{'name':'EPSG:4612'}},features=features)
    dest.write_text(json.dumps(payload,ensure_ascii=False)+'\n')
    report=dict(records=len(features),corrected_records=len(changes),
                unresolved_invalid_codes=len(invalid),coordinate_reference_system='JGD2000 (EPSG:4612)',
                changes=changes)
    out=ROOT/'data/exp/data-preprocessing'
    out.mkdir(parents=True,exist_ok=True)
    (out/'municipal_office_code_correction_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
