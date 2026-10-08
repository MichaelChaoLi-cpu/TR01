"""Validate audit totals and independently parse three small source archives."""
from pathlib import Path
import csv,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/road-quality-check'
rows={r['archive']:r for r in csv.DictReader((OUT/'archive_summary.csv').open())}
s=json.loads((OUT/'summary.json').read_text())
assert len(rows)==s['archives']==160
assert sum(int(r['features']) for r in rows.values())==s['features']
assert s['raw_hashes_all_match'] and s['internal_count_validation']=='passed'
assert json.loads((OUT/'code_validation.json').read_text())['count']==0
checks=[]
for tile in ['3622','5039','6848']:
    name=f'N13-24_{tile}_GEOJSON.zip'
    with zipfile.ZipFile(ROOT/'data/raw/roads/n13_2024_geojson'/name) as z:
        member=next(n for n in z.namelist() if n.endswith('.geojson'))
        data=json.loads(z.read(member))
    count=len(data['features']); assert count==int(rows[name]['features'])
    checks.append({'archive':name,'full_json_feature_count':count,'matches_streamed_count':True})
(OUT/'independent_parse_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Validation passed.')
