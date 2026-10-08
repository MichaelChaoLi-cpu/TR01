"""Download only the approved public format sample into data/exp, never raw.

Run with network access: .venv/bin/python src/exp/fetch_maintenance_source_sample.py
"""
import csv, hashlib, json, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/feasibility-check/source-matching-sample/sources'
BASE='https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def get(name,url):
        data=urllib.request.urlopen(url,timeout=50).read()
        (OUT/name).write_bytes(data)
        manifest.append({'file':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        return data
    data=get('traffic_tokyo_2021.csv','https://www.mlit.go.jp/road/census/r3/data/csv/kasyo13.csv')
    for name in ['webmap.html','property.js']:get(name,BASE+name)
    for filename,official in [('bridges_national_manager_2024.xlsx','01-1'),('bridges_local_managers_2024.xlsx','01-3')]:
        get(filename,'https://www.mlit.go.jp/road/sisaku/yobohozen/xls/r06/'+official+'.xlsx')
    rows=list(csv.DictReader(data.decode('cp932').splitlines()))
    for row in [r for r in rows if r['道路種別']=='3'][:12]:
        key=row['交通調査基本区間番号']
        get(key+'.geojson',BASE+'census/13/'+key+'.geojson')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
