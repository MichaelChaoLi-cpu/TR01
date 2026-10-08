"""Fetch a bounded public API/traffic sample and published documentation."""
import csv,hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/feasibility-check/bridge-access-rural-sample/sources'
API='https://road-structures-db.mlit.go.jp/xROAD/api/v1/bridges'
def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest=[]
    def fetch(name,url):
        with urllib.request.urlopen(url,timeout=50) as response:
            data=response.read();status=response.status
        (OUT/name).write_bytes(data)
        manifest.append({'file':name,'url':url,'status':status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'retrieved_utc':datetime.now(timezone.utc).isoformat()})
        return data
    traffic=fetch('traffic_tochigi_2021.csv','https://www.mlit.go.jp/road/census/r3/data/csv/kasyo09.csv')
    rows=list(csv.reader(traffic.decode('cp932').splitlines()))[1:]
    nikko=[r for r in rows if r[18].zfill(5)=='09206' and float(r[23] or 0)>0]
    selected=[r for r in nikko if r[4]=='120' and r[3]=='3'][:2]
    for kind in ['4','6']:selected += [r for r in nikko if r[3]==kind][:2]
    for row in selected:
        key=row[0].zfill(11)
        fetch(key+'.geojson','https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/census/09/'+key+'.geojson')
    for name,query in [('bridge_api_city5','?city=09206&limit=5'),('bridge_api_pref','?pref=09&limit=5'),('bridge_api_area','?area=36.7,36.8,139.6,139.75&limit=5'),('bridge_nikko_latest','/lastest?city=092061&limit=10')]:
        data=fetch(name+'.json',API+query)
        obj=json.loads(data)
        if obj.get('resultset',{}).get('is_error'):raise RuntimeError('API returned an error; stop rather than expand requests.')
    for name,url in [('inspection_access_terms.pdf','https://rirs.or.jp/tenken-db/pdf/view/etsuran_kiyaku.pdf'),('inspection_api_specification.pdf','https://rirs.or.jp/tenken-db/pdf/view/api_specification.pdf'),('inspection_basic_parameters.pdf','https://rirs.or.jp/tenken-db/pdf/view/00_Parameters_base.pdf')]:fetch(name,url)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
