"""Acquire official GSI cartographic evidence and render reproducible review cards.
Uses existing Pillow; saves caches only in data/exp. No source geometry is changed.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,hashlib,io,json,math,time,urllib.request
from PIL import Image,ImageDraw,ImageFont
from pyproj import Transformer
from build_vehicle_network_pilot import ROOT,PROC,csvout
OUT=ROOT/'data/exp/network-map-review';CACHE=OUT/'tile_cache';ZOOM=18;SIZE=256

def pixel(lon,lat):
    n=256*2**ZOOM
    return ((lon+180)/360*n,(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)

def tiles_for(row):
    x,y=pixel(float(row['longitude']),float(row['latitude']))
    return [(layer,xx,yy) for layer in ['std','seamlessphoto'] for xx in range(math.floor((x-128)/256),math.floor((x+128)/256)+1) for yy in range(math.floor((y-128)/256),math.floor((y+128)/256)+1)]

def fetch(key):
    layer,x,y=key;ext='png' if layer=='std' else 'jpg';name=f'{layer}_{ZOOM}_{x}_{y}.{ext}';path=CACHE/name
    url=f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{ZOOM}/{x}/{y}.{ext}'
    error='';status='cached' if path.exists() else 'downloaded'
    if not path.exists():
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'TR01 research topology audit; non-commercial'})
                with urllib.request.urlopen(req,timeout=20) as f:data=f.read()
                Image.open(io.BytesIO(data)).verify();path.write_bytes(data);break
            except Exception as exc:
                error=str(exc)
                if attempt<2:time.sleep(.3*(attempt+1))
        else:status='failed'
    if path.exists():
        error='';digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();size=path.stat().st_size
    else:digest='';size=0
    return dict(layer=layer,z=ZOOM,x=x,y=y,filename=name,url=url,status=status,error=error,sha256=digest,bytes=size,retrieved_at=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat())

def panel(row,layer):
    x,y=pixel(float(row['longitude']),float(row['latitude']));left=int(x-128);top=int(y-128)
    result=Image.new('RGB',(256,256),'#eeeeee');missing=0
    for ll,xx,yy in tiles_for(row):
        if ll!=layer:continue
        ext='png' if layer=='std' else 'jpg';path=CACHE/f'{ll}_{ZOOM}_{xx}_{yy}.{ext}'
        if path.exists():
            with Image.open(path) as im:result.paste(im.convert('RGB'),(xx*256-left,yy*256-top))
        else:missing+=1
    d=ImageDraw.Draw(result);d.ellipse((121,121,135,135),outline='red',width=2)
    return result,left,top,missing

def main():
    CACHE.mkdir(parents=True,exist_ok=True);cards=OUT/'cards';sheets=OUT/'sheets';cards.mkdir(exist_ok=True);sheets.mkdir(exist_ok=True)
    rows=list(csv.DictReader((OUT/'review_queue.csv').open()));keys=sorted({k for r in rows for k in tiles_for(r)})
    manifests=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,result in enumerate(pool.map(fetch,keys),1):
            manifests.append(result)
            if i%100==0:print(f'{i}/{len(keys)} tiles',flush=True)
    csvout(OUT/'tile_manifest.csv',manifests)
    features=json.loads((PROC/'edges_layer_aware.geojson').read_text())['features'];inverse=Transformer.from_crs(6670,6668,always_xy=True)
    font=ImageFont.load_default(size=13);small=ImageFont.load_default(size=11);card_manifest=[]
    for r in rows:
        card=Image.new('RGB',(520,316),'white');d=ImageDraw.Draw(card)
        d.text((4,2),r['review_id']+' '+r['kind'],font=font,fill='black')
        d.text((4,19),'L='+r['levels']+' S='+r['road_states']+' '+r['source_ids'],font=small,fill='black')
        total_missing=0
        for li,layer in enumerate(['std','seamlessphoto']):
            p,left,top,missing=panel(r,layer);total_missing+=missing
            if layer=='std':
                dd=ImageDraw.Draw(p)
                for ei,eid in enumerate(r['edge_ids'].split(';')):
                    coords=features[int(eid)]['geometry']['coordinates'];pts=[]
                    for xx,yy in coords:
                        lon,lat=inverse.transform(xx,yy);px,py=pixel(lon,lat);pts.append((px-left,py-top))
                    dd.line(pts,fill=['#d300b2','#00a8ce','#df8b00'][ei%3],width=1)
                dd.ellipse((121,121,135,135),outline='red',width=2)
            card.paste(p,(li*260,38));d.text((li*260+2,295),layer+' | GSI tiles z18',font=small,fill='black')
        card.save(cards/(r['review_id']+'.png'))
        card_manifest.append(dict(review_id=r['review_id'],file='cards/'+r['review_id']+'.png',missing_tiles=total_missing))
    csvout(OUT/'card_manifest.csv',card_manifest)
    for start in range(0,len(rows),16):
        selected=rows[start:start+16];sheet=Image.new('RGB',(2080,4*316+28),'white')
        for i,r in enumerate(selected):
            with Image.open(cards/(r['review_id']+'.png')) as im:sheet.paste(im,((i%4)*520,(i//4)*316))
        d=ImageDraw.Draw(sheet);d.text((5,4*316+4),'Source: Geospatial Information Authority of Japan (GSI), standard map and seamless aerial photo. N13 lines overlaid only on map; photos unaltered except candidate marker.',font=small,fill='black')
        sheet.save(sheets/f'review_{start//16+1:02}.png')
    summary=dict(locations=len(rows),tiles=len(keys),failed_tiles=sum(m['status']=='failed' for m in manifests),sheets=math.ceil(len(rows)/16),cards_with_missing_tiles=sum(m['missing_tiles']>0 for m in card_manifest),source='https://maps.gsi.go.jp/development/ichiran.html',photo_dates='mixed/unknown at individual candidate; no synchronized 2024 claim',download_date='2026-10-06')
    (OUT/'map_acquisition.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
