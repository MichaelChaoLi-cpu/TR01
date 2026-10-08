"""Download bounded GSI cartographic evidence and render ten bridge review cards."""
import csv,hashlib,io,json,math,urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/feasibility-check/bridge-crosswalk-review'
CACHE=OUT/'tile_cache';SIZE=512;ZOOM=18
def pixel(lon,lat):
    n=256*2**ZOOM;return ((lon+180)/360*n,(1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)
def keys(case):
    x,y=pixel(case['longitude'],case['latitude'])
    return [(layer,xx,yy) for layer in ['std','seamlessphoto'] for xx in range(math.floor((x-256)/256),math.floor((x+256)/256)+1) for yy in range(math.floor((y-256)/256),math.floor((y+256)/256)+1)]
def fetch(key):
    layer,x,y=key;ext='png' if layer=='std' else 'jpg';filename=f'{layer}_{ZOOM}_{x}_{y}.{ext}';path=CACHE/filename
    url=f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{ZOOM}/{x}/{y}.{ext}';status='cached';error=''
    try:
        if not path.exists():
            with urllib.request.urlopen(url,timeout=25) as response:data=response.read()
            Image.open(io.BytesIO(data)).verify();path.write_bytes(data);status='downloaded'
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
    except Exception as e:status='failed';error=str(e);data=b'';digest=''
    return {'file':filename,'url':url,'status':status,'error':error,'bytes':len(data),'sha256':digest,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
def panel(case,layer):
    x,y=pixel(case['longitude'],case['latitude']);left=int(x-256);top=int(y-256);im=Image.new('RGB',(SIZE,SIZE),'#eeeeee');missing=0
    for ll,xx,yy in keys(case):
        if ll!=layer:continue
        ext='png' if layer=='std' else 'jpg';path=CACHE/f'{layer}_{ZOOM}_{xx}_{yy}.{ext}'
        if path.exists():
            with Image.open(path) as tile:im.paste(tile.convert('RGB'),(xx*256-left,yy*256-top))
        else:missing+=1
    draw=ImageDraw.Draw(im)
    if layer=='std':
        for road in reversed(case['candidate_geometries']):
            pts=[(pixel(lon,lat)[0]-left,pixel(lon,lat)[1]-top) for lon,lat in road['coordinates']]
            near=road['source_id']==case['nearest_source_id'];structure=str(road['properties']['N13_004'])=='2'
            draw.line(pts,fill='#00a9ce' if near else '#e000ad' if structure else '#999999',width=3 if near or structure else 1)
    draw.ellipse((x-left-7,y-top-7,x-left+7,y-top+7),outline='red',width=3)
    # Metric scale at latitude; retain the original aerial pixels apart from marker/scale.
    resolution=math.cos(math.radians(case['latitude']))*40075016.686/(256*2**ZOOM);bar=50/resolution
    draw.line((15,492,15+bar,492),fill='black',width=3);draw.text((15,476),'50 m',fill='black',stroke_width=1,stroke_fill='white')
    return im,missing
def main():
    CACHE.mkdir(parents=True,exist_ok=True);(OUT/'cards').mkdir(exist_ok=True)
    cases=json.loads((OUT/'review_cases.json').read_text());allkeys=sorted({k for c in cases for k in keys(c)})
    with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(fetch,allkeys))
    with (OUT/'tile_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    cardmanifest=[];font=ImageFont.load_default(size=15);small=ImageFont.load_default(size=12)
    for c in cases:
        card=Image.new('RGB',(1032,586),'white');d=ImageDraw.Draw(card)
        d.text((6,3),f"{c['review_id']}  asset length={c['asset_length_m']:g} m | width={c['asset_width_m']:g} m | inspection={c['inspection_year']}",fill='black',font=font)
        d.text((6,22),f"Nearest {c['nearest_source_id']} | distance={c['nearest_distance_m']:.2f} m | state={c['nearest_state']} | feature length={c['nearest_feature_length_m']:.1f} m",fill='black',font=small)
        missing=0
        for i,layer in enumerate(['std','seamlessphoto']):
            im,m=panel(c,layer);missing+=m;card.paste(im,(i*520,40))
        d.text((6,554),'GSI std + N13: cyan nearest, magenta bridge/elevated, grey other | red = API asset location',font=small,fill='black')
        d.text((6,568),'GSI seamlessphoto z18 | imagery dates mixed/unverified; source: maps.gsi.go.jp/development/ichiran.html',font=small,fill='black')
        card.save(OUT/'cards'/f"{c['review_id']}.png");cardmanifest.append({'review_id':c['review_id'],'missing_tiles':missing})
    (OUT/'map_validation.json').write_text(json.dumps({'cards':len(cases),'tiles':len(manifest),'failed_tiles':sum(m['status']=='failed' for m in manifest),'missing_card_tiles':sum(m['missing_tiles'] for m in cardmanifest),'source':'https://maps.gsi.go.jp/development/ichiran.html','dates':'mixed/unverified'},indent=2)+'\n')
    print((OUT/'map_validation.json').read_text())
if __name__=='__main__':main()
