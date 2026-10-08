"""Bounded official GSI review cards with explicit nearest-candidate overlays."""
import csv
import hashlib
import io
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

from prepare_stratified_bridge_review import ROOT, OUT

CACHE = OUT / 'tile_cache'
SIZE = 512


def pixel(longitude, latitude, zoom):
    n = 256 * 2**zoom
    return ((longitude+180)/360*n, (1-math.asinh(math.tan(math.radians(latitude)))/math.pi)/2*n)


def specs(case):
    length = float(case['bridge_length_m'])
    detail = 18 if length <= 150 else 17
    return [('std', overview(case)[0], True), ('std', detail, True), ('seamlessphoto', detail, False)]


def overview(case):
    longitude=float(case['source_longitude']); latitude=float(case['source_latitude'])
    if case['stratum'] != 'shared_feature': return 16,longitude,latitude
    road=next(r for r in case['candidate_geometries'] if r['source_id']==case['nearest_source_id'])
    points=road['coordinates']+[(longitude,latitude)]+[(float(p['source_longitude']),float(p['source_latitude'])) for p in case['partners']]
    longitude=(min(p[0] for p in points)+max(p[0] for p in points))/2
    latitude=(min(p[1] for p in points)+max(p[1] for p in points))/2
    zoom=16
    while zoom>10:
        pixels=[pixel(*p,zoom) for p in points]
        cx,cy=pixel(longitude,latitude,zoom)
        if max(abs(p[0]-cx) for p in pixels)<=215 and max(abs(p[1]-cy) for p in pixels)<=215: break
        zoom-=1
    return zoom,longitude,latitude


def centre(case,spec):
    if spec[0]=='std' and spec[1]<=16: return overview(case)[1:]
    return float(case['source_longitude']),float(case['source_latitude'])


def keys(case, spec):
    layer,zoom,_ = spec
    x,y = pixel(*centre(case,spec), zoom)
    return [(layer,zoom,xx,yy) for xx in range(math.floor((x-256)/256),math.floor((x+255)/256)+1)
            for yy in range(math.floor((y-256)/256),math.floor((y+255)/256)+1)]


def filename(key):
    layer,zoom,x,y = key
    return f'{layer}_{zoom}_{x}_{y}.' + ('png' if layer == 'std' else 'jpg')


def fetch(key):
    layer,zoom,x,y = key; name = filename(key); path = CACHE/name
    url = f'https://cyberjapandata.gsi.go.jp/xyz/{layer}/{zoom}/{x}/{y}.' + ('png' if layer=='std' else 'jpg')
    try:
        status='cached'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=25) as response: data=response.read()
            Image.open(io.BytesIO(data)).verify(); path.write_bytes(data); status='downloaded'
        data=path.read_bytes()
        return {'file':name,'url':url,'status':status,'error':'','bytes':len(data),
                'sha256':hashlib.sha256(data).hexdigest(),'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    except Exception as error:
        return {'file':name,'url':url,'status':'failed','error':str(error),'bytes':0,'sha256':'','retrieved_utc':datetime.now(timezone.utc).isoformat()}


def panel(case, spec, font):
    layer,zoom,overlay=spec
    longitude=float(case['source_longitude']); latitude=float(case['source_latitude'])
    x,y=pixel(longitude,latitude,zoom)
    cx,cy=pixel(*centre(case,spec),zoom); left=int(cx-256); top=int(cy-256)
    im=Image.new('RGB',(SIZE,SIZE),'#eeeeee'); missing=0
    for key in keys(case,spec):
        path=CACHE/filename(key)
        if path.exists():
            with Image.open(path) as tile: im.paste(tile.convert('RGB'),(key[2]*256-left,key[3]*256-top))
        else: missing+=1
    draw=ImageDraw.Draw(im)
    if overlay:
        first_five = {r['source_id']:i for i,r in enumerate(case['candidate_geometries'][:5])}
        for road in reversed(case['candidate_geometries']):
            pts=[(pixel(lon,lat,zoom)[0]-left,pixel(lon,lat,zoom)[1]-top) for lon,lat in road['coordinates']]
            if road['source_id']==case['nearest_source_id']: color='#00bfdc'; width=4
            elif road['distance_m'] <= float(case['nearest_distance_m'])+5: color='#eb00b4'; width=3
            elif road['road_state']==2: color='#3658cc'; width=2
            else: color='#898989'; width=1
            draw.line(pts,fill=color,width=width)
            if road['source_id'] in first_five:
                # Label closest vertex in this viewport; source attributes remain in the ledger.
                p=min(pts,key=lambda p:(p[0]-256)**2+(p[1]-256)**2)
                if 0<=p[0]<512 and 0<=p[1]<512:
                    draw.text((p[0]+4,p[1]+4+first_five[road['source_id']]*14),f'{road["feature_ordinal"]} C{road["road_class"]} L{road["road_level"]}',font=font,fill=color,stroke_fill='white',stroke_width=1)
    for i,partner in enumerate(case['partners'],1):
        px,py=pixel(float(partner['source_longitude']),float(partner['source_latitude']),zoom)
        px-=left;py-=top
        draw.rectangle((px-5,py-5,px+5,py+5),outline='#e58000',width=3)
        if 0<=px<512 and 0<=py<512: draw.text((px+8,py-14),f'P{i}',font=font,fill='#9b5700',stroke_fill='white',stroke_width=1)
    draw.ellipse((x-left-7,y-top-7,x-left+7,y-top+7),outline='red',width=3)
    resolution=math.cos(math.radians(latitude))*40075016.686/(256*2**zoom)
    scale=200 if zoom<=16 else 100 if zoom==17 else 50
    draw.line((15,492,15+scale/resolution,492),fill='black',width=3)
    draw.text((15,474),f'{scale} m',font=font,fill='black',stroke_fill='white',stroke_width=1)
    return im,missing


def main():
    CACHE.mkdir(parents=True,exist_ok=True); (OUT/'cards').mkdir(exist_ok=True)
    cases=json.loads((OUT/'review_cases.json').read_text())
    allkeys=sorted({k for case in cases for spec in specs(case) for k in keys(case,spec)})
    with ThreadPoolExecutor(max_workers=4) as pool: manifest=list(pool.map(fetch,allkeys))
    with (OUT/'tile_manifest.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    english=ImageFont.load_default(size=14)
    jp=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',19)
    missing=0; hashes={}
    for case in cases:
        card=Image.new('RGB',(1552,650),'white');draw=ImageDraw.Draw(card)
        draw.text((8,3),f'{case["review_id"]} {case["bridge_name"]} | {case["route_name"]}',font=jp,fill='black')
        draw.text((8,29),f'{case["stratum"]} | pref={case["source_prefecture"]} | bridge {case["bridge_length_m"]} x {case["bridge_width_m"]} m | inspection={case["inspection_year"]}',font=english,fill='black')
        draw.text((8,48),f'Nearest {case["nearest_source_id"]} | {float(case["nearest_distance_m"]):.2f} m | feature={float(case["nearest_source_length_m"]):.1f} m | state={case["nearest_road_state"]} | level={case["nearest_road_level"]} | shared assets={case["assets_sharing_nearest_feature"]}',font=english,fill='black')
        for i,spec in enumerate(specs(case)):
            draw.text((i*520+8,70),f'{spec[0]} z{spec[1]}'+(' + N13 candidates' if spec[2] else ' (no road overlay)'),font=english,fill='black')
            im,m=panel(case,spec,english);missing+=m;card.paste(im,(i*520,94))
        draw.text((8,611),'Cyan: nearest | magenta: nearest+5 m | blue: other bridge/elevated | grey: other | red: source point | orange: partners',font=english,fill='black')
        draw.text((8,631),'Source: GSI tiles; N13 overlay added. Photograph dates mixed/unverified. No accepted maintenance assignment or footprint.',font=english,fill='black')
        path=OUT/'cards'/f'{case["review_id"]}.png';card.save(path);hashes[case['review_id']]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={'cards':len(cases),'tiles':len(manifest),'failed_tiles':sum(r['status']=='failed' for r in manifest),
            'missing_card_tiles':missing,'source':'https://maps.gsi.go.jp/development/ichiran.html',
            'card_sha256':hashes,'photograph_dates':'mixed/unverified'}
    (OUT/'map_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
