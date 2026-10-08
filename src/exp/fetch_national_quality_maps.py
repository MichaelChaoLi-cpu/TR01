"""Fetch GSI evidence for the deterministic national screening review sample."""
import json,math
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw,ImageFont
import fetch_network_review_maps as gsi
from screen_national_network_tiles import OUT
from build_vehicle_network_pilot import csvout


def main():
    rows=json.loads((OUT/'map_review_samples.json').read_text())
    gsi.CACHE=OUT/'tile_cache';gsi.CACHE.mkdir(exist_ok=True)
    cards=OUT/'cards';sheets=OUT/'sheets';cards.mkdir(exist_ok=True);sheets.mkdir(exist_ok=True)
    font=ImageFont.load_default(size=13);small=ImageFont.load_default(size=11)
    manifest=[];cardmanifest=[]
    for zoom in [18,13]:
        selected=[r for r in rows if (13 if r['kind']=='large_point_offset' else 18)==zoom]
        gsi.ZOOM=zoom
        keys=sorted({key for r in selected for key in gsi.tiles_for(r)})
        with ThreadPoolExecutor(max_workers=6) as pool:manifest.extend(pool.map(gsi.fetch,keys))
        for r in selected:
            card=Image.new('RGB',(520,316),'white');d=ImageDraw.Draw(card)
            d.text((4,2),r['review_id']+' '+r['kind']+' '+r['band'],font=font,fill='black')
            d.text((4,19),'L='+r.get('levels','')+' S='+r.get('road_states','')+' '+r['source_ids'],font=small,fill='black')
            missing=0
            for i,layer in enumerate(['std','seamlessphoto']):
                panel,left,top,m=gsi.panel(r,layer);missing+=m
                if layer=='std':
                    draw=ImageDraw.Draw(panel)
                    for j,coords in enumerate(r['geometries']):
                        pts=[(gsi.pixel(x,y)[0]-left,gsi.pixel(x,y)[1]-top) for x,y in coords]
                        if len(pts)>1:draw.line(pts,fill=['#d300b2','#00a8ce','#df8b00'][j%3],width=1)
                    draw.ellipse((121,121,135,135),outline='red',width=2)
                card.paste(panel,(i*260,38));d.text((i*260+2,295),f'{layer} | GSI tiles z{zoom}',font=small,fill='black')
            card.save(cards/(r['review_id']+'.png'))
            cardmanifest.append(dict(review_id=r['review_id'],zoom=zoom,missing_tiles=missing))
    csvout(OUT/'tile_manifest.csv',manifest);csvout(OUT/'card_manifest.csv',cardmanifest)
    for start in range(0,len(rows),12):
        sheet=Image.new('RGB',(2080,3*316+28),'white')
        for i,r in enumerate(rows[start:start+12]):
            with Image.open(cards/(r['review_id']+'.png')) as im:sheet.paste(im,((i%4)*520,(i//4)*316))
        ImageDraw.Draw(sheet).text((4,3*316+4),'Source: GSI standard map and seamless aerial photo; source N13 lines on map only; mixed/unknown photo dates.',font=small,fill='black')
        sheet.save(sheets/f'review_{start//12+1:02}.png')
    result=dict(review_records=len(rows),tiles=len(manifest),failed_tiles=sum(r['status']=='failed' for r in manifest),cards_with_missing_tiles=sum(r['missing_tiles']>0 for r in cardmanifest),sheets=math.ceil(len(rows)/12),photo_dates='mixed/unknown; absence does not refute N13-2024 road presence')
    (OUT/'map_acquisition.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
