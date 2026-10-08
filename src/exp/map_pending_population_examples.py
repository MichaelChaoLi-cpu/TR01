"""Render purposive GSI context cards; census mesh support is outlined."""
import json,math
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import fetch_network_review_maps as gsi
from audit_pending_population_entries import OUT
from build_vehicle_network_pilot import csvout


def main():
    data=pd.read_csv(OUT/'pending_mesh_audit.csv',dtype={'point_id':str,'mesh_code':str},keep_default_na=False)
    sample=data.sort_values(['reason','published_local_population','point_id'],ascending=[True,False,True]).groupby('reason').head(2).to_dict('records')
    for i,r in enumerate(sample,1):r['review_id']=f'A{i:03}'
    csvout(OUT/'map_sample_queue.csv',sample)
    gsi.CACHE=OUT/'tile_cache';gsi.CACHE.mkdir(exist_ok=True);gsi.ZOOM=14
    keys=sorted({key for r in sample for key in gsi.tiles_for(r)})
    with ThreadPoolExecutor(max_workers=6) as pool:manifest=list(pool.map(gsi.fetch,keys))
    csvout(OUT/'map_tile_manifest.csv',manifest)
    cards=OUT/'cards';cards.mkdir(exist_ok=True);font=ImageFont.load_default(size=13);small=ImageFont.load_default(size=11)
    height=math.ceil(len(sample)/3)*340
    sheet=Image.new('RGB',(1560,height+25),'white');results=[]
    for i,r in enumerate(sample):
        card=Image.new('RGB',(520,340),'white');d=ImageDraw.Draw(card)
        d.text((3,2),f"{r['review_id']} {r['point_id']} population={r['published_local_population']}",font=font,fill='black')
        d.text((3,19),r['reason'].replace('_',' '),font=small,fill='black');missing=0
        x,y=r['longitude'],r['latitude'];bounds=[(x-1/160,y-1/240),(x+1/160,y-1/240),(x+1/160,y+1/240),(x-1/160,y+1/240),(x-1/160,y-1/240)]
        for j,layer in enumerate(['std','seamlessphoto']):
            panel,left,top,m=gsi.panel(r,layer);missing+=m
            pts=[(gsi.pixel(xx,yy)[0]-left,gsi.pixel(xx,yy)[1]-top) for xx,yy in bounds];ImageDraw.Draw(panel).line(pts,fill='#176FC1',width=2)
            card.paste(panel,(260*j,40));d.text((260*j+3,300),f'{layer} | GSI z14',font=small,fill='black')
        d.text((3,320),'Blue: mesh boundary. Red: mesh center, not observed residence.',font=small,fill='black')
        card.save(cards/f"{r['review_id']}.png");sheet.paste(card,((i%3)*520,(i//3)*340))
        results.append(dict(review_id=r['review_id'],missing_tiles=missing))
    ImageDraw.Draw(sheet).text((4,height+4),'Source: GSI tiles. Mixed/unknown imagery dates; mapped context does not establish vehicle access.',font=small,fill='black')
    sheet.save(OUT/'pending_map_examples.png');csvout(OUT/'map_card_manifest.csv',results)
    status=dict(records=len(sample),tiles=len(manifest),failed_tiles=sum(r['status']=='failed' for r in manifest),selection='Up to two largest-population records in each source-rule stratum; purposive, not representative',zoom=14)
    (OUT/'map_acquisition.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))


if __name__=='__main__':main()
