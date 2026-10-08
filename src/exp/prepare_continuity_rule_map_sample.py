"""Select a geographic/state holdout sample from applied continuity candidates."""
import csv,hashlib,io,json,zipfile
from collections import defaultdict
from screen_national_network_tiles import ROOT,band,csvout
from validate_national_continuity_rule import OUT,sitekey


def main():
    known=list(csv.DictReader((OUT/'prior_visual_evidence_rule_checks.csv').open()))
    excluded={sitekey(r['longitude'],r['latitude']) for r in known}
    applied=list(csv.DictReader((OUT/'applied_structure_transitions.csv').open()));groups=defaultdict(list)
    for r in applied:
        if sitekey(r['longitude'],r['latitude']) in excluded:continue
        x,y=float(r['longitude']),float(r['latitude']);kind='bridge_endpoint' if '2' in r['road_states'].split(';') else 'tunnel_endpoint'
        groups[band(x,y),kind].append(dict(r,longitude=x,latitude=y,kind=kind,band=band(x,y)))
    selected=[]
    for group,options in sorted(groups.items()):
        selected.extend(sorted(options,key=lambda r:hashlib.sha256(r['source_ids'].encode()).digest())[:2])
    wanted=defaultdict(set)
    for r in selected:
        for source in r['source_ids'].split(';'):
            tile,fid=source.split(':');wanted[tile].add(int(fid))
    geometry={}
    for tile,ids in wanted.items():
        with zipfile.ZipFile(ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip') as z:
            name=next(n for n in z.namelist() if n.endswith('.geojson'));i=0
            with io.TextIOWrapper(z.open(name),encoding='utf-8-sig') as stream:
                for text in stream:
                    if not text.lstrip().startswith('{ "type": "Feature",'):continue
                    i+=1
                    if i in ids:geometry[f'{tile}:{i}']=json.loads(text.strip().rstrip(','))['geometry']['coordinates']
                    if i>=max(ids):break
    for i,r in enumerate(selected,1):
        r['review_id']=f'R{i:03}';r['geometries']=[geometry[source] for source in r['source_ids'].split(';')]
    (OUT/'map_review_samples.json').write_text(json.dumps(selected,indent=2)+'\n')
    csvout(OUT/'map_review_queue.csv',[{k:r[k] for k in ['review_id','kind','band','longitude','latitude','source_ids','levels','road_states']} for r in selected])
    print(f'{len(selected)} independent geographic/state sample records',flush=True)


if __name__=='__main__':main()
