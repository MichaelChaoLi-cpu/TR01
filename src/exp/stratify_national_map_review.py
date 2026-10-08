"""Add explicit published-population context and coverage to geographic map sampling.

Higher/lower local counts are audit proxies, not formal urban/rural definitions.
Missing census mesh records are never assigned zero.
"""
import csv,json,hashlib,math
from screen_national_network_tiles import OUT,CACHE,band,csvout


def code(x,y):
    a=math.floor(y*1.5);b=math.floor(x)-100
    yy=(y-a/1.5)*12;xx=(x-math.floor(x))*8
    c=math.floor(yy);d=math.floor(xx)
    e=math.floor((yy-c)*10);f=math.floor((xx-d)*10)
    return f'{a:02}{b:02}{c}{d}{e}{f}'


def main():
    with (OUT/'points.csv').open() as stream:
        population={r['point_id'].split(':')[1]:r['population_raw'] for r in csv.DictReader(stream) if r['dataset']=='population'}
    def context(r):
        mesh=code(r['longitude'],r['latitude']);value=population.get(mesh)
        r['context_mesh_code']=mesh;r['published_local_population']=value if value is not None else ''
        r['population_audit_proxy']='no_published_record' if value is None else 'higher_count_1000_plus' if int(value)>=1000 else 'lower_count_below_1000'
        return r
    rows=[context(r) for r in json.loads((OUT/'map_review_samples.json').read_text())]
    summaries=json.loads((OUT/'tile_screen_summary.json').read_text());pool=[]
    for r in summaries:pool.extend(context(x) for x in json.loads((CACHE/(r['tile']+'_samples.json')).read_text()))
    additions=[]
    for area in sorted({r['band'] for r in rows}):
        for density in ['higher_count_1000_plus','lower_count_below_1000']:
            present=any(r['band']==area and r['population_audit_proxy']==density and r['kind']!='large_point_offset' for r in rows)
            if present:continue
            options=[r for r in pool if r['band']==area and r['population_audit_proxy']==density]
            if options:
                chosen=min(options,key=lambda r:hashlib.sha256((r['kind']+r['source_ids']).encode()).digest())
                chosen['review_id']=f'N{len(rows)+1:03}';rows.append(chosen);additions.append(chosen['review_id'])
    (OUT/'map_review_samples.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=['review_id','kind','band','longitude','latitude','tile','source_ids','levels','road_states','point_id','offset_m','context_mesh_code','published_local_population','population_audit_proxy']
    csvout(OUT/'map_review_queue.csv',[{k:r.get(k,'') for k in fields} for r in rows])
    coverage=[dict(band=area,proxy=density,records=sum(r['band']==area and r['population_audit_proxy']==density for r in rows)) for area in sorted({r['band'] for r in rows}) for density in ['higher_count_1000_plus','lower_count_below_1000','no_published_record']]
    csvout(OUT/'map_sample_strata.csv',coverage)
    s=json.loads((OUT/'summary.json').read_text());s['map_review_records']=len(rows);(OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
    result=dict(records=len(rows),supplemental_record_ids=additions,proxy='Published 2020 total population in the containing 1 km mesh; 1000-person diagnostic threshold, not a formal urban/rural definition',missing_policy='No published record is not zero population')
    (OUT/'map_sample_design.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
