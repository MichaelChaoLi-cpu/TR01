"""Read-only nationwide N13 audit. Stdlib, streaming JSON, disk-backed exact endpoints.
Run: .venv/bin/python src/exp/check_national_roads.py
No coordinate rounding, snapping, geometry repair, or road routing is performed.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, io, json, math, re, sqlite3, struct, zipfile
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/road-quality-check'
FIELDS = [f'N13_{i:03}' for i in range(1,9)]

def features(stream, header):
    decoder = json.JSONDecoder()
    buf = ''
    while '"features"' not in buf:
        chunk = stream.read(65536)
        if not chunk: raise ValueError('Missing features')
        buf += chunk
    pos = buf.index('"features"')
    header.update(json.loads(buf[:pos].rstrip().rstrip(',') + '}'))
    buf = buf[buf.index('[',pos)+1:]
    while True:
        buf = buf.lstrip(' \r\n\t,')
        if buf.startswith(']'): return
        try:
            obj, end = decoder.raw_decode(buf)
        except json.JSONDecodeError:
            chunk = stream.read(1024*1024)
            if not chunk: raise ValueError('Truncated feature array')
            buf += chunk
            continue
        yield obj
        buf = buf[end:]

def write_csv(name, rows, fields):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dbpath=OUT/'diagnostic_index.sqlite'
    if dbpath.exists(): dbpath.unlink()  # regenerated derived index only
    db=sqlite3.connect(dbpath)
    db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('CREATE TABLE endpoints (xy BLOB PRIMARY KEY, lon REAL, lat REAL, n INTEGER, first_tile TEXT, cross_tile INTEGER, level_min INTEGER, level_max INTEGER, boundary INTEGER) WITHOUT ROWID')
    db.execute('CREATE TABLE geometries (hash BLOB PRIMARY KEY, n INTEGER, first_tile TEXT, first_feature INTEGER, cross_tile INTEGER) WITHOUT ROWID')
    ep_sql='INSERT INTO endpoints VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(xy) DO UPDATE SET n=n+excluded.n,cross_tile=MAX(cross_tile,first_tile!=excluded.first_tile),level_min=MIN(level_min,excluded.level_min),level_max=MAX(level_max,excluded.level_max),boundary=MAX(boundary,excluded.boundary)'
    geo_sql='INSERT INTO geometries VALUES (?,?,?,?,?) ON CONFLICT(hash) DO UPDATE SET n=n+1,cross_tile=MAX(cross_tile,first_tile!=excluded.first_tile)'
    manifest={r['filename']:r for r in csv.DictReader((ROOT/'data/exp/data-acquisition/n13_2024_download_manifest.csv').open())}
    counts=Counter(); attrs={k:Counter() for k in FIELDS}; tiles=[]; anomalies=[]; crs=Counter(); integrity=[]
    total_km=0.; bounds=[math.inf,math.inf,-math.inf,-math.inf]
    for ti,path in enumerate(sorted((ROOT/'data/raw/roads/n13_2024_geojson').glob('*.zip')),1):
        tile=path.name.split('_')[1]; s=Counter(); s['archive']=path.name
        y0=int(tile[:2])/1.5; x0=100+int(tile[2:]); ep=[]; geos=[]; header={}
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        integrity.append({'archive':path.name,'sha256_matches_manifest':digest==manifest[path.name]['sha256']})
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('.geojson')]
            if len(names)!=1: raise ValueError((path,names))
            with io.TextIOWrapper(z.open(names[0]),encoding='utf-8-sig') as stream:
                for i,f in enumerate(features(stream,header),1):
                    s['features']+=1
                    p=f.get('properties') or {}; g=f.get('geometry') or {}
                    for k in FIELDS:
                        attrs[k][str(p.get(k))]+=1
                        if k not in p or p[k] is None or p[k]=='': s['missing_attribute_values']+=1
                    if set(p)!=set(FIELDS): s['unexpected_schema_features']+=1
                    if g.get('type')!='LineString': s['non_linestring']+=1; continue
                    c=g.get('coordinates',[])
                    valid=len(c)>=2 and all(isinstance(v,list) and len(v)==2 and all(isinstance(t,(float,int)) and math.isfinite(t) for t in v) for v in c)
                    if not valid:
                        s['invalid_geometry']+=1
                        if len(anomalies)<200: anomalies.append({'archive':path.name,'feature':i,'issue':'invalid_geometry'})
                        continue
                    s['vertices']+=len(c)
                    if any(not(120<=x<=155 and 20<=y<=46) for x,y in c): s['outside_broad_japan_bounds']+=1
                    for x,y in c:
                        bounds[0]=min(bounds[0],x); bounds[1]=min(bounds[1],y); bounds[2]=max(bounds[2],x); bounds[3]=max(bounds[3],y)
                    length=0.; repeated=0
                    for (x1,y1),(x2,y2) in zip(c,c[1:]):
                        if x1==x2 and y1==y2: repeated+=1
                        a=math.sin(math.radians(y2-y1)/2)**2+math.cos(math.radians(y1))*math.cos(math.radians(y2))*math.sin(math.radians(x2-x1)/2)**2
                        length+=2*6371.0088*math.asin(math.sqrt(min(1,a)))
                    total_km+=length; s['length_km']+=length
                    s['repeated_consecutive_vertices']+=repeated
                    if repeated: s['features_with_repeated_vertices']+=1
                    if length==0: s['zero_length_features']+=1
                    if c[0]==c[-1]: s['closed_lines']+=1
                    level=p.get('N13_005')
                    if not isinstance(level,int): s['non_integer_levels']+=1; level=0
                    for x,y in (c[0],c[-1]):
                        boundary=int(min(abs(x-x0),abs(x-x0-1),abs(y-y0),abs(y-y0-2/3))<1e-8)
                        ep.append((struct.pack('!dd',x,y),x,y,1,tile,0,level,level,boundary))
                    binary=b''.join(struct.pack('!dd',x,y) for x,y in c)
                    reverse=b''.join(struct.pack('!dd',x,y) for x,y in reversed(c))
                    geos.append((hashlib.sha256(min(binary,reverse)).digest(),1,tile,i,0))
                    if len(geos)>=10000:
                        db.executemany(ep_sql,ep); db.executemany(geo_sql,geos); ep.clear(); geos.clear()
        db.executemany(ep_sql,ep); db.executemany(geo_sql,geos); db.commit()
        crs[json.dumps(header.get('crs'),sort_keys=True)]+=1
        tiles.append(dict(s)); counts.update({k:v for k,v in s.items() if k not in ('archive','length_km')})
        print(f'{ti}/160 {tile}: {s["features"]:,} features; cumulative {counts["features"]:,}',flush=True)
    scalar=lambda sql:db.execute(sql).fetchone()[0]
    summary=dict(counts)
    summary.update(archives=len(tiles),length_km_haversine=total_km,bounds=bounds,crs=dict(crs),raw_hashes_all_match=all(r['sha256_matches_manifest'] for r in integrity),
        unique_exact_endpoints=scalar('SELECT COUNT(*) FROM endpoints'),degree_one_endpoints=scalar('SELECT COUNT(*) FROM endpoints WHERE n=1'),
        exact_cross_tile_endpoints=scalar('SELECT COUNT(*) FROM endpoints WHERE cross_tile=1'),mixed_level_endpoints=scalar('SELECT COUNT(*) FROM endpoints WHERE level_min!=level_max'),
        boundary_degree_one_endpoints=scalar('SELECT COUNT(*) FROM endpoints WHERE boundary=1 AND n=1'),
        duplicate_geometry_groups=scalar('SELECT COUNT(*) FROM geometries WHERE n>1'),duplicate_geometry_excess=scalar('SELECT COALESCE(SUM(n-1),0) FROM geometries WHERE n>1'),
        cross_tile_duplicate_geometry_groups=scalar('SELECT COUNT(*) FROM geometries WHERE n>1 AND cross_tile=1'))
    assert summary['features']==sum(t['features'] for t in tiles)
    assert scalar('SELECT SUM(n) FROM endpoints')==2*(summary['features']-summary.get('non_linestring',0)-summary.get('invalid_geometry',0))
    assert scalar('SELECT SUM(n) FROM geometries')*2==scalar('SELECT SUM(n) FROM endpoints')
    summary['internal_count_validation']='passed'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv('archive_summary.csv',tiles,sorted(set().union(*(t.keys() for t in tiles))))
    write_csv('attribute_frequencies.csv',[{'field':k,'value':v,'count':n} for k,a in attrs.items() for v,n in a.most_common()],['field','value','count'])
    write_csv('endpoint_degree_distribution.csv',[{'endpoint_incidence':n,'locations':k} for n,k in db.execute('SELECT n,COUNT(*) FROM endpoints GROUP BY n ORDER BY n')],['endpoint_incidence','locations'])
    for name,where in [('boundary_unjoined_endpoint_examples','boundary=1 AND n=1'),('mixed_level_endpoint_examples','level_min!=level_max')]:
        cols=['longitude','latitude','incidence','first_tile','cross_tile','min_level','max_level']
        write_csv(name+'.csv',[dict(zip(cols,r)) for r in db.execute('SELECT lon,lat,n,first_tile,cross_tile,level_min,level_max FROM endpoints WHERE '+where+' LIMIT 200')],cols)
    write_csv('duplicate_geometry_examples.csv',[dict(zip(['sha256','incidence','first_tile','first_feature','cross_tile'],r)) for r in db.execute('SELECT hex(hash),n,first_tile,first_feature,cross_tile FROM geometries WHERE n>1 LIMIT 200')],['sha256','incidence','first_tile','first_feature','cross_tile'])
    (OUT/'source_integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n')
    db.close()
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
