"""Screen exact tile-boundary endpoints for nearby endpoints across archive tiles.
Requires check_national_roads.py. Read-only on derived index; no snapping.
Only exact-boundary points are inspected; this is not a general gap inventory.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, json, math, sqlite3
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/road-quality-check'

def main():
    db=sqlite3.connect(f'file:{OUT / "diagnostic_index.sqlite"}?mode=ro',uri=True)
    grid=defaultdict(list); counts=Counter(); examples=[]
    # 1e-4 degrees exceeds 5 m in both axes within the screened Japan extent.
    for x,y,n,tile,lo,hi in db.execute('SELECT lon,lat,n,first_tile,level_min,level_max FROM endpoints WHERE boundary=1'):
        cell=(math.floor(x*10000),math.floor(y*10000))
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for xx,yy,nn,tt,ll,hh in grid[(cell[0]+dx,cell[1]+dy)]:
                    if tt==tile or (n!=1 and nn!=1): continue
                    a=math.sin(math.radians(y-yy)/2)**2+math.cos(math.radians(y))*math.cos(math.radians(yy))*math.sin(math.radians(x-xx)/2)**2
                    distance=2*6371008.8*math.asin(math.sqrt(min(1,a)))
                    if 0<distance<=5:
                        counts['pairs_within_5m']+=1
                        same=lo==hi==ll==hh
                        if same: counts['pairs_within_5m_same_single_level']+=1
                        if distance<=1.5:
                            counts['pairs_within_1_5m']+=1
                            if same: counts['pairs_within_1_5m_same_single_level']+=1
                        if len(examples)<200:
                            examples.append(dict(longitude_a=xx,latitude_a=yy,tile_a=tt,incidence_a=nn,longitude_b=x,latitude_b=y,tile_b=tile,incidence_b=n,distance_m=distance,same_single_level=same))
        grid[cell].append((x,y,n,tile,lo,hi)); counts['screened_boundary_locations']+=1
    db.close()
    (OUT/'tile_seam_screen.json').write_text(json.dumps(dict(counts),indent=2)+'\n')
    fields=['longitude_a','latitude_a','tile_a','incidence_a','longitude_b','latitude_b','tile_b','incidence_b','distance_m','same_single_level']
    with (OUT/'nearby_tile_seam_examples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(examples)
    print(json.dumps(dict(counts),indent=2))
if __name__=='__main__': main()
