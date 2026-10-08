"""Clip original national road curves to native meshes and conserve source costs.

Each geographic source segment is split at every mesh crossing. Tile-local
AEQD lengths of the clipped curve pieces allocate the original source length.
Half-open mesh cells assign boundary-only segments once, never twice.
"""
from pathlib import Path
import hashlib
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/national-pavement-allocation-computation'
POLICY=ROOT/'data/exp/national-pavement-policy-computation'


def digest(p):
    with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def clip_batch(curves,ids,lengths,selected,transform):
    counts=np.array([len(c) for c in curves],dtype=np.int64)
    coords=np.concatenate(curves);offsets=np.r_[0,np.cumsum(counts)]
    segment_starts=np.arange(len(coords)-1)
    keep=np.ones(len(segment_starts),dtype=bool);keep[offsets[1:-1]-1]=False
    starts=segment_starts[keep]
    road=np.repeat(np.arange(len(ids)),counts-1)
    a=coords[starts];b=coords[starts+1];vector=b-a;n=len(a)
    segments=[np.arange(n),np.arange(n)];cuts=[np.zeros(n),np.ones(n)]
    for axis,scale in [(0,80),(1,120)]:
        lo=np.minimum(a[:,axis],b[:,axis])*scale;hi=np.maximum(a[:,axis],b[:,axis])*scale
        crossings=np.maximum(0,np.ceil(hi).astype(int)-np.floor(lo).astype(int)-1)
        for step in range(1,int(crossings.max(initial=0))+1):
            index=np.flatnonzero(crossings>=step)
            position=(np.floor(lo[index])+step)/scale
            fraction=(position-a[index,axis])/vector[index,axis]
            # Multiplication by the mesh scale can round an endpoint onto an
            # adjacent integer. Endpoints already have cuts at zero and one.
            assert ((fraction>=-1e-7)&(fraction<=1+1e-7)).all()
            interior=(fraction>0)&(fraction<1)
            segments.append(index[interior]);cuts.append(fraction[interior])
    segment=np.concatenate(segments);cut=np.concatenate(cuts)
    order=np.lexsort((cut,segment));segment=segment[order];cut=cut[order]
    same=segment[:-1]==segment[1:];valid=same&(cut[1:]-cut[:-1]>1e-14)
    seg=segment[:-1][valid];left=cut[:-1][valid];right=cut[1:][valid]
    c=a[seg]+vector[seg]*left[:,None];d=a[seg]+vector[seg]*right[:,None]
    cx,cy=transform.transform(c[:,0],c[:,1]);dx,dy=transform.transform(d[:,0],d[:,1])
    metric=np.hypot(dx-cx,dy-cy);unit=road[seg]
    totals=np.bincount(unit,weights=metric,minlength=len(ids));assert (totals>0).all()
    weight=metric/totals[unit]*lengths[unit]
    midpoint=(c+d)/2
    # Longitude columns start at 100 degrees in the original JIS encoding.
    row=np.floor(midpoint[:,1]*120).astype(np.int64)
    col=np.floor((midpoint[:,0]-100)*80).astype(np.int64)
    key=row*100000+col
    unique,inverse=np.unique(key,return_inverse=True)
    sums={'Mesh Grid Key':unique,'Candidate Source Road Length':np.bincount(inverse,weights=weight)}
    for rule in ['Traffic','Condition','Access']:
        sums[f'{rule} Maintained Source Road Length']=np.bincount(inverse,
            weights=weight*selected[rule][ids[unit]],minlength=len(unique))
    np.testing.assert_allclose(np.bincount(unit,weights=weight,minlength=len(ids)),lengths,rtol=1e-12,atol=1e-8)
    # Independent rectangle-intersection oracle on an actual curve from each
    # batch, favouring a crossing road when one exists.
    per_road=np.bincount(unit,minlength=len(ids))
    candidate=int(np.argmax(per_road));line=shapely.LineString(curves[candidate])
    hit=unit==candidate;oracle=[];observed=[]
    for k in np.unique(key[hit]):
        rr,cc=divmod(int(k),100000)
        box=shapely.box(100+cc/80,rr/120,100+(cc+1)/80,(rr+1)/120)
        geom=shapely.intersection(line,box)
        xmin,ymin,xmax,ymax=box.bounds
        excluded=shapely.MultiLineString([[(xmin,ymax),(xmax,ymax)],[(xmax,ymin),(xmax,ymax)]])
        geom=shapely.difference(geom,excluded)
        projected=shapely.transform(geom,transform.transform,interleaved=False)
        oracle.append(shapely.length(projected));observed.append(float(metric[hit&(key==k)].sum()))
    # Independent closed-rectangle clipping becomes the same half-open cell
    # after removing its north/east boundary lines.
    np.testing.assert_allclose(observed,oracle,rtol=1e-7,atol=1e-5)
    return pd.DataFrame(sums)


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);(OUT/'tiles').mkdir(exist_ok=True)
    lengths=np.load(ROOT/'data/processed/national-network-rule-validation/edge_lengths.npy',mmap_mode='r')
    selected={};selection_hashes={}
    for rule in ['Traffic','Condition','Access']:
        path=POLICY/'cases'/f'{rule.lower()}_0.01_selected.npy'
        chosen=np.load(path);flag=np.zeros(len(lengths),dtype=bool);flag[chosen]=True;selected[rule]=flag
        selection_hashes[rule]=digest(path)
    config={'authorization':'MILI-D-20261008-001','selected_set_hashes':selection_hashes,
        'script_sha256':digest(Path(__file__)),'mesh_boundary_convention':'half-open east/north ownership',
        'metric_clipping':'tile AEQD metric lengths; normalized to original source length'}
    config_path=OUT/'configuration.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config
    else:config_path.write_text(json.dumps(config,indent=2)+'\n')
    frames=[];roads=0;checks=0;source_hashes={}
    offsets=np.load(ROOT/'data/processed/national-network-quality/tile_offsets.npy')
    for tile,start,count in offsets:
        tile=int(tile);start=int(start);count=int(count)
        target=OUT/'tiles'/f'{tile:04}.parquet';receipt=target.with_suffix('.json')
        source=ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{tile:04}_GEOJSON.zip'
        source_hash=digest(source);source_hashes[str(source.relative_to(ROOT))]=source_hash
        if receipt.exists():
            info=json.loads(receipt.read_text());assert info['source_sha256']==source_hash
            assert info['configuration']==config
            frames.append(pd.read_parquet(target));roads+=count;checks+=info['independent_clipping_checks'];continue
        cache=np.load(ROOT/f'data/processed/national-network-quality/tiles/{tile:04}.npz')
        fids=cache['fids'];assert len(fids)==count
        x0=100+int(str(tile)[2:]);y0=int(str(tile)[:2])/1.5
        tr=Transformer.from_crs(6668,f'+proj=aeqd +lat_0={y0+1/3} +lon_0={x0+.5} +datum=WGS84 +units=m',always_xy=True)
        pointer=ordinal=0;curves=[];ids=[];parts=[];tile_checks=0
        def flush():
            nonlocal tile_checks
            if ids:
                ids_array=np.array(ids,dtype=np.int64)
                parts.append(clip_batch(curves,ids_array,lengths[ids_array],selected,tr));tile_checks+=1
                curves.clear();ids.clear()
        with zipfile.ZipFile(source) as archive:
            name=next(n for n in archive.namelist() if n.endswith('.geojson'))
            with io.TextIOWrapper(archive.open(name),encoding='utf-8-sig') as stream:
                for line in stream:
                    if not line.lstrip().startswith('{ "type": "Feature",'):continue
                    ordinal+=1
                    if pointer==count or ordinal!=fids[pointer]:continue
                    feature=json.loads(line.strip().rstrip(','));assert feature['geometry']['type']=='LineString'
                    curves.append(np.asarray(feature['geometry']['coordinates'],dtype=float));ids.append(start+pointer);pointer+=1
                    if len(ids)==10000:flush()
        flush();assert pointer==count
        columns=['Mesh Grid Key','Candidate Source Road Length']+[f'{r} Maintained Source Road Length' for r in selected]
        frame=pd.concat(parts,ignore_index=True).groupby('Mesh Grid Key',as_index=False).sum() if parts else pd.DataFrame({c:pd.Series(dtype='int64' if c=='Mesh Grid Key' else 'float64') for c in columns})
        np.testing.assert_allclose(frame['Candidate Source Road Length'].sum(),lengths[start:start+count].sum(),rtol=1e-12,atol=1e-6)
        assert digest(source)==source_hash
        frame.to_parquet(target,index=False)
        info={'source_sha256':source_hash,'configuration':config,'candidate_roads':count,'independent_clipping_checks':tile_checks}
        receipt.write_text(json.dumps(info,indent=2)+'\n');frames.append(frame);roads+=count;checks+=tile_checks
        progress={'completed_tiles':len(frames),'total_tiles':len(offsets),'candidate_roads':roads,'elapsed_seconds':time.perf_counter()-started}
        (OUT/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(f'Native road-length allocation: {len(frames)}/{len(offsets)} tiles; {roads:,} roads',flush=True)
    total=pd.concat(frames,ignore_index=True).groupby('Mesh Grid Key',as_index=False).sum()
    assert roads==21415800
    key=total['Mesh Grid Key'].to_numpy();row=key//100000;col=key%100000
    mesh=[f'{r//80:02}{c//80:02}{r%80//10}{c%80//10}{r%10}{c%10}' for r,c in zip(row,col)]
    results=[]
    for rule in selected:
        numerator=total[f'{rule} Maintained Source Road Length'].to_numpy();denominator=total['Candidate Source Road Length'].to_numpy()
        share=numerator/denominator;assert ((share>=0)&(share<=1+1e-12)).all()
        np.testing.assert_allclose(numerator.sum(),lengths[selected[rule]].sum(),rtol=1e-12,atol=1e-6)
        results.append(pd.DataFrame({'Mesh Code':mesh,'Longitude':100+(col+.5)/80,'Latitude':(row+.5)/120,
            'Pavement Maintenance Priority':rule,'Pavement Budget Share':.01,
            'Candidate Source Road Length':denominator,'Maintained Source Road Length':numerator,
            'Mesh Maintained Road Length Share':share}))
    np.testing.assert_allclose(denominator.sum(),lengths.sum(),rtol=1e-12,atol=1e-6)
    pd.concat(results,ignore_index=True).to_parquet(OUT/'national_pavement_mesh_allocation.parquet',index=False)
    overlap=[]
    sets={r:np.flatnonzero(f) for r,f in selected.items()}
    for a in sets:
        for b in sets:
            intersection=len(np.intersect1d(sets[a],sets[b],assume_unique=True));union=len(sets[a])+len(sets[b])-intersection
            overlap.append({'First Pavement Priority':a,'Second Pavement Priority':b,
                'Pavement Budget Share':.01,'Selection Intersection Road Count':intersection,
                'Selection Union Road Count':union,'Pavement Rule Selection Overlap':intersection/union if union else np.nan})
    pd.DataFrame(overlap).to_parquet(OUT/'national_pavement_rule_overlap.parquet',index=False)
    report={'status':'passed_national_mesh_allocation_construction','candidate_roads':roads,'road_bearing_meshes':len(total),
        'independent_clipping_checks':checks,'original_source_length_conserved':True,'selected_lengths_conserved':True,
        'original_curves_used':True,'boundary_segments_counted_once':True,'all_source_hashes_unchanged':source_hashes,
        'selected_set_hashes':selection_hashes,'elapsed_seconds':time.perf_counter()-started}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='all_source_hashes_unchanged'},indent=2))


if __name__=='__main__': main()
