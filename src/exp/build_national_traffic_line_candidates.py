"""National full-curve traffic correspondence candidates, without acceptance.

Road class, overlap and local tangent agreement are retained for a concrete
matching-rule review. Diagnostic thresholds do not assign measured traffic.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/exp/national-traffic-geometry'
PROC=ROOT/'data/processed/national-network-rule-validation'
FIELDS=['Graph Source Edge ID','Registry Group Index','Source Road Length',
        'Minimum Line Separation','Road Fraction Within Diagnostic Buffer',
        'Aligned Sample Fraction','Complete Observation Unit Geometry',
        'Technical Proposal Qualified','N13 Road Level','N13 Road State']
REGISTRY=None
GEOMETRIES=None
BOUNDS=None


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def initialize(path):
    global REGISTRY,GEOMETRIES,BOUNDS
    REGISTRY=pd.read_parquet(path)
    GEOMETRIES=shapely.from_wkb(REGISTRY['Geometry WKB'].to_numpy())
    BOUNDS=REGISTRY[['Min Longitude','Min Latitude','Max Longitude','Max Latitude']].to_numpy()


def aligned_fraction(roads,targets,road_index,target_index):
    """Undirected local tangent agreement at five full-road sample positions."""
    selected=roads[road_index]
    lengths=shapely.length(selected)
    positions=lengths[:,None]*np.linspace(0,1,5)
    samples=shapely.line_interpolate_point(selected[:,None],positions)
    before=shapely.line_interpolate_point(selected[:,None],np.maximum(0,positions-5))
    after=shapely.line_interpolate_point(selected[:,None],np.minimum(lengths[:,None],positions+5))
    vectors=shapely.get_coordinates(after.ravel())-shapely.get_coordinates(before.ravel())
    vectors=vectors.reshape(-1,5,2)
    result=np.zeros(len(road_index))
    for target in np.unique(target_index):
        chosen=np.flatnonzero(target_index==target)
        parts=shapely.get_parts(targets[target]);parts=parts[shapely.length(parts)>0]
        assert len(parts)>0
        points=samples[chosen].ravel()
        matches=shapely.STRtree(parts).query_nearest(points,all_matches=False)
        assert np.array_equal(matches[0],np.arange(len(points)))
        nearest=parts[matches[1]]
        along=shapely.line_locate_point(nearest,points)
        first=shapely.line_interpolate_point(nearest,np.maximum(0,along-5))
        last=shapely.line_interpolate_point(nearest,np.minimum(shapely.length(nearest),along+5))
        target_vectors=shapely.get_coordinates(last)-shapely.get_coordinates(first)
        source_vectors=vectors[chosen].reshape(-1,2)
        norms=np.linalg.norm(source_vectors,axis=1)*np.linalg.norm(target_vectors,axis=1)
        dot=np.abs(np.sum(source_vectors*target_vectors,axis=1))
        cos=np.divide(dot,norms,out=np.zeros_like(dot),where=norms>0)
        result[chosen]=(cos.reshape(-1,5)>=np.cos(np.deg2rad(30))).mean(axis=1)
    return result


def process_tile(tile,config):
    started=time.perf_counter();code=f'{tile:04d}'
    out=ROOT/config['output_directory'];summary_path=out/'tiles'/f'{code}.json';array_path=out/'tiles'/f'{code}.npz'
    road_path=ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{code}_GEOJSON.zip'
    source_hash=digest(road_path);assert source_hash==config['raw_hashes'][road_path.name]
    if summary_path.exists() and array_path.exists():
        previous=json.loads(summary_path.read_text());assert previous['registry_sha256']==config['registry_sha256']
        assert previous['candidate_sha256']==digest(array_path) and previous['raw_sha256']==source_hash
        return previous
    offset=np.load(PROC/'tile_offsets.npy');row=offset[offset[:,0]==tile][0]
    start,count=map(int,row[1:]);ordinals=np.load(PROC/'edge_source_ordinals.npy',mmap_mode='r')[start:start+count]
    x0,y0=100+int(code[2:]),int(code[:2])/1.5
    active=np.flatnonzero((BOUNDS[:,0]<=x0+1+.001)&(BOUNDS[:,2]>=x0-.001)&
                          (BOUNDS[:,1]<=y0+2/3+.001)&(BOUNDS[:,3]>=y0-.001))
    chunks={field:[] for field in FIELDS}
    total=0;pointer=0;incompatible=0
    if len(active):
        transformer=Transformer.from_crs(6668,f'+proj=aeqd +lat_0={y0+1/3} +lon_0={x0+.5} +datum=WGS84 +units=m',always_xy=True)
        targets=shapely.transform(GEOMETRIES[active],transformer.transform,interleaved=False)
        tree=shapely.STRtree(targets);buffers=shapely.buffer(targets,20)
        target_classes=REGISTRY['Target N13 Road Class'].to_numpy()[active]
        complete=REGISTRY['Complete Observation Unit Geometry'].to_numpy()[active]
        coordinates=[];ends=[0];metadata=[]

        def flush():
            nonlocal incompatible
            if not metadata:return
            roads=shapely.from_ragged_array(shapely.GeometryType.LINESTRING,np.asarray(coordinates,dtype=float),(np.asarray(ends,dtype=np.int64),))
            roads=shapely.transform(roads,transformer.transform,interleaved=False)
            meta=np.asarray(metadata,dtype=np.int64)
            matches=tree.query(roads,predicate='dwithin',distance=50)
            if matches.shape[1]:
                ri,ti=matches;consistent=meta[ri,1]==target_classes[ti]
                incompatible+=int((~consistent).sum());ri,ti=ri[consistent],ti[consistent]
                if len(ri):
                    lengths=shapely.length(roads[ri]);assert (lengths>0).all()
                    separation=shapely.distance(roads[ri],targets[ti])
                    overlap=np.clip(shapely.length(shapely.intersection(roads[ri],buffers[ti]))/lengths,0,1)
                    alignment=aligned_fraction(roads,targets,ri,ti)
                    eligible=complete[ti]&(overlap>=.8)&(alignment>=.8)
                    values=[meta[ri,0].astype(np.int32),active[ti].astype(np.int32),lengths,separation,
                            overlap,alignment,complete[ti],eligible,meta[ri,2].astype(np.int16),meta[ri,3].astype(np.int8)]
                    for field,value in zip(FIELDS,values):chunks[field].append(value)
            coordinates.clear();ends[:]=[0];metadata.clear()

        with zipfile.ZipFile(road_path) as archive:
            name=next(n for n in archive.namelist() if n.endswith('.geojson'))
            with io.TextIOWrapper(archive.open(name),encoding='utf-8-sig') as handle:
                for line in handle:
                    if not line.lstrip().startswith('{ "type": "Feature",'):continue
                    total+=1
                    if pointer>=count or total!=ordinals[pointer]:continue
                    edge=start+pointer;pointer+=1
                    feature=json.loads(line.strip().rstrip(','));coords=feature['geometry']['coordinates'];props=feature['properties']
                    coordinates.extend(coords);ends.append(len(coordinates))
                    metadata.append((edge,int(props['N13_003']),int(props['N13_005']),int(props['N13_004'])))
                    if len(metadata)>=25000:flush()
        flush();assert pointer==count and total==config['audited_counts'][road_path.name]
    arrays={field:np.concatenate(parts) if parts else np.empty(0,dtype=bool if field in ['Complete Observation Unit Geometry','Technical Proposal Qualified'] else np.int32 if field in ['Graph Source Edge ID','Registry Group Index','N13 Road Level','N13 Road State'] else float) for field,parts in chunks.items()}
    part=array_path.with_suffix('.part')
    with part.open('wb') as handle:np.savez_compressed(handle,**arrays)
    part.replace(array_path);assert digest(road_path)==source_hash
    result={'tile':code,'candidate_pairs':len(arrays['Graph Source Edge ID']),
            'proposal_qualified_pairs':int(arrays['Technical Proposal Qualified'].sum()),
            'class_incompatible_pairs_excluded':incompatible,'active_geometry_groups':len(active),
            'source_ordinal_count_checked':bool(len(active)),'raw_sha256':source_hash,
            'registry_sha256':config['registry_sha256'],'candidate_sha256':digest(array_path),
            'seconds':time.perf_counter()-started,'accepted_correspondences':0}
    summary_path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registry',default='data/exp/national-traffic-geometry/registry/geometry_registry.parquet')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2);parser.add_argument('--tiles',nargs='*',type=int)
    args=parser.parse_args();registry=ROOT/args.registry;assert registry.exists()
    registry_receipt=json.loads((registry.parent/'validation.json').read_text())
    registry_hash=digest(registry)
    out=BASE/'line-candidates'/registry_hash[:12];(out/'tiles').mkdir(parents=True,exist_ok=True)
    config={'registry_path':args.registry,'registry_sha256':registry_hash,'registry_mode':registry_receipt['mode'],
            'output_directory':str(out.relative_to(ROOT)),
            'audited_counts':{r['archive']:int(r['features']) for r in csv.DictReader((ROOT/'data/exp/road-quality-check/archive_summary.csv').open())},
            'raw_hashes':{r['filename']:r['sha256'] for r in csv.DictReader((ROOT/'data/exp/data-acquisition/n13_2024_download_manifest.csv').open())},
            'diagnostic_search_metres':50,'diagnostic_overlap_buffer_metres':20,
            'proposal_minimum_road_overlap':.8,'proposal_minimum_aligned_sample_fraction':.8,
            'tangent_angle_limit_degrees':30,'accepted_correspondences':0}
    config_path=out/'configuration.json'
    if config_path.exists():assert json.loads(config_path.read_text())==config
    else:config_path.write_text(json.dumps(config,indent=2)+'\n')
    available=np.load(PROC/'tile_offsets.npy')[:,0].astype(int).tolist();tiles=args.tiles or available
    assert set(tiles)<=set(available)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(registry,)) as pool:
        futures={pool.submit(process_tile,tile,config):tile for tile in tiles}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps({k:result[k] for k in ['tile','candidate_pairs','proposal_qualified_pairs','seconds']}),flush=True)
            (out/'progress.json').write_text(json.dumps({'completed_tiles_this_run':len(results),'requested_tiles':len(tiles),'accepted_correspondences':0},indent=2)+'\n')
    assert digest(registry)==registry_hash
    validation={'status':'candidate_generation_complete','registry_mode':config['registry_mode'],
                'tiles':len(results),'all_national_tiles':set(tiles)==set(available),
                'candidate_pairs':sum(r['candidate_pairs'] for r in results),
                'proposal_qualified_pairs':sum(r['proposal_qualified_pairs'] for r in results),
                'accepted_correspondences':0,'full_source_curves':True,
                'output_directory':str(out.relative_to(ROOT)),'geometry_registry_sha256':registry_hash}
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
