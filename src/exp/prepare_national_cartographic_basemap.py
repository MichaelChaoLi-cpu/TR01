#!/usr/bin/env python3
"""Acquire official 2024 N03 and prepare a display-only prefecture basemap.

No research observations or data/raw files are modified. The archived source,
download receipt, and display geometry are stored under data/exp.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
import zipfile
import shapefile
import requests
from shapely.geometry import shape, mapping
from shapely import make_valid, union_all

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/cartographic-basemap"
PAGE = "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2024.html"
URL = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2024/N03-20240101_GML.zip"
ARCHIVE = OUT / "N03-20240101_GML.zip"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        partial = ARCHIVE.with_suffix(".zip.part")
        with requests.get(URL, stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length",0))
            written = 0
            milestone = 0
            with partial.open("wb") as target:
                for chunk in response.iter_content(1024*1024):
                    target.write(chunk)
                    written += len(chunk)
                    if written // (64*1024*1024) > milestone:
                        milestone = written // (64*1024*1024)
                        print(f"Downloaded {written/1024**2:.0f} MiB", flush=True)
            assert not expected or written == expected
            partial.rename(ARCHIVE)
    assert zipfile.is_zipfile(ARCHIVE)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    sources = OUT / "source_shapefile"
    sources.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        # The official national package includes a pre-dissolved prefecture layer.
        # Prefer it over the separate municipal layer; still union by prefecture.
        shp = [n for n in archive.namelist() if n.lower().endswith('_prefecture.shp')]
        assert len(shp)==1, shp
        stem = shp[0][:-4]
        members = []
        for suffix in ['.shp','.shx','.dbf','.prj']:
            member = stem+suffix
            path = sources / Path(member).name
            if not path.exists():
                with archive.open(member) as source, path.open('wb') as target:
                    while chunk := source.read(1024*1024):target.write(chunk)
            members.append(member)
    reader=shapefile.Reader(str(sources/Path(stem+'.shp').name),encoding='utf-8')
    fields=[x[0] for x in reader.fields[1:]]
    name_column=fields.index('N03_001')
    groups=defaultdict(list)
    invalid=0
    for record in reader.iterShapeRecords():
        geom=shape(record.shape.__geo_interface__)
        if not geom.is_valid:
            geom=make_valid(geom);invalid+=1
        groups[str(record.record[name_column])].append(geom)
    print(f"Loaded {sum(map(len,groups.values())):,} polygons in {len(groups)} prefectures",flush=True)
    assert len(groups)==47
    features=[]
    for index,(name,parts) in enumerate(sorted(groups.items()),1):
        merged=union_all(parts)
        # Display-only tolerance approximately 55 m in latitude; preserve topology.
        simplified=merged.simplify(.0005,preserve_topology=True)
        assert simplified.is_valid and not simplified.is_empty
        features.append({'type':'Feature','properties':{'Prefecture Name':name},'geometry':mapping(simplified)})
        print(f"Prepared prefecture {index}/47",flush=True)
    geojson=OUT/'prefecture_display_boundaries.geojson'
    geojson.write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False))
    receipt={'source_page':PAGE,'source_url':URL,'source_archive':str(ARCHIVE.relative_to(ROOT)),
             'source_sha256':digest,'bytes':ARCHIVE.stat().st_size,'reference_date':'2024-01-01',
             'license':'CC BY 4.0','attribution':'Ministry of Land, Infrastructure, Transport and Tourism, National Land Numerical Information, Administrative Areas (2024)',
             'coordinate_system':'JGD2011 geographic (EPSG:6668)',
             'members_extracted':members,'source_polygons':sum(map(len,groups.values())),
             'source_invalid_geometries_repaired_for_display':invalid,'prefectures':len(features),
             'aggregation':'union by N03_001 prefecture name; municipal boundaries removed',
             'display_simplification_tolerance_degrees':.0005,'preserve_topology':True,
             'role':'Display-only basemap; no sample selection, spatial join or distance change',
             'output':str(geojson.relative_to(ROOT)),'output_sha256':hashlib.sha256(geojson.read_bytes()).hexdigest()}
    (OUT/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f"Saved: {geojson.relative_to(ROOT)}",flush=True)


if __name__=='__main__':main()
