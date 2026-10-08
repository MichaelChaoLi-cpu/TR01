"""Download official nationwide medical and welfare source archives.

Run: python3 src/exp/download_national_facilities.py
Archives remain unchanged and are not extracted or filtered.
"""
import csv
import hashlib
import io
import re
import shutil
import time
import urllib.parse
import zipfile
from pathlib import Path

from download_national_roads import fetch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'data/exp/data-acquisition'
SOURCES = [
    ('medical', 'https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P04-2020.html', 'P04-20_GML.zip'),
    ('welfare', 'https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P14-2023.html', 'P14-23_GML.zip'),
]


def validate(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'CRC failure: {bad}')
        names = archive.namelist()
        spatial = any(n.lower().endswith(('.gml', '.xml', '.shp', '.geojson')) for n in names)
        nested = [n for n in names if n.lower().endswith('.zip')]
        for member in nested:
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as child:
                if child.testzip():
                    raise ValueError(f'Nested ZIP CRC failure: {member}')
                if not any(n.lower().endswith(('.gml', '.xml', '.shp', '.geojson')) for n in child.namelist()):
                    raise ValueError(f'No spatial data in {member}')
        if not spatial and not nested:
            raise ValueError('No spatial data member')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest(), names


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = []
    for category, source, name in SOURCES:
        with fetch(source) as response:
            page = response.read()
        (REPORT / f'{category}_source_page.html').write_bytes(page)
        links = re.findall(r"DownLd\('([^']+)','([^']+)','([^']+)'", page.decode('utf-8'))
        matches = [r for r in links if r[1] == name]
        if len(matches) != 1:
            raise ValueError(f'Expected one official national archive for {category}')
        size, _, relative = matches[0]
        url = urllib.parse.urljoin(source, relative)
        dest = ROOT / 'data/raw/poi' / category / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            partial = dest.with_suffix('.zip.part')
            for attempt in range(3):
                try:
                    with fetch(url) as response, partial.open('wb') as stream:
                        shutil.copyfileobj(response, stream)
                    validate(partial)
                    partial.replace(dest)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        digest, members = validate(dest)
        rows.append(dict(category=category, filename=name, source_page=source,
                         url=url, advertised_size=size, bytes=dest.stat().st_size,
                         sha256=digest, status='validated', members=';'.join(members)))
        print(f'{category}: {name}, {dest.stat().st_size} bytes, validated; archive members: {len(members)}', flush=True)
    with (REPORT / 'facility_download_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
