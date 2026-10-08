"""Download all N13 2024 GeoJSON archives advertised by MLIT.

Run from the project root: python3 src/exp/download_national_roads.py
Existing archives are preserved and validated; interrupted downloads are resumable.
No extraction or transformation of source data is performed.
"""
import concurrent.futures
import csv
import hashlib
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html'
DEST = ROOT / 'data/raw/roads/n13_2024_geojson'
REPORT = ROOT / 'data/exp/data-acquisition'


def fetch(url):
    return urllib.request.urlopen(url, timeout=120)


def validate(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'CRC failure: {bad}')
        if not any(n.lower().endswith(('.geojson', '.json')) for n in archive.namelist()):
            raise ValueError('No GeoJSON/JSON member')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(row):
    size, name, relative = row
    url = 'https://nlftp.mlit.go.jp' + relative
    path = DEST / name
    try:
        if path.exists():
            digest = validate(path)
        else:
            partial = path.with_suffix('.zip.part')
            for attempt in range(3):
                try:
                    with fetch(url) as response, partial.open('wb') as stream:
                        shutil.copyfileobj(response, stream)
                    digest = validate(partial)
                    partial.replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        return dict(filename=name, url=url, advertised_mb=size,
                    bytes=path.stat().st_size, sha256=digest, status='validated', error='')
    except Exception as error:
        return dict(filename=name, url=url, advertised_mb=size,
                    bytes='', sha256='', status='failed', error=str(error))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    with fetch(SOURCE) as response:
        page = response.read()
    snapshot = REPORT / 'n13_2024_source_page.html'
    snapshot.write_bytes(page)
    rows = re.findall(r"DownLd\('([\d.]+)MB','(N13-24_\d+_GEOJSON\.zip)','([^']+)'", page.decode('utf-8'))
    rows = sorted(set(rows), key=lambda r: r[1])
    if not rows:
        raise ValueError('No download links found; source format may have changed')
    print(f'Official inventory: {len(rows)} archives, {sum(float(r[0]) for r in rows):.2f} MB', flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(download, rows):
            results.append(result)
            if len(results) % 10 == 0 or result['status'] == 'failed':
                print(f"{len(results)}/{len(rows)}: {result['filename']} {result['status']}", flush=True)
    manifest = REPORT / 'n13_2024_download_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    failed = sum(r['status'] != 'validated' for r in results)
    print(f'Validated {len(results)-failed}/{len(rows)} archives. Manifest: {manifest}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
