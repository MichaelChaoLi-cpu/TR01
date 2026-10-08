"""Download requested nationwide P17, P30, P35 and P34 source archives.

Run: python3 src/exp/download_additional_facilities.py
Source archives remain unchanged. Only standard-library modules are required.
"""
import concurrent.futures
import csv
import re
import shutil
import time
import urllib.parse

from download_national_facilities import ROOT, REPORT, validate
from download_national_roads import fetch

SOURCES = [
    ('fire_stations', 'P17', '2012', r'P17-12_\d{2}_GML\.zip', 47),
    ('post_offices', 'P30', '2013', r'P30-13\.zip', 1),
    ('roadside_stations', 'P35', '2018', r'P35-18_GML\.zip', 1),
    ('municipal_offices', 'P34', '2014', r'P34-14_\d{2}_GML\.zip', 47),
]
TERMS = 'https://nlftp.mlit.go.jp/ksj/other/agreement_02.html'


def download(row):
    dest = ROOT / 'data/raw/poi' / row['category'] / row['filename']
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not dest.exists():
            partial = dest.with_suffix('.zip.part')
            for attempt in range(3):
                try:
                    with fetch(row['url']) as response, partial.open('wb') as stream:
                        shutil.copyfileobj(response, stream)
                    validate(partial)
                    partial.replace(dest)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        digest, members = validate(dest)
        row.update(bytes=dest.stat().st_size, sha256=digest,
                   status='validated', error='', members=';'.join(members))
    except Exception as error:
        row.update(bytes='', sha256='', status='failed', error=str(error), members='')
    return row


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    with fetch(TERMS) as response:
        (REPORT / 'additional_facility_terms.html').write_bytes(response.read())
    inventory = []
    for category, code, year, pattern, expected in SOURCES:
        source = f'https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-{code}.html'
        with fetch(source) as response:
            page = response.read()
        (REPORT / f'{category}_source_page.html').write_bytes(page)
        links = re.findall(r"DownLd\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'", page.decode('utf-8'))
        matches = sorted(set(r for r in links if re.fullmatch(pattern, r[1])), key=lambda r: r[1])
        if len(matches) != expected:
            raise ValueError(f'{category}: expected {expected} archives, found {len(matches)}')
        if expected == 47:
            codes = {int(re.search(r'_(\d{2})_', r[1]).group(1)) for r in matches}
            if codes != set(range(1, 48)):
                raise ValueError(f'{category}: incomplete prefecture coverage')
        for size, name, relative in matches:
            inventory.append(dict(category=category, release_year=year,
                                  filename=name, source_page=source,
                                  url=urllib.parse.urljoin(source, relative),
                                  advertised_size=size, license='Non-commercial; source-specific terms apply',
                                  terms_url=TERMS))
        print(f'{category}: {len(matches)} official archives, release {year}', flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(download, inventory):
            results.append(row)
            if len(results) % 10 == 0 or row['status'] == 'failed':
                print(f"{len(results)}/{len(inventory)}: {row['filename']} {row['status']}", flush=True)
    manifest = REPORT / 'additional_facility_download_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    failed = sum(r['status'] != 'validated' for r in results)
    print(f'Validated {len(results)-failed}/{len(results)} archives. Manifest: {manifest}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
