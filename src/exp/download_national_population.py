"""Acquire nationwide 2020 census 1km JGD2011 tables from e-Stat.

Run: python3 src/exp/download_national_population.py
Preserves original ZIPs, encoding, suppression and aggregation codes.
"""
import concurrent.futures
import csv
import hashlib
import html
import json
import re
import shutil
import time
import urllib.parse
import zipfile

from download_national_roads import ROOT, REPORT, fetch

BASE = 'https://www.e-stat.go.jp'
TABLES = [('T001140', 'population_households'), ('T001188', 'five_year_age_groups')]


def validate(path, stats, code):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError('ZIP CRC failure')
        expected = f'tbl{stats}S{code}.txt'
        if archive.namelist() != [expected]:
            raise ValueError(f'Unexpected members: {archive.namelist()}')
        with archive.open(expected) as stream:
            if not stream.readline().startswith(b'KEY_CODE,HTKSYORI,HTKSAKI,GASSAN,'):
                raise ValueError('Unexpected table header')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(row):
    path = ROOT / 'data/raw/population/census_2020_1km_jgd2011' / row['table_id'] / row['filename']
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not path.exists():
            partial = path.with_suffix('.zip.part')
            for attempt in range(3):
                try:
                    with fetch(row['url']) as response, partial.open('wb') as stream:
                        shutil.copyfileobj(response, stream)
                    validate(partial, row['table_id'], row['mesh_tile'])
                    partial.replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2*(attempt+1))
        digest = validate(path, row['table_id'], row['mesh_tile'])
        row.update(status='validated', bytes=path.stat().st_size, sha256=digest, error='')
    except Exception as error:
        row.update(status='failed', bytes='', sha256='', error=str(error))
    return row


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for stats, title in TABLES:
        params = dict(type='1', toukeiCode='00200521', toukeiYear='2020', aggregateUnit='S',
                      serveyId='S002005112020', statsId=stats, datum='2011',
                      mesh_data_flg='1', download_disp_flg='1', page='1')
        urls = set()
        page, pages = 1, 1
        while page <= pages:
            params['page'] = str(page)
            url = BASE + '/gis/statmap-search/search_detail?' + urllib.parse.urlencode(params)
            with fetch(url) as response:
                payload = response.read()
            (REPORT / f'{stats}_catalog_page_{page}.json').write_bytes(payload)
            info = json.loads(payload)
            if page == 1:
                counts = re.findall(r'data-page="(\d+)"', info['paginate'])
                pages = max(map(int, counts), default=1)
            matches = re.findall(r'href="([^"]*statsId='+stats+r'&code=\d+&downloadType=2)"', info['detail'])
            if not matches:
                raise ValueError(f'{stats} page {page}: no official download links')
            urls.update(BASE + html.unescape(u) for u in matches)
            page += 1
        print(f'{stats}: {len(urls)} official first-level tiles, {pages} catalog pages', flush=True)
        for url in sorted(urls):
            code = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)['code'][0]
            inventory.append(dict(table_id=stats, table=title, census_year='2020',
                                  grid='1km', datum='JGD2011', mesh_tile=code,
                                  filename=f'{stats}_{code}.zip', url=url))
        with fetch(BASE + f'/gis/statmap-search/data?datatype=1&statsId={stats}&downloadType=1') as response:
            definition = response.read()
        if not definition.startswith(b'%PDF'):
            raise ValueError('Unexpected definition document format')
        (REPORT / f'{stats}_definition.pdf').write_bytes(definition)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(download, inventory):
            results.append(row)
            if len(results) % 25 == 0 or row['status'] == 'failed':
                print(f"{len(results)}/{len(inventory)}: {row['filename']} {row['status']}", flush=True)
    manifest = REPORT / 'population_2020_download_manifest.csv'
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
