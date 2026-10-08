"""Acquire the public latest pavement inventory with resumable source evidence.

Reuse the tested public pagination/cache primitives in a separate process with
a pavement-specific endpoint and output directory. No source-to-road assignment.
"""
import csv
import gzip
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fetch_national_bridge_inventory as api

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-pavement-inventory'
ENDPOINT = 'https://road-structures-db.mlit.go.jp/xROAD/api/v1/pavements/lastest'
SIZE = 5000


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + '.' + key if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def audit_pages(results, national_count):
    manifests = sorted((OUT / 'sources').glob('pref_*/offset_*.request.json'))
    fields, ids, rows = set(), set(), 0
    categories, managers, years = {}, {}, {}
    prefecture_rows = {}
    duplicates = []
    for manifest_path in manifests:
        meta = json.loads(manifest_path.read_text())
        stored = manifest_path.with_name(manifest_path.name.replace('.request.json', '.json.gz')).read_bytes()
        assert api.sha(stored) == meta['stored_sha256']
        body = gzip.decompress(stored)
        assert api.sha(body) == meta['response_sha256']
        obj = json.loads(body)
        assert len(obj['result']) == meta['rows']
        for record in obj['result']:
            flat = flatten(record)
            fields.update(flat)
            key = str(record['shisetsu_id'])
            if key in ids:
                duplicates.append(key)
            ids.add(key)
            for target, name in [(categories, 'tenken.kiroku.hantei_kubun'),
                                 (managers, 'syogen.kanrisya.kubun'), (years, 'tenken.nendo')]:
                value = str(flat.get(name))
                target[value] = target.get(value, 0) + 1
            rows += 1
            prefecture_rows[meta['pref']] = prefecture_rows.get(meta['pref'], 0) + 1
    assert rows == sum(result['downloaded_rows'] for result in results) == national_count
    assert not duplicates, 'Repeated record identities require review; source pages remain preserved.'
    index = OUT / 'pavement_basic_index.csv.gz'
    with gzip.open(index, 'wt', encoding='utf-8', newline='') as stream:
        names = ['source_prefecture', 'source_page', 'source_row'] + sorted(fields)
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for manifest_path in manifests:
            meta = json.loads(manifest_path.read_text())
            archive = manifest_path.with_name(manifest_path.name.replace('.request.json', '.json.gz'))
            with gzip.open(archive, 'rt', encoding='utf-8') as source:
                obj = json.load(source)
            for row, record in enumerate(obj['result']):
                writer.writerow({'source_prefecture': meta['pref'], 'source_page': str(archive.relative_to(ROOT)),
                                 'source_row': row, **flatten(record)})
    summary = {'status': 'passed_inventory_identity_and_counts', 'prefectures': len(prefecture_rows),
               'records': rows, 'unique_record_ids': len(ids), 'source_pages': len(manifests),
               'national_end_count': national_count, 'inspection_code_counts': categories,
               'administrator_category_counts': managers, 'inspection_year_counts': years,
               'index_fields': names, 'index_sha256': api.sha(index.read_bytes()),
               'index_path': str(index.relative_to(ROOT)), 'accepted_road_assignments': 0,
               'missing_condition_imputed': False, 'raw_or_processed_data_modified': False}
    api.write_json(OUT / 'validation.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    api.API = ENDPOINT
    api.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    api.write_json(OUT / 'acquisition_config.json', {
        'endpoint': ENDPOINT, 'page_size': SIZE, 'workers': 2,
        'minimum_request_start_interval_seconds': 1,
        'authorization': 'MILI-D-20261007-042',
        'scope': 'Latest public basic pavement records for prefectures 01-47, without authenticated APIs.',
        'no_policy_input_assignments': True})
    # Validate the pavement endpoint's pagination before the nationwide pass.
    first, meta = api.page('26', 0, SIZE)
    second, other = api.page('26', SIZE, SIZE)
    assert first['resultset']['count'] == second['resultset']['count']
    assert not ({row['shisetsu_id'] for row in first['result']} & {row['shisetsu_id'] for row in second['result']})
    api.write_json(OUT / 'pagination_validation.json', {'status': 'passed', 'prefecture': '26',
                  'page_size': SIZE, 'nonoverlapping_first_two_pages': True})
    del first, second
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(api.acquire_pref, f'{i:02d}', SIZE): f'{i:02d}' for i in range(1, 48)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({'prefecture': futures[future], 'error': str(exc)})
            api.write_json(OUT / 'download_progress.json', {'completed_prefectures': len(results),
                           'downloaded_rows_completed_prefectures': sum(r['downloaded_rows'] for r in results),
                           'results': sorted(results, key=lambda row: row['pref']), 'errors': errors,
                           'updated_utc': api.now()})
    assert not errors and len(results) == 47 and all(r['status'] == 'downloaded_count_reconciled' for r in results), errors
    body, obj, meta = api.request(ENDPOINT + '?limit=1&offset=0')
    meta.update({'url': ENDPOINT + '?limit=1&offset=0', 'response_sha256': api.sha(body)})
    api.write_json(OUT / 'national_end_count.json', obj)
    api.write_json(OUT / 'national_end_count.request.json', meta)
    audit_pages(results, int(obj['resultset']['count']))


if __name__ == '__main__':
    main()
