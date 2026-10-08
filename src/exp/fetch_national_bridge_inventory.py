"""Resumable, rate-limited acquisition of public basic latest bridge records.

Writes only exploratory evidence. Source pages are losslessly gzip-compressed;
no records are dropped or assigned to roads. Paid/private APIs are not used.
"""
import argparse
import gzip
import hashlib
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-inventory'
API = 'https://road-structures-db.mlit.go.jp/xROAD/api/v1/bridges/lastest'
LOCK = threading.Lock()
LAST_START = 0.0


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def request(url):
    global LAST_START
    for attempt in range(4):
        with LOCK:
            time.sleep(max(0, 1.0 - (time.monotonic() - LAST_START)))
            LAST_START = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'TR01-research-basic-inventory/1.0'})
            started = now()
            with urllib.request.urlopen(req, timeout=90) as response:
                data = response.read()
                info = {'http_status': response.status, 'started_utc': started,
                        'retrieved_utc': now(), 'etag': response.headers.get('ETag'),
                        'last_modified': response.headers.get('Last-Modified')}
            obj = json.loads(data)
            if obj.get('resultset', {}).get('is_error') is not False:
                raise ValueError('API error response; no schema or authentication workaround attempted')
            if not isinstance(obj.get('result'), list):
                raise ValueError('Unexpected result schema')
            return data, obj, info
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            try:
                delay = min(60, max(5, int(exc.headers.get('Retry-After', 5 * 2**attempt))))
            except ValueError:
                delay = 5 * 2**attempt
            time.sleep(delay)
        except (urllib.error.URLError, socket.timeout, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError('Request attempts exhausted')


def page(pref, offset, size):
    directory = OUT / 'sources' / ('pref_' + pref)
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f'offset_{offset:06d}'
    archive = base.with_suffix('.json.gz')
    metadata = base.with_suffix('.request.json')
    query = {'pref': pref, 'limit': size, 'offset': offset}
    url = API + '?' + urllib.parse.urlencode(query)
    if archive.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        assert info['url'] == url
        compressed = archive.read_bytes()
        assert sha(compressed) == info['stored_sha256']
        data = gzip.decompress(compressed)
        assert sha(data) == info['response_sha256']
        obj = json.loads(data)
    else:
        # Reuse the verified bounded Tochigi probe rather than downloading it twice.
        probe = OUT / 'probes' / ('pref09_page' + str(offset // size))
        if pref == '09' and size == 5000 and offset in (0, 5000) and probe.with_suffix('.json').exists():
            data = probe.with_suffix('.json').read_bytes()
            old = json.loads((OUT / 'probes' / (probe.name + '_request.json')).read_text())
            assert old['url'] == url and sha(data) == old['sha256']
            obj = json.loads(data)
            info = {'http_status': 200, 'retrieved_utc': old['retrieved_utc'], 'reused_probe': True}
        else:
            data, obj, info = request(url)
        resultset = obj['resultset']
        assert resultset['is_error'] is False
        assert int(resultset['offset']) == offset
        assert int(resultset['limit']) == size
        expected = min(size, max(0, int(resultset['count']) - offset))
        if len(obj['result']) != expected:
            raise ValueError(f'Unexpected page length: pref={pref}, offset={offset}, expected={expected}, got={len(obj["result"])}')
        compressed = gzip.compress(data, compresslevel=6, mtime=0)
        temporary = archive.with_suffix('.part')
        temporary.write_bytes(compressed)
        os.replace(temporary, archive)
        info.update({'url': url, 'pref': pref, 'offset': offset, 'requested_limit': size,
                     'query_count': int(resultset['count']), 'rows': len(obj['result']),
                     'response_sha256': sha(data), 'stored_sha256': sha(compressed),
                     'response_bytes': len(data), 'stored_bytes': len(compressed)})
        write_json(metadata, info)
    return obj, info


def acquire_pref(pref, size):
    first, info = page(pref, 0, size)
    count = int(first['resultset']['count'])
    rows = len(first['result'])
    pages = 1
    del first
    for offset in range(size, count, size):
        obj, info = page(pref, offset, size)
        if int(obj['resultset']['count']) != count:
            raise ValueError(f'Count changed during acquisition for pref {pref}; preserve evidence and review before completion')
        rows += len(obj['result'])
        pages += 1
        del obj
    # Check a fresh count at the end; saved pages themselves remain immutable.
    url = API + '?' + urllib.parse.urlencode({'pref': pref, 'limit': 1, 'offset': 0})
    data, obj, final = request(url)
    final.update({'url': url, 'response_sha256': sha(data), 'query_count': int(obj['resultset']['count'])})
    write_json(OUT / 'sources' / ('pref_' + pref) / 'end_count.request.json', final)
    write_json(OUT / 'sources' / ('pref_' + pref) / 'end_count.json', obj)
    status = 'downloaded_count_reconciled' if rows == count == final['query_count'] else 'count_changed'
    result = {'pref': pref, 'status': status, 'query_count_start': count,
              'query_count_end': final['query_count'], 'downloaded_rows': rows, 'pages': pages}
    write_json(OUT / 'sources' / ('pref_' + pref) / 'acquisition_status.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2, choices=[1, 2])
    parser.add_argument('--page-size', type=int, default=5000)
    args = parser.parse_args()
    if args.page_size != 5000:
        raise ValueError('Only the verified 5000-record page size is enabled')
    OUT.mkdir(parents=True, exist_ok=True)
    contract = ROOT / 'docs/AnaSOP.md'
    contract_hash = sha(contract.read_bytes())
    write_json(OUT / 'acquisition_config.json', {'endpoint': API, 'workers': args.workers,
               'page_size': args.page_size, 'minimum_request_start_interval_seconds': 1,
               'source_attribution': '全国道路施設点検データベース',
               'scope': 'Latest public basic bridge records for prefecture codes 01-47; preserve all source rows.',
               'anasop_sha256': contract_hash})
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire_pref, f'{i:02}', args.page_size): f'{i:02}' for i in range(1, 48)}
        for future in as_completed(futures):
            pref = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                error = {'pref': pref, 'error_type': type(exc).__name__, 'message': str(exc)}
                errors.append(error)
                print(json.dumps(error), flush=True)
            write_json(OUT / 'download_progress.json', {'completed_prefectures': len(results),
                       'downloaded_rows_completed_prefectures': sum(r['downloaded_rows'] for r in results),
                       'results': sorted(results, key=lambda r:r['pref']), 'errors': errors,
                       'updated_utc': now()})
    data, obj, meta = request(API + '?limit=1&offset=0')
    meta.update({'url': API + '?limit=1&offset=0', 'response_sha256': sha(data)})
    write_json(OUT / 'national_end_count.json', obj)
    write_json(OUT / 'national_end_count.request.json', meta)
    assert sha(contract.read_bytes()) == contract_hash
    if errors or len(results) != 47 or any(r['status'] != 'downloaded_count_reconciled' for r in results):
        raise RuntimeError('Acquisition incomplete or counts changed; cached evidence is resumable')
    print(json.dumps({'status': 'download_complete_pending_unique_id_and_quality_audit',
                      'prefectures': 47, 'downloaded_rows': sum(r['downloaded_rows'] for r in results),
                      'national_end_query_count': obj['resultset']['count']}), flush=True)


if __name__ == '__main__':
    main()
