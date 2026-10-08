"""Resumable acquisition of advertised official section GeoJSON geometry.

Every table section is requested; missing sections are recorded, never inferred.
Cache files are validated before reuse. Acquisition does not assign road traffic.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/national-traffic-geometry'
BASE = 'https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/'
LOCAL = threading.local()


def fetch(section):
    section = str(section)
    url = BASE + f'census/{section[:2]}/{section}.geojson'
    path = OUT / 'sources' / section[:2] / f'{section}.geojson'
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            cached = path.exists()
            if cached:
                data = path.read_bytes()
            else:
                # Reuse one validated HTTPS connection per worker instead of
                # performing a new TLS handshake for every small section file.
                if not getattr(LOCAL, 'connection', None):
                    LOCAL.connection = http.client.HTTPSConnection('www.mlit.go.jp', timeout=30)
                LOCAL.connection.request('GET', url.split('www.mlit.go.jp', 1)[1],
                                         headers={'User-Agent': 'TR01 academic research source acquisition'})
                response = LOCAL.connection.getresponse()
                data = response.read()
                time.sleep(0.03)
                if response.status != 200:
                    raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
            obj = json.loads(data)
            assert obj['type'] == 'FeatureCollection'
            features = obj['features']
            types = sorted(set(f['geometry']['type'] for f in features if f.get('geometry')))
            assert all(t in ['LineString', 'MultiLineString'] for t in types)
            source_ids = sorted(set(str(f['properties']['R03cen']).zfill(11) for f in features))
            assert not source_ids or source_ids == [section]
            if not cached:
                part = path.with_suffix('.part'); part.write_bytes(data); part.replace(path)
            return {'Traffic Geometry Section': section, 'URL': url,
                    'Path': str(path.relative_to(ROOT)), 'Status': 'available' if features else 'empty',
                    'Feature Count': len(features), 'SHA256': hashlib.sha256(data).hexdigest(),
                    'Bytes': len(data), 'Error': ''}
        except urllib.error.HTTPError as error:
            if error.code in [404, 410]:
                return {'Traffic Geometry Section': section, 'URL': url, 'Status': 'not_found', 'Error': str(error)}
            message = repr(error)
        except Exception as error:
            message = repr(error)
            if path.exists():
                # An invalid cache is not silently overwritten.
                return {'Traffic Geometry Section': section, 'URL': url, 'Status': 'invalid_cache', 'Error': message}
        if getattr(LOCAL, 'connection', None):
            LOCAL.connection.close(); LOCAL.connection = None
        time.sleep(0.5 * (attempt + 1))
    return {'Traffic Geometry Section': section, 'URL': url, 'Status': 'failed', 'Error': message}


def checkpoint(rows, total, complete=False):
    frame = pd.DataFrame(rows).sort_values('Traffic Geometry Section')
    manifest = OUT / 'manifest.csv'; part = OUT / f'manifest.{os.getpid()}.part'
    frame.to_csv(part, index=False); part.replace(manifest)
    receipt = {'requested_sections': total, 'processed_sections': len(rows),
               'statuses': frame['Status'].value_counts().to_dict(), 'acquisition_pass_complete': complete,
               'national_geometry_complete': bool(complete and frame['Status'].eq('available').all()),
               'graph_assignments': 0, 'source_year': 2021,
               'geometry_route': 'Section URL pattern advertised by the official visualization HTML',
               'measurement_unit_limit': 'Geometry sections are not independent traffic observations.'}
    progress = OUT / f'validation.{os.getpid()}.part'
    progress.write_text(json.dumps(receipt, indent=2) + '\n')
    progress.replace(OUT / 'validation.json')
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int); args = parser.parse_args()
    assert 1 <= args.workers <= 8
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'acquisition.pid'
    if lock.exists():
        previous = int(lock.read_text())
        try: os.kill(previous, 0)
        except ProcessLookupError: pass
        else: raise RuntimeError(f'An acquisition process already holds this output: {previous}')
    lock.write_text(str(os.getpid()))
    df = pd.read_parquet(ROOT / 'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet')
    sections = sorted(df['Traffic Geometry Section'].unique())
    if args.limit is not None: sections = sections[:args.limit]
    rows = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(fetch, section): section for section in sections}
            for future in as_completed(tasks):
                rows.append(future.result())
                if len(rows) % 500 == 0: checkpoint(rows, len(sections))
        checkpoint(rows, len(sections), complete=True)
    finally:
        if lock.exists() and int(lock.read_text()) == os.getpid(): lock.unlink()


if __name__ == '__main__': main()
