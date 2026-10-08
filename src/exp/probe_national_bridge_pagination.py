"""Check national count and bounded latest-record pagination through the free API."""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-inventory/probes'
API = 'https://road-structures-db.mlit.go.jp/xROAD/api/v1/bridges/lastest'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, query in [('national_count', 'limit=1&offset=0'),
                        ('pref09_page0', 'pref=09&limit=5000&offset=0'),
                        ('pref09_page1', 'pref=09&limit=5000&offset=5000')]:
        url = API + '?' + query
        start = time.monotonic()
        path = OUT / (name + '.json')
        meta = OUT / (name + '_request.json')
        if path.exists() and meta.exists():
            data = path.read_bytes()
            manifest = json.loads(meta.read_text())
            assert hashlib.sha256(data).hexdigest() == manifest['sha256']
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            manifest = {'url': url, 'sha256': hashlib.sha256(data).hexdigest(),
                        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                        'bytes': len(data), 'elapsed_s': round(time.monotonic()-start, 2)}
            path.write_bytes(data)
            meta.write_text(json.dumps(manifest, indent=2) + '\n')
        obj = json.loads(data)
        assert obj['resultset']['is_error'] is False
        print(json.dumps({'probe': name, 'resultset': obj['resultset'],
                          'returned': len(obj['result']), **manifest}), flush=True)
        time.sleep(1)


if __name__ == '__main__':
    main()
