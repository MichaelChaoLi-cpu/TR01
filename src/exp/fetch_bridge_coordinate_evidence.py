"""Fetch bounded historical basic responses for the fifteen withheld assets.

Historical responses are evidence for review, never automatic corrections.
"""
import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-crosswalk/coordinate-review'
API = 'https://road-structures-db.mlit.go.jp/xROAD/api/v1/bridges'


def main():
    (OUT / 'sources').mkdir(parents=True, exist_ok=True)
    with (OUT.parent / 'coordinate_withheld.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    manifest = []
    for i, row in enumerate(rows, 1):
        name = f'C{i:02}_history.json'
        url = API + '?' + urllib.parse.urlencode({'shisetsu': row['shisetsu_id'], 'limit': 100, 'offset': 0})
        path = OUT / 'sources' / name
        metadata_path = path.with_suffix('.request.json')
        if path.exists() and metadata_path.exists():
            data = path.read_bytes(); metadata = json.loads(metadata_path.read_text())
            assert hashlib.sha256(data).hexdigest() == metadata['sha256'] and metadata['url'] == url
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            metadata = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            path.write_bytes(data)
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        obj = json.loads(data)
        assert obj['resultset']['is_error'] is False
        assert int(obj['resultset']['count']) <= 100, 'Bounded response truncated; retain pending rather than omit history'
        item = {'review_id': f'C{i:02}', 'shisetsu_id': row['shisetsu_id'], 'source_file': name,
                'returned_rows': len(obj['result']), **metadata}
        manifest.append(item)
        locations = [{'year': r.get('tenken', {}).get('nendo'),
                      'location': r.get('syogen', {}).get('ichi'),
                      'name': r.get('syogen', {}).get('shisetsu', {}).get('meisyou')}
                     for r in obj['result']]
        print(json.dumps({'review_id': item['review_id'], 'rows': item['returned_rows'], 'locations': locations}, ensure_ascii=False), flush=True)
        time.sleep(1)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
