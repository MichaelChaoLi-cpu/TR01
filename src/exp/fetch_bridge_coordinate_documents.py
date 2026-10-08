"""Preserve bounded independent official documents for coordinate review."""
import hashlib
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-crosswalk/coordinate-review/sources'
DOCUMENTS = {
    'hitachiomiya_bridge_inventory.pdf': 'https://www.city.hitachiomiya.lg.jp/data/doc/1650968713_doc_105_2.pdf',
    'ikeda_bridge_plan.pdf': 'https://www.town.gifu-ikeda.lg.jp/cmsfiles/contents/0000001/1320/kensetsu.pdf',
    'komatsushima_bridge_plan.pdf': 'https://www.city.komatsushima.lg.jp/fs/2/9/9/4/_/4209.pdf',
    'nishihara_bridge_plan.pdf': 'https://www.vill.nishihara.kumamoto.jp/kiji003275/3_275_237_up_x3ynlhh1.pdf',
    'kochi_heritage_bridge_review.pdf': 'https://www.pref.kochi.lg.jp/doc/tyouzyumyukakeikaku/file_contents/dai1kaikaigisiryou.pdf',
    'sabae_bridge_tender.pdf': 'https://www.city.sabae.fukui.jp/about_city/bid/shimeikyosonyusatsu/reiwa2/Kkanri0120200619121.files/20200702-07_12koukoku.pdf',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, url in DOCUMENTS.items():
        path = OUT / name
        receipt_path = path.with_suffix('.request.json')
        if path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt['url'] == url
            assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
        else:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read(20 * 1024 * 1024 + 1)
            except (urllib.error.URLError, TimeoutError) as error:
                failed = {'file': name, 'url': url, 'status': 'download_failed',
                          'error': str(error), 'attempted_utc': datetime.now(timezone.utc).isoformat()}
                manifest.append(failed)
                print(json.dumps(failed), flush=True)
                time.sleep(1)
                continue
            assert len(data) <= 20 * 1024 * 1024 and data.startswith(b'%PDF')
            receipt = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                       'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            path.write_bytes(data)
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        manifest.append({'file': name, **receipt})
        print(json.dumps({'file': name, 'bytes': receipt['bytes']}), flush=True)
        time.sleep(1)
    (OUT.parent / 'document_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
