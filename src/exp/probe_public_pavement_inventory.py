"""Bounded public pavement-API feasibility probe; no analysis-variable adoption."""
import argparse
import csv
import hashlib
import json
import urllib.error
import urllib.request
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/public_api_probe'
URL = 'https://road-structures-db.mlit.go.jp/xROAD/api/v1/pavements/lastest?limit=10&offset=0'


def summarize_saved_probe():
    sample = json.loads((OUT / 'sample_response.json').read_text())['result']
    def flatten(value, prefix=''):
        result = {}
        for key, item in value.items():
            name = prefix + '.' + key if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, name))
            else:
                result[name] = item
        return result
    rows = [flatten(record) for record in sample]
    fields = sorted(set().union(*(record.keys() for record in rows)))
    with (OUT / 'sample_records.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / 'field_availability.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_field', 'present_records', 'nonmissing_records'])
        writer.writeheader()
        for field in fields:
            writer.writerow({'source_field': field, 'present_records': sum(field in row for row in rows),
                             'nonmissing_records': sum(row.get(field) is not None and row.get(field) != '' for row in rows)})
    if (OUT / 'prefecture_requests.json').exists():
        requests = json.loads((OUT / 'prefecture_requests.json').read_text())
        nationwide = json.loads((OUT / 'request.json').read_text())
        validation = {'status': 'passed', 'prefectures_checked': len(requests),
                      'reported_national_records': nationwide['resultset']['count'],
                      'reported_prefecture_sum': sum(row.get('count') or 0 for row in requests),
                      'all_requests_successful': all(row.get('is_error') is False for row in requests),
                      'all_prefectures_nonempty': all((row.get('count') or 0) > 0 for row in requests),
                      'sample_prefecture_matches_filter': all(row['prefecture'] == row.get('sample_prefecture') for row in requests),
                      'all_response_hashes_valid': all(hashlib.sha256((OUT / ('pref_' + row['prefecture'] + '.json')).read_bytes()).hexdigest() == row.get('sha256') for row in requests),
                      'complete_inventory_downloaded': False,
                      'road_correspondence_or_pavement_variables_adopted': False}
        assert validation['prefectures_checked'] == 47
        assert validation['reported_national_records'] == validation['reported_prefecture_sum']
        assert all(validation[key] for key in ['all_requests_successful', 'all_prefectures_nonempty', 'sample_prefecture_matches_filter', 'all_response_hashes_valid'])
        (OUT / 'coverage_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
        print(json.dumps(validation))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize-only', action='store_true', help='Rebuild local probe tables without network requests.')
    parser.add_argument('--prefecture-counts', action='store_true',
                        help='At most 47 additional public requests, one sample per prefecture.')
    args = parser.parse_args()
    if args.summarize_only:
        summarize_saved_probe()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    info = {'url': URL, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'One public latest-record request, at most ten records; no authentication.'}
    try:
        req = urllib.request.Request(URL, headers={'User-Agent': 'TR01-research-pavement-feasibility/1.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            info['http_status'] = response.status
        (OUT / 'sample_response.json').write_bytes(body)
        info['sha256'] = hashlib.sha256(body).hexdigest()
        obj = json.loads(body)
        info['top_level_keys'] = list(obj)
        info['resultset'] = obj.get('resultset')
        records = obj.get('result', [])
        info['sample_size'] = len(records) if isinstance(records, list) else None
        info['usable_response'] = (obj.get('resultset', {}).get('is_error') is False
                                   and isinstance(records, list))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        info['error'] = str(exc)
        info['usable_response'] = False
    (OUT / 'request.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(info, ensure_ascii=False))
    if args.prefecture_counts and info['usable_response']:
        rows = []
        for number in range(1, 48):
            time.sleep(1)
            pref = f'{number:02d}'
            url = URL.replace('limit=10', 'limit=1') + '&pref=' + pref
            row = {'prefecture': pref, 'url': url}
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'TR01-research-pavement-feasibility/1.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    body = response.read()
                    row['http_status'] = response.status
                (OUT / f'pref_{pref}.json').write_bytes(body)
                obj = json.loads(body)
                row['sha256'] = hashlib.sha256(body).hexdigest()
                row['count'] = obj.get('resultset', {}).get('count')
                row['is_error'] = obj.get('resultset', {}).get('is_error')
                records = obj.get('result', [])
                row['sample_size'] = len(records)
                row['sample_prefecture'] = (records[0].get('syogen', {}).get('gyousei_kuiki', {})
                                             .get('todoufuken_code')) if records else None
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                row['error'] = str(exc)
            row['retrieved_utc'] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
            (OUT / 'prefecture_requests.json').write_text(json.dumps(rows, indent=2) + '\n')
        fields = ['prefecture', 'count', 'sample_size', 'sample_prefecture', 'is_error', 'error']
        with (OUT / 'prefecture_availability.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({'prefecture_requests': len(rows),
                          'successful_prefectures': sum(r.get('is_error') is False for r in rows),
                          'reported_record_sum': sum(r.get('count') or 0 for r in rows)}))
        summarize_saved_probe()


if __name__ == '__main__':
    main()
