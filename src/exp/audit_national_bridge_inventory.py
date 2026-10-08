"""Audit saved national bridge source pages without accepting road assignments.

The CSV is an exploratory index with original dotted API names. Full source
records remain in the lossless source pages; flagged records are retained.
"""
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-inventory'
FIELDS = ['shisetsu_id', 'shisetsu_kubun', 'koushin_nichiji', 'kanrisya_code',
          'syogen.shisetsu.meisyou', 'syogen.rosen.meisyou',
          'syogen.kanrisya.kubun', 'syogen.kanrisya.meisyou', 'syogen.kanrisya.jimusyo',
          'syogen.gyousei_kuiki.todoufuken_code', 'syogen.gyousei_kuiki.todoufuken_mei',
          'syogen.gyousei_kuiki.shikuchouson_code', 'syogen.gyousei_kuiki.shikuchouson_mei',
          'syogen.ichi.ido', 'syogen.ichi.keido', 'syogen.kyouchou', 'syogen.fukuin',
          'syogen.kasetsu_nendo', 'tenken.nendo', 'tenken.kiroku.hantei_kubun',
          'tenken.syuzen.sochi_joukyou']


def get(record, path):
    value = record
    for key in path.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def absent(value):
    return value is None or str(value).strip() == ''


def source_text(value):
    return '' if value is None else str(value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def csv_file(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = json.loads((OUT / 'acquisition_config.json').read_text())
    progress = json.loads((OUT / 'download_progress.json').read_text())
    assert progress['completed_prefectures'] == 47 and not progress['errors']
    states = [json.loads(p.read_text()) for p in sorted((OUT / 'sources').glob('pref_*/acquisition_status.json'))]
    assert {s['pref'] for s in states} == {f'{i:02}' for i in range(1, 48)}
    assert all(s['status'] == 'downloaded_count_reconciled' for s in states)
    for state in states:
        directory = OUT / 'sources' / ('pref_' + state['pref'])
        end = json.loads((directory / 'end_count.json').read_text())
        assert end['resultset']['is_error'] is False
        assert int(end['resultset']['count']) == state['query_count_start'] == state['query_count_end']
    expected_counts = {s['pref']: s['downloaded_rows'] for s in states}
    start_count = json.loads((OUT / 'probes/national_count.json').read_text())['resultset']['count']
    end_count = json.loads((OUT / 'national_end_count.json').read_text())['resultset']['count']
    national_probe = (OUT / 'probes/national_count.json').read_bytes()
    national_probe_meta = json.loads((OUT / 'probes/national_count_request.json').read_text())
    assert sha(national_probe) == national_probe_meta['sha256']
    contract_hash = sha((ROOT / 'docs/AnaSOP.md').read_bytes())
    assert contract_hash == config['anasop_sha256']
    temp_db = OUT / 'identity_index.sqlite.part'
    if temp_db.exists():
        temp_db.unlink()
    db = sqlite3.connect(temp_db)
    db.execute('CREATE TABLE records (facility_id TEXT, record_sha TEXT, pref TEXT, source_page TEXT, source_row INTEGER)')
    counts = defaultdict(Counter)
    flag_counts, missing = Counter(), Counter()
    source_fields = Counter()
    page_manifest, prefecture_counts, times = [], Counter(), []
    flags_rows = 0
    current_year = datetime.now(timezone.utc).year

    def leaf_fields(record, prefix=''):
        for key, value in record.items():
            path = prefix + key
            if isinstance(value, dict):
                yield from leaf_fields(value, path + '.')
            else:
                yield path

    with gzip.open(OUT / 'bridge_basic_index.csv.gz', 'wt', encoding='utf-8', newline='') as inventory, \
         gzip.open(OUT / 'quality_flags.csv.gz', 'wt', encoding='utf-8', newline='') as flags:
        writer = csv.DictWriter(inventory, fieldnames=FIELDS + ['source_pref_query', 'source_page', 'source_row'])
        writer.writeheader()
        flag_writer = csv.DictWriter(flags, fieldnames=['shisetsu_id', 'source_pref_query', 'source_page', 'source_row', 'flags'])
        flag_writer.writeheader()
        for state in states:
            pref = state['pref']
            offsets = list(range(0, max(1, expected_counts[pref]), config['page_size']))
            for offset in offsets:
                page = OUT / 'sources' / ('pref_' + pref) / f'offset_{offset:06d}.json.gz'
                meta = json.loads(page.with_name(page.name.replace('.json.gz', '.request.json')).read_text())
                compressed = page.read_bytes()
                assert sha(compressed) == meta['stored_sha256']
                data = gzip.decompress(compressed)
                assert sha(data) == meta['response_sha256']
                obj = json.loads(data)
                assert obj['resultset']['is_error'] is False
                assert int(obj['resultset']['offset']) == offset
                assert int(obj['resultset']['count']) == expected_counts[pref]
                assert len(obj['result']) == min(config['page_size'], max(0, expected_counts[pref]-offset))
                relative = str(page.relative_to(OUT))
                page_manifest.append({'source_page': relative, **meta})
                times.append(meta['retrieved_utc'])
                inserts = []
                for i, record in enumerate(obj['result'], 1):
                    row = {field: get(record, field) for field in FIELDS}
                    row.update(source_pref_query=pref, source_page=relative, source_row=i)
                    writer.writerow(row)
                    source_fields.update(leaf_fields(record))
                    for field in FIELDS:
                        if absent(row[field]):
                            missing[field] += 1
                    facility = row['shisetsu_id']
                    f = []
                    if absent(facility):
                        f.append('missing_facility_id')
                        facility = None
                    latitude, longitude = numeric(row['syogen.ichi.ido']), numeric(row['syogen.ichi.keido'])
                    if latitude is None or longitude is None:
                        f.append('missing_or_nonfinite_coordinates')
                    elif not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                        f.append('invalid_geographic_coordinates')
                    elif not (20 <= latitude <= 46 and 122 <= longitude <= 154):
                        f.append('outside_broad_japan_screen')
                    record_pref = source_text(row['syogen.gyousei_kuiki.todoufuken_code'])
                    city = source_text(row['syogen.gyousei_kuiki.shikuchouson_code'])
                    if record_pref != pref:
                        f.append('pref_query_record_mismatch')
                    if not re.fullmatch(r'\d{6}', city) or not city.startswith(pref):
                        f.append('municipal_code_format_or_prefix')
                    year = source_text(row['tenken.nendo'])
                    if not re.fullmatch(r'\d{4}', year) or not 1900 <= int(year) <= current_year:
                        f.append('missing_or_invalid_inspection_year')
                    health = source_text(row['tenken.kiroku.hantei_kubun'])
                    action = str(row['tenken.syuzen.sochi_joukyou'] if row['tenken.syuzen.sochi_joukyou'] is not None else '')
                    if health not in ['1', '2', '3', '4', '8']:
                        f.append('missing_or_unrecognized_health_code')
                    if action not in ['0', '1', '2', '3', '8']:
                        f.append('missing_or_unrecognized_action_code')
                    for field, label in [('syogen.kyouchou', 'bridge_length'), ('syogen.fukuin', 'width')]:
                        value = numeric(row[field])
                        if value is None or value <= 0:
                            f.append('missing_or_nonpositive_' + label)
                    if str(row['shisetsu_kubun']) != '1':
                        f.append('unexpected_facility_type')
                    if f:
                        flags_rows += 1
                        flag_counts.update(f)
                        flag_writer.writerow({'shisetsu_id': facility, 'source_pref_query': pref,
                                               'source_page': relative, 'source_row': i, 'flags': ';'.join(f)})
                    counts['inspection_year'][year] += 1
                    counts['health_code'][health] += 1
                    counts['action_code'][action] += 1
                    counts['manager_category'][source_text(row['syogen.kanrisya.kubun'])] += 1
                    counts['municipality_code'][city] += 1
                    counts['record_prefecture'][record_pref] += 1
                    counts['update_month'][source_text(row['koushin_nichiji'])[:7]] += 1
                    prefecture_counts[pref] += 1
                    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
                    inserts.append((facility, sha(canonical), pref, relative, i))
                db.executemany('INSERT INTO records VALUES (?, ?, ?, ?, ?)', inserts)
                db.commit()
            assert prefecture_counts[pref] == expected_counts[pref]
            print(json.dumps({'audited_pref': pref, 'rows': prefecture_counts[pref]}), flush=True)

    db.execute('CREATE INDEX facility_index ON records(facility_id)')
    db.commit()
    total = db.execute('SELECT count(*) FROM records').fetchone()[0]
    unique = db.execute('SELECT count(DISTINCT facility_id) FROM records').fetchone()[0]
    missing_ids = db.execute('SELECT count(*) FROM records WHERE facility_id IS NULL').fetchone()[0]
    duplicates = list(db.execute('SELECT facility_id, count(*), count(DISTINCT record_sha), group_concat(DISTINCT pref) FROM records WHERE facility_id IS NOT NULL GROUP BY facility_id HAVING count(*)>1'))
    with gzip.open(OUT / 'bridge_basic_index.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
        saved_index_rows = sum(1 for _ in csv.DictReader(handle))
    with gzip.open(OUT / 'quality_flags.csv.gz', 'rt', encoding='utf-8', newline='') as handle:
        saved_flag_rows = 0
        for row in csv.DictReader(handle):
            assert db.execute('SELECT 1 FROM records WHERE facility_id=? AND source_page=? AND source_row=?',
                              (row['shisetsu_id'], row['source_page'], int(row['source_row']))).fetchone()
            saved_flag_rows += 1
    assert saved_index_rows == total and saved_flag_rows == flags_rows
    assert all(sum(counter.values()) == total for counter in counts.values())
    csv_file('duplicate_facility_ids.csv',
             [{'shisetsu_id': row[0], 'rows': row[1], 'distinct_payloads': row[2], 'query_prefectures': row[3]} for row in duplicates],
             ['shisetsu_id', 'rows', 'distinct_payloads', 'query_prefectures'])
    db.close()
    os.replace(temp_db, OUT / 'identity_index.sqlite')
    assert total == sum(expected_counts.values()) == sum(prefecture_counts.values())
    csv_file('page_manifest.csv', page_manifest, sorted({key for row in page_manifest for key in row}))
    csv_file('prefecture_coverage.csv', states, list(states[0]))
    csv_file('field_missingness.csv', [{'source_field': f, 'missing_rows': missing[f], 'total_rows': total} for f in FIELDS], ['source_field', 'missing_rows', 'total_rows'])
    csv_file('source_field_presence.csv', [{'source_field': f, 'records_with_field': n} for f,n in sorted(source_fields.items())], ['source_field', 'records_with_field'])
    for name, counter in counts.items():
        csv_file(name + '_counts.csv', [{'source_value': value, 'rows': n} for value,n in sorted(counter.items())], ['source_value', 'rows'])
    # Replace this script's superseded per-second timestamp frequency output.
    obsolete_timestamp_table = OUT / 'update_datetime_counts.csv'
    if obsolete_timestamp_table.exists():
        obsolete_timestamp_table.unlink()
    csv_file('quality_flag_counts.csv', [{'flag': f, 'rows': n} for f,n in sorted(flag_counts.items())], ['flag', 'rows'])
    validation = {
        'status': 'passed' if total == unique == start_count == end_count and missing_ids == 0 else 'review_required',
        'downloaded_prefectures': 47, 'source_pages': len(page_manifest), 'source_rows': total,
        'unique_facility_ids': unique, 'missing_facility_ids': missing_ids,
        'duplicate_facility_ids': len(duplicates), 'conflicting_duplicate_ids': sum(row[2]>1 for row in duplicates),
        'national_query_count_start': start_count, 'national_query_count_end': end_count,
        'rows_reconcile_to_national_query': total == start_count == end_count,
        'acquisition_complete': total == start_count == end_count,
        'flagged_rows_retained': flags_rows, 'quality_flag_counts': dict(flag_counts),
        'inspection_year_counts': dict(counts['inspection_year']), 'health_code_counts': dict(counts['health_code']),
        'manager_category_counts': dict(counts['manager_category']),
        'distinct_municipal_code_source_values_including_blank': len(counts['municipality_code']),
        'distinct_nonempty_municipal_code_source_values': sum(bool(value) for value in counts['municipality_code']),
        'saved_index_rows_verified': saved_index_rows, 'saved_flag_rows_and_source_references_verified': saved_flag_rows,
        'earliest_page_retrieved_utc': min(times), 'latest_page_retrieved_utc': max(times),
        'source_and_stored_page_hashes_verified': True, 'anasop_unchanged': True,
        'anasop_sha256': contract_hash, 'accepted_road_matches': 0,
        'simultaneous_condition_snapshot': False, 'real_world_complete_coverage_proven': False,
        'source_attribution': '全国道路施設点検データベース',
    }
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    years = ', '.join(f'{y}: {n:,}' for y,n in sorted(counts['inspection_year'].items()))
    report = [
        '# National public basic bridge inventory', '',
        f'Review date: 2026-10-06. Source attribution: 全国道路施設点検データベース. Research contract: docs/AnaSOP.md Sections 1-2; national all-road scope retained. This is an acquired source inventory and quality diagnostic, not a completed road crosswalk or maintenance-allocation dataset.', '',
        '## Acquisition and reconciliation', '',
        f'Acquired {total:,} source rows in {len(page_manifest)} losslessly compressed pages across 47 prefecture queries. There are {unique:,} distinct nonmissing facility IDs, {missing_ids:,} missing IDs, {len(duplicates):,} repeated IDs and {sum(row[2]>1 for row in duplicates):,} IDs with different payloads. No source rows were removed or resolved by arbitrary deduplication.', '',
        f'The national latest endpoint reported {start_count:,} rows before acquisition and {end_count:,} after it. Prefecture start/end counts and paginated row counts were checked. Counts reconcile to the national query: {validation["rows_reconcile_to_national_query"]}. This verifies acquisition against the exposed query, not completeness against every physical bridge in Japan. Equal counts cannot prove an atomic or unchanged database snapshot.', '',
        f'Source pages were retrieved from {min(times)} to {max(times)}. The downloader uses at most two concurrent requests, a one-second minimum between request starts, bounded 5,000-record pages, retry/backoff for transient failures, source hashes and resumable caches. Only the documented unauthenticated free basic endpoint was used.', '',
        '## Quality and temporal coverage', '',
        f'{flags_rows:,} rows carry one or more diagnostic flags and are retained in the inventory. Flag categories can overlap. Unknown/invalid values are not recoded as good condition, absent bridges or zero values.', '',
        '| Diagnostic flag | Rows |', '| --- | ---: |',
    ]
    report += [f'| {f} | {n:,} |' for f,n in sorted(flag_counts.items())]
    if not flag_counts:
        report.append('| No listed field-screening flags | 0 |')
    report += ['', f'Inspection-year source values and counts: {years}. Latest records are not a common inspection year. Health code 8 denotes uninspected records; health categories are not road closures or failure probabilities. Action status is not a repair cost or an estimated maintenance effect.', '',
               'Municipal codes are screened as six-digit source values with a matching prefecture prefix; this is not a validated historical municipal-boundary concordance. Coordinates are checked for numeric validity and a deliberately broad Japan bounding box; this is not land-boundary validation, a datum determination or accepted road matching. The coordinate datum remains unverified from explicit source metadata. The previous ten-case crosswalk review remains applicable: short bridges can occupy ordinary-road features, distinct assets can share a source feature, and old bridges require conflict review. A health/action code absent from the saved basic dictionary is marked unrecognized, not invalidated or assigned a closure meaning.', '',
               '## Files and reproducibility', '',
               '- sources/pref_XX/: unchanged response content in gzip pages, request metadata and end-count evidence.',
               '- bridge_basic_index.csv.gz: exploratory index using original dotted API field names, plus source-page/row references. Full source records remain in pages; this index is not final variable construction.',
               '- identity_index.sqlite and duplicate_facility_ids.csv: reproducible identity audit without silently selecting a duplicate.',
               '- quality_flags.csv.gz and quality_flag_counts.csv: record-level and aggregate diagnostic flags.',
               '- prefecture_coverage.csv, inspection_year_counts.csv, health_code_counts.csv, action_code_counts.csv, manager_category_counts.csv, municipality_code_counts.csv: source coverage counts.',
               '- update_month_counts.csv: source update timestamps summarized by month for provenance review; exact timestamps remain in the source pages and exploratory index.',
               '- field_missingness.csv, source_field_presence.csv, page_manifest.csv and validation.json: field, provenance and reconciliation checks.', '',
               '```bash', '.venv/bin/python src/exp/probe_national_bridge_pagination.py', '.venv/bin/python src/exp/fetch_national_bridge_inventory.py', '.venv/bin/python src/exp/audit_national_bridge_inventory.py', '```', '',
               'Downloading requires network access; the audit runs locally using the Python standard library. Cached source pages are hash-checked and reused on rerun. No raw data or AnaSOP was modified, no dependencies were installed, and no Git or DVC operation was performed.', '',
               '## Next research step', '',
               'Develop and validate a national candidate bridge-to-road crosswalk that retains withheld and unmatched records and shared-feature relationships. Confirm physical asset extent and affected graph links before accepting maintenance units. National traffic coverage, common intervention costs and maintenance effects remain prerequisites for the three-rule budget comparison. Acquisition does not approve universal closure, equal costs, or a bridge-only research scope.', '',
               '## Primary sources', '',
               '- [Free basic API specification](https://rirs.or.jp/tenken-db/pdf/view/api_specification.pdf): endpoint, latest path spelling lastest, prefecture filters and limit/offset pagination.',
               '- [Basic parameter dictionary](https://rirs.or.jp/tenken-db/pdf/view/00_Parameters_base.pdf): source identity, coordinates, dimensions, manager, inspection and action categories.',
               '- [Published access and reuse terms](https://rirs.or.jp/tenken-db/pdf/view/etsuran_kiyaku.pdf): free basic data in Appendix 1 and secondary-use attribution in Article 11. Documentation copies were retained in the preceding bridge-access-rural-sample/sources directory.', '']
    (OUT / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
