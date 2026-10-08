"""Report retained national candidates and independent coordinate evidence.

No coordinate edits, accepted assignments, AnaSOP edits or policy estimates.
"""
import csv
import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/national-bridge-crosswalk'
COORD = OUT / 'coordinate-review'


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    national = read_json(OUT / 'validation.json')
    assert national['status'] == 'passed'
    with (OUT / 'coordinate_withheld.csv').open() as handle:
        withheld = list(csv.DictReader(handle))
    assert len(withheld) == 15
    api_manifest = read_json(COORD / 'manifest.json')
    document_manifest = read_json(COORD / 'document_manifest.json')
    for item in document_manifest:
        if 'sha256' in item:
            assert hashlib.sha256((COORD / 'sources' / item['file']).read_bytes()).hexdigest() == item['sha256']
    # Exact source row and page are checked; decimals come from DMS, never the asset ID.
    text = (COORD / 'sources/hitachiomiya_bridge_inventory.txt').read_text()
    page7 = text.split('\f')[6]
    exact = next(line for line in page7.splitlines() if '469' in line and '42934' in line)
    assert all(token in exact for token in ['50042-1号橋', '9.1', '2.3', '36°37′29″', '140°17′14″'])
    normalized_ikeda = unicodedata.normalize('NFKC', (COORD / 'sources/ikeda_bridge_plan.txt').read_text().split('\f')[8])
    ikeda_row = next(line for line in normalized_ikeda.splitlines() if '北村11号橋' in line)
    assert '上田57号線' in ikeda_row and '2.2' in ikeda_row and '5.5' in ikeda_row
    assert all((COORD / name).exists() for name in ['hitachiomiya_page_7.png', 'ikeda_page_9.png', 'kochi_page_17.png'])
    reviews = []
    connection = sqlite3.connect(f'file:{OUT / "candidate_summary.sqlite"}?mode=ro', uri=True)
    for i, source in enumerate(withheld, 1):
        review_id = f'C{i:02}'
        manifest = next(m for m in api_manifest if m['review_id'] == review_id)
        path = COORD / 'sources' / manifest['source_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
        payload = read_json(path)
        assert payload['resultset']['count'] == 1 and len(payload['result']) == 1
        asset = payload['result'][0]
        assert asset['shisetsu_id'] == source['shisetsu_id']
        position = asset['syogen']['ichi']
        assert float(source['source_latitude']) == float(position['ido'])
        assert float(source['source_longitude']) == float(position['keido'])
        accepted, candidate_status = connection.execute('SELECT accepted_match,candidate_status FROM bridge_summary WHERE shisetsu_id=?', (source['shisetsu_id'],)).fetchone()
        assert accepted == '0' and candidate_status == 'coordinate_withheld'
        evidence = ''; locator = ''; finding = 'No independently verified alternative coordinate obtained in this review.'
        if review_id == 'C02':
            evidence = 'sources/hitachiomiya_bridge_inventory.pdf'; locator = 'PDF page 7, row 469, registry 42934'
            finding = 'Official DMS alternative found; normalized name/route agree. Length/width differ: municipal 9.1/2.3 m versus API 9.0/1.8 m. Datum, temporal correspondence and dimensions require reconciliation; correction withheld.'
        elif review_id == 'C05':
            evidence = 'sources/ikeda_bridge_plan.pdf'; locator = 'PDF page 9, Kitamura 11 row'
            finding = 'Name, route and dimensions 2.2/5.5 m agree with API. This table does not supply a replacement coordinate; correction withheld.'
        elif review_id == 'C14':
            evidence = 'sources/kochi_heritage_bridge_review.pdf'; locator = 'PDF page 17, heritage table row 13'
            finding = 'Name and Kitagawa locality/road context corroborated; document does not supply a replacement coordinate or establish current graph correspondence.'
        elif review_id in ['C07', 'C08', 'C09', 'C10', 'C11']:
            evidence = 'sources/komatsushima_bridge_plan.pdf'; locator = 'All extracted pages searched using normalized names'
            finding = 'Retrieved municipal general plan supplies no identifiable exact coordinate for these named assets. Absence of a text hit does not prove an asset is absent.'
        elif review_id == 'C06':
            finding = 'API has no asset/route name and zero coordinates; identity and replacement position remain unresolved.'
        elif review_id == 'C15':
            finding = 'Candidate municipal plan link returned HTTP 404. No independently verified replacement coordinate obtained.'
        reviews.append({'review_id': review_id, 'bridge_index': source['bridge_index'],
                        'shisetsu_id': source['shisetsu_id'], 'bridge_name': asset['syogen']['shisetsu']['meisyou'],
                        'route_name': asset['syogen']['rosen']['meisyou'],
                        'source_longitude': source['source_longitude'], 'source_latitude': source['source_latitude'],
                        'api_returned_rows': 1, 'anomaly_repeated_in_basic_response': True,
                        'api_response_file': 'sources/' + manifest['source_file'],
                        'independent_document': evidence, 'document_locator': locator,
                        'finding': finding, 'coordinate_correction_accepted': False,
                        'primary_candidate_status': 'coordinate_withheld', 'original_values_retained': True})
    connection.close()
    write_csv(COORD / 'coordinate_review.csv', reviews)
    proposal = {'review_id': 'C02', 'shisetsu_id': withheld[1]['shisetsu_id'],
                'source_document': 'sources/hitachiomiya_bridge_inventory.pdf',
                'source_url': next(d['url'] for d in document_manifest if d['file'] == 'hitachiomiya_bridge_inventory.pdf'),
                'pdf_page': 7, 'table_row': 469, 'registry_number': 42934,
                'source_latitude_dms': '36°37′29″', 'source_longitude_dms': '140°17′14″',
                'alternative_latitude': 36 + 37/60 + 29/3600,
                'alternative_longitude': 140 + 17/60 + 14/3600,
                'conversion': 'degrees + minutes/60 + seconds/3600',
                'source_coordinate_resolution_arcseconds': 1,
                'document_length_m': 9.1, 'document_width_m': 2.3,
                'api_length_m': 9.0, 'api_width_m': 1.8,
                'datum': 'unverified', 'temporal_correspondence': 'unverified',
                'status': 'proposal_withheld_pending_identity_and_dimension_reconciliation',
                'accepted': False, 'used_in_primary_candidate_run': False}
    write_csv(COORD / 'alternative_coordinate_proposals.csv', [proposal])
    evidence = {'status': 'passed', 'source_anomalies_retained': 15,
                'basic_response_requests': 15, 'requests_returning_one_row': 15,
                'same_anomaly_repeated': 15, 'independent_coordinate_proposals': 1,
                'coordinate_corrections_accepted': 0, 'primary_records_still_withheld': 15,
                'independent_pdf_downloads_saved': sum('sha256' in d for d in document_manifest),
                'independent_pdf_download_failures': sum(d.get('status') == 'download_failed' for d in document_manifest),
                'candidate_primary_run_uses_alternatives': False,
                'visually_reviewed_pdf_pages': ['hitachiomiya:7', 'ikeda:9', 'kochi:17']}
    (COORD / 'validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (COORD / 'README.md').write_text('''# Withheld coordinate review

All 15 anomalous positions remain retained and withheld in the primary candidate run. Individual basic API requests each returned one record repeating the anomaly. These are the actual bounded responses; they do not prove that all historical records were exposed by the interface. No facility-ID string was parsed to manufacture a coordinate, and source fields were not overwritten.

One independent alternative position was found in the [Hitachiomiya official inventory](https://www.city.hitachiomiya.lg.jp/data/doc/1650968713_doc_105_2.pdf), PDF page 7, row 469, registry 42934. Its name and route correspond to C02, but its dimensions differ from the API (9.1 x 2.3 m versus 9.0 x 1.8 m). DMS conversion is saved separately in `alternative_coordinate_proposals.csv`; reference datum, observation date and dimension definitions remain unresolved. The proposal is not used in national candidates. The following inspection table also contains a different inspection date, so no synchronized asset state is assumed.

Ikeda's official plan, page 9, corroborates C05's normalized name, route and 2.2 x 5.5 m dimensions without supplying an alternative position. Kochi's heritage document, page 17, corroborates C14's locality and road context without an exact coordinate. The retrieved Komatsushima general plan does not resolve the five named coordinate anomalies. Two candidate document links returned HTTP 404 and are recorded as failed acquisitions, not evidence. For the remaining records no independently verified alternative position was obtained in this bounded review. An unresolved position does not mean the physical bridge is absent.

`coordinate_review.csv` is the complete 15-case ledger. `manifest.json` records individual API receipts; `document_manifest.json` records successful PDF hashes and failed URLs. Downloaded source files remain under `sources/`. Three relevant PDF pages were rendered and visually inspected. Reproduce acquisition with `src/exp/fetch_bridge_coordinate_evidence.py` and `src/exp/fetch_bridge_coordinate_documents.py`; extract the downloaded PDFs with local `pdftotext -layout`, render the recorded pages with `pdftoppm`, then run `src/exp/report_national_bridge_crosswalk.py`.
''')
    s = national
    lines = ['# National bridge-to-N13 candidate crosswalk', '',
             'This exploratory result serves the maintenance-input feasibility question in AnaSOP. Nationwide all-road scope is retained. No source coordinate, research question, maintenance decision unit, cost or intervention effect is changed.', '',
             '## Coverage and retained records', '',
             '| Diagnostic | Count |', '| --- | ---: |',
             f'| Source bridge records retained | {s["inventory_records_retained"]:,} |',
             f'| N13 source archives checked | {s["source_tiles"]:,} |',
             f'| Positive ordinary-road source features screened | {s["eligible_road_features"]:,} |',
             f'| Candidate point-to-feature pairs within 200 m | {s["candidate_pairs"]:,} |',
             f'| Assets with a candidate | {s["candidate_status_counts"]["candidate_present"]:,} |',
             f'| Assets without a candidate within 200 m | {s["candidate_status_counts"]["no_candidate_within_200m"]:,} |',
             '| Coordinate anomalies withheld | 15 |',
             f'| Assets whose nearest candidate is within 50 m | {s["assets_nearest_within_50m"]:,} |',
             f'| Assets with bridge/elevated geometry within 50 m | {s["assets_bridge_geometry_within_50m"]:,} |',
             f'| Assets sharing a nearest source feature | {s["assets_sharing_nearest_feature"]:,} |',
             '| Accepted maintenance assignments | 0 |', '',
             f'Among assets with candidates, median nearest distance is {s["nearest_distance_median"]:.3f} m and the 95th percentile is {s["nearest_distance_p95"]:.3f} m. These distances describe proximity; they are not matching accuracy. The broad geographic screen does not detect every plausible-looking erroneous coordinate.', '',
             '## Method and interpretation', '',
             'The source is the national public basic latest bridge inventory and the acquired [N13 September 2024 road centerlines](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html). N13_002=1, N13_004 in 1-4 and positive length define the candidate road set. Expressways and all road levels remain candidates; legal vehicle routing has not been inferred. Each bridge query includes full source-polyline coordinates across neighbouring tiles. Archive hashes and audited feature ordinals are checked. Source IDs are tile plus one-based feature ordinal, tied to the archive hash.', '',
             'All candidate pairs within 200 m are retained. Distances use a local azimuthal equidistant projection per source tile, treating explicit bridge coordinates geographically and N13 as JGD2011. The bridge datum remains unverified. Small projection differences between tiles are possible. The 200 m retrieval radius, 50 m reporting band and nearest-plus-5 m ambiguity flag are analyst diagnostics, not human-approved acceptance criteria. Closest candidates within a one-micrometre numeric tie tolerance are ordered by source key for deterministic summaries. This does not choose an intervention unit.', '',
             'Road-category hints use explicit normalized route-name prefixes. Category agreement is not route identity, and disagreement is a review signal rather than proof of mismatch. Different road-level codes and tunnel-nearest cases need vertical/identity review. Ordinary-road geometry can contain short bridges: absence of a bridge/elevated candidate is not proof that a physical asset is unmatched. Source feature length and the along-line projected point are saved as diagnostics; neither defines the bridge footprint or the complete set of affected graph links.', '',
             f'{s["shared_nearest_feature_groups"]:,} source features are nearest to multiple assets. These relationships are retained; no bridge assets are merged or double-counted as independent road interventions. All {s["intervention_extents_unresolved"]:,} affected-link extents remain unresolved.', '',
             '## Review signals', '', '| Signal | Assets |', '| --- | ---: |']
    lines += [f'| {key} | {value:,} |' for key, value in s['review_flag_counts'].items()]
    lines += ['', f'The queue contains {s["review_queue_records"]:,} distinct flagged records. Flags overlap and indicate different tasks; their count is not an estimated error rate. `review_examples.csv` takes the first five records for each signal for convenient inspection and is not a representative validation sample. National matching accuracy has not been estimated.', '',
              '## Coordinate exceptions', '', '[Coordinate review](coordinate-review/README.md) records all 15 anomalies, repeated API responses, preserved official documents and one independently sourced alternative proposal. No coordinate correction was accepted: the proposal has dimension and temporal/datum questions. All 15 remain withheld rather than being dropped or silently replaced.', '',
              '## Files and reproducibility', '',
              '- `bridge_candidate_summary.csv.gz`: every original asset, nearest-candidate attributes, source provenance and unresolved status.',
              '- `candidate_summary.sqlite`: portable read-only queries by asset ID or nearest source feature.',
              '- `tiles/*.npz` and `tiles/*.json`: complete pairs, source hashes, projection, array hash and feature ordinals; no nearest-only pruning.',
              '- `review_queue.csv.gz`, `review_flag_counts.csv`, `review_examples.csv`: flagged records and inspection entry points.',
              '- `shared_nearest_features.csv`, `prefecture_candidate_coverage.csv`: shared-feature and 47-prefecture coverage diagnostics.',
              '- `coordinate_withheld.csv` and `coordinate-review/`: preserved coordinate exceptions and evidence.',
              '- `candidate_config.json`, `tile_summary.json`, `validation.json`, `sample_parity_validation.json`: inputs, completeness and reconciliation.', '',
              'Run `.venv/bin/python src/exp/build_national_bridge_candidates.py --workers 2`, then `.venv/bin/python src/exp/summarize_national_bridge_candidates.py`, then `.venv/bin/python src/exp/report_national_bridge_crosswalk.py` after the coordinate-evidence steps. Completed tile checkpoints are hash verified and reused. Query one record with `.venv/bin/python src/exp/query_national_bridge_candidates.py --facility-id "41.41056,140.17472"`; add `--full-pairs` to inspect all retained pairs for it.', '',
              'Source hashes, 160 archive feature counts, all candidate-array hashes and 730,844 saved summary rows passed checks. The earlier ten-case review has identical nearest source IDs and 82 retained candidate pairs; distance differences are below 0.05 m. This verifies implementation consistency, not geographic truth. AnaSOP hash is unchanged; all-road national research scope and five feasibility labels remain unchanged.', '',
              '## Next evidence requirement', '',
              'Review candidate identities and affected-link extents before accepting maintenance units. Use a designed validation sample across geography and ambiguity signals if national accuracy is to be claimed. Traffic correspondence, comparable intervention costs and defensible effects remain separate unresolved inputs. No population losses, road closures, maintenance rankings or policy effects are estimated by this crosswalk.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
