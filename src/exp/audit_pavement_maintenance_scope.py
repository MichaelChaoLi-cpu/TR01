"""Read-only local input audit prompted by the omitted pavement-maintenance scope.

Does not adopt a road-maintenance unit, cost model, deterioration assumption,
traffic imputation or new intervention effect. No AnaSOP or raw-file edits.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sop = ROOT / 'docs/AnaSOP.md'
    before = digest(sop)
    condition_terms = ['舗装', 'ひび', 'わだち', '路面', 'MCI', 'IRI', '平たん', '平坦']
    rows, headers, hashes = [], [], {}
    files = sorted((ROOT / 'data/exp/national-traffic-acquisition/sources').glob('kasyo*.csv'))
    assert len(files) == 47
    for path in files:
        hashes[str(path.relative_to(ROOT))] = digest(path)
        with path.open(encoding='cp932', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) == 159
            condition = [(i,h) for i,h in enumerate(header) if any(t.lower() in h.lower() for t in condition_terms)]
            widths = [(i,h) for i,h in enumerate(header) if '幅員' in h]
            positive = 0
            count = 0
            for row in reader:
                assert len(row) == 159
                count += 1
                try:
                    width = float(row[114])
                    positive += int(math.isfinite(width) and width > 0)
                except ValueError:
                    pass
            rows.append({'Prefecture Source': path.stem, 'Rows': count, 'Header Fields': len(header),
                         'Candidate Pavement Condition Header Count': len(condition),
                         'Width Header Count': len(widths), 'Rows With Positive Carriageway Width': positive})
            headers.extend({'Prefecture Source': path.stem, 'Field Position': i,
                            'Original Header': h, 'Candidate Pavement Condition Field': any(t.lower() in h.lower() for t in condition_terms)}
                           for i,h in enumerate(header))
    for name, records in [('traffic_width_and_condition_availability.csv', rows), ('traffic_header_inventory.csv', headers)]:
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    assert sum(r['Rows'] for r in rows) == 99417
    assert sum(r['Candidate Pavement Condition Header Count'] for r in rows) == 0
    for p,h in hashes.items():
        assert digest(ROOT / p) == h
    assert digest(sop) == before
    result = {'status': 'passed_scoped_input_audit', 'traffic_prefectures': 47, 'traffic_rows': 99417,
              'pavement_condition_fields_in_reviewed_traffic_headers': 0,
              'rows_with_positive_carriageway_width': sum(r['Rows With Positive Carriageway Width'] for r in rows),
              'geometry_and_candidate_lengths_available': True, 'N13_widths_are_categories': True,
              'N13_road_state_is_not_pavement_condition': True,
              'complete_national_pavement_condition_dataset_availability': 'not established',
              'current_allocation_domain': '46,688 conditional bridge-associated units',
              'pavement_cost_or_intervention_model_adopted': False, 'AnaSOP_unchanged': True,
              'source_hashes_unchanged': hashes,
              'official_pavement_guideline': 'https://www.mlit.go.jp/road/sisaku/yobohozen/tenken/yobo7_11.pdf',
              'official_N13_catalog': 'https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html'}
    (OUT / 'scope_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT / 'README.md').write_text('''# Pavement-maintenance scope review

The current common-budget allocation results cover 46,688 conditional bridge-associated units, with bridge area as a relative-cost proxy and observed bridge inspection categories. They do not cover general pavement maintenance. The user identified pavement maintenance as a relevant need; no revised cost, intervention unit or pavement-deterioration assumption has yet been adopted.

Existing nationwide candidate roads and lengths support road-segment locators. N13 has width categories and geometric/source-state attributes, not observed pavement deterioration. All 47 acquired 2021 traffic tables (99,417 records, 159 fields per table) have width fields but no pavement-condition header among the reviewed crack, rutting, roughness and maintenance-index terms. Positive carriageway-width availability is documented separately; it does not establish a validated road-width crosswalk or all-road coverage. Sparse local-road traffic coverage remains unresolved.

The official MLIT pavement inspection guideline treats crack ratio, rut depth and IRI as principal pavement-management indicators and distinguishes pavement inspections from structural bridge inspections. It does not supply a nationwide geocoded dataset or justify assigning structural bridge categories to pavement. A current absence in this project is not evidence of nationwide unavailability.

Before a pavement-based policy comparison, confirm road-level decision units, comparable relative costs, pavement-condition coverage (or explicit assumptions), and how intervention affects network service. Pavement maintenance cannot be silently equated with preventing complete road closure, and existing bridge-relative budget percentages cannot be carried to a new domain unchanged. Existing demographic, medical-access, traffic-linkage and road-dependence constructions remain reusable; bridge allocation results remain their original bounded scenario.

Recommended next step: a bounded pavement-source availability and modelling decision through the responsible research-question and estimation-planning workflows, before recomputing allocations or presenting bridge outputs as an all-road result. No research contract or accepted output is rewritten by this audit.
''')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes_unchanged'},indent=2))


if __name__ == '__main__':
    main()
