"""Close a two-case boundary-evidence check without expanding research scope."""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

from fetch_bounded_bridge_boundary_sources import ROOT, OUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    manifest=json.loads((OUT/'source_manifest.json').read_text())
    assets=json.loads((OUT/'source_asset_records.json').read_text())
    previous=json.loads((OUT.parent/'validation.json').read_text())
    assert len(manifest)==4 and len(assets)==2
    assert {r['review_id'] for r in assets}=={'S03','S12'}
    assert digest(ROOT/'docs/AnaSOP.md')==previous['anasop_sha256']
    for receipt in manifest:
        assert digest(OUT/'sources'/receipt['file'])==receipt['sha256']
    for source in assets:
        assert digest(ROOT/'data/exp/feasibility-check/national-bridge-inventory'/source['source_page'])==source['source_page_sha256']
    pdfs={r['file'].split('_')[0]:r for r in manifest if r['file'].endswith('.pdf')}
    cases=[]
    for source in assets:
        id=source['review_id']; a=source['record']['syogen']
        pages=(OUT/'sources'/f'{id}_manager_plan.txt').read_text().split('\f')
        normalized=[unicodedata.normalize('NFKC',p) for p in pages]
        page=49 if id=='S03' else 5
        name=a['shisetsu']['meisyou']
        row=next(line for line in normalized[page-1].splitlines() if name in line)
        length=45.18 if id=='S03' else 19.7
        width=7.9 if id=='S03' else 16.0
        assert str(length) in row and str(width) in row
        assert round(length,1)==a['kyouchou'] and width==a['fukuin']
        if id=='S03':
            assert '市道1号線' in row and '1972' in row
            notes='Bridge name, route, 1972 construction year and width agree; 45.18 m rounds to the API length 45.2 m. Citywide point map does not locate both deck endpoints.'
            doc_year=1972; inspected='PDF pages 6 and 49 (printed pages 3 and 46)'
        else:
            assert '19 号線' in row
            assert any('北浦跨線橋' in line and '1981' in line for line in normalized[13].splitlines())
            notes='Bridge name, route number and dimensions agree. Construction year differs: API 1980 versus manager table 1981. Named repair photographs are not georeferenced two-endpoint geometry; do not overwrite either source.'
            doc_year=1981; inspected='PDF pages 5, 8 and 14 (printed pages 3, 6 and 12)'
        cases.append({'review_id':id,'facility_id':source['record']['shisetsu_id'],'manager':a['kanrisya']['meisyou'],
                      'bridge_name':name,'api_route_name':a['rosen']['meisyou'],
                      'api_length_m':a['kyouchou'],'manager_length_m':length,
                      'api_width_m':a['fukuin'],'manager_width_m':width,
                      'api_construction_year':a['kasetsu_nendo'],'manager_construction_year':doc_year,
                      'manager_document_url':pdfs[id]['url'],'document_locator':inspected,
                      'identity_evidence':'corroborated_name_route_dimensions; not_unique_boundary_verification',
                      'finding':notes,'both_georeferenced_endpoints_obtained':False,
                      'affected_graph_links_accepted':False,'original_record_retained':True})
    write_csv('case_boundary_findings.csv',cases)
    gates=[
        {'requirement':'Asset identity and physical affected-link extent','current_evidence':'Two official manager plans corroborate names/routes/dimensions. No sufficient georeferenced endpoint/component evidence was obtained for either case.','status':'not_passed','consequence':'No national accepted maintenance units. Retain bridge candidates and source conflicts.','reopen_condition':'Usable manager boundary/component geometry or a separately human-approved conditional extent model.'},
        {'requirement':'Measured traffic linkage and comparable coverage','current_evidence':'Prior bounded traffic-format evidence; nationally accepted traffic-to-network coverage remains unresolved.','status':'not_passed','consequence':'Cannot implement the national measured-traffic priority baseline.','reopen_condition':'Accepted source-to-network traffic correspondence with explicit coverage.'},
        {'requirement':'Comparable intervention costs','current_evidence':'Downloaded municipal plans include local planning information, but no harmonized comparable national action-cost data have been established.','status':'not_passed','consequence':'Cannot call rankings a comparison under an equivalent monetary budget.','reopen_condition':'Comparable asset/action costs or an explicitly approved normalized-budget scenario.'},
        {'requirement':'Intervention-to-network-function relationship','current_evidence':'A repair plan or health category does not establish closure probability or accessibility preserved by maintenance.','status':'not_passed','consequence':'No automatic health-to-closure or maintenance-to-avoided-loss mapping.','reopen_condition':'Defensible function/effect evidence or a separately approved conditional scenario.'}
    ]
    write_csv('maintenance_comparison_gates.csv',gates)
    closeout={'investigation_status':'closed_at_two_case_evidence_limit',
              'human_instruction_record':'MILI-D-20261006-015',
              'human_instruction':'Keep within the research-question boundary and converge promptly.',
              'analyst_execution_choice':'Inspect two existing clearer municipal cases and their official manager plans; stop further case expansion after the boundary-evidence finding.',
              'two_cases_do_not_establish':'That endpoint records are unavailable from every manager, or that the national maintenance study is impossible.',
              'accepted_scope':'National Japan, existing central maintenance-allocation question and supporting questions retained.',
              'analyst_recommendation':'Move the next empirical work to the already planned national baseline service-access and ageing question; retain maintenance-policy comparison behind its explicit input gates.',
              'baseline_pending_choices':'Age denominator, publication-group-to-origin access support and eligible medical destinations remain subject to the responsible preprocessing workflow and human choices.',
              'not_adopted':['Bridge-only research scope','Whole-source-feature maintenance unit','Equal monetary costs','Automatic bridge closure','Maintenance-caused restoration','A replacement central research question'],
              'reopen_boundary_work_only_for':'New usable boundary/component evidence or an explicitly approved conditional modeling choice; another visual sample alone is not sufficient.'}
    (OUT/'closeout_decision.json').write_text(json.dumps(closeout,indent=2)+'\n')
    lines=['# Bounded bridge boundary evidence: closeout','',
           'The current bridge-boundary investigation ends after two existing cases and two official manager plans. The human requested that research work stay within its question boundary and converge promptly (MILI-D-20261006-015). Case choice and this stopping threshold are analyst execution choices, not a new human-approved research scope. No additional visual sample is generated.','',
           '## Evidence and result','',
           '| Existing case | Official plan | Corroboration | Remaining boundary evidence |',
           '| --- | --- | --- | --- |',
           f'| S03, Shukuura Bridge, Hasuda | [December 2025 plan]({pdfs["S03"]["url"]}), PDF pp. 6/49 | Name, municipal route 1, width and construction year agree; document length 45.18 m rounds to API 45.2 m | The citywide point map and asset schedule do not establish both georeferenced deck endpoints or complete affected links |',
           f'| S12, Kitaura railway overbridge, Ushiku | [March 2024 plan]({pdfs["S12"]["url"]}), PDF pp. 5/8/14 | Name, route 19 and 19.7 x 16.0 m dimensions agree; construction-year discrepancy retained (API 1980, plan 1981) | Named repair photographs and register entries do not yield sufficient georeferenced endpoint/component geometry |','',
           'Both cases retain provisional corridor evidence. No complete affected-link set or maintenance assignment is accepted. This bounded finding is not a claim that detailed endpoint records are absent from every municipality. Additional original manager drawings might resolve cases, but systematically obtaining them would be a new data effort; this check does not expand into that effort.','',
           'The full documents were locally text-screened; the cited asset register, overview and repair pages were rendered and visually inspected. Keyword screening alone is not proof that all possible drawings were absent. The conclusion is that this check did not obtain sufficient mapping evidence. Planning figures and repair illustrations are not an observed national maintenance-cost/effect dataset.','',
           '## Return to the research question','',
           'The central question compares traffic, condition and basic-access priorities under a shared constrained maintenance budget. Improving a proximity match alone cannot complete that comparison: decision units/affected links, measured traffic correspondence, comparable intervention costs and intervention effects are all unresolved. maintenance_comparison_gates.csv records these four requirements without adding assumptions. Equal cost, whole-feature repair and automatic closure are not silently adopted.','',
           'The analyst recommendation is to prioritize the existing supporting question on national baseline service access and ageing, using already acquired population, destination and candidate-network inputs through the responsible preprocessing workflow. This keeps empirical progress tied to the accepted question rather than making bridge matching the study itself. Age denominator, aggregation support and eligible medical destinations still need their previously pending choices; baseline work is not declared complete or unconditionally ready. The maintenance-policy contrast remains the accepted central question, pending its input gates. There is no AnaSOP revision or automatic conversion to a bridge-only or purely descriptive paper.','',
           'Further boundary investigation should be reopened only when a usable manager boundary source is available, or the human explicitly adopts a conditional geometry/intervention model. A new map sample by itself is not a reason to continue indefinitely.','',
           '## Files and reproducibility','',
           '- case_boundary_findings.csv: two-case source correspondence, dimensions, discrepancy and retained mapping status.',
           '- maintenance_comparison_gates.csv: requirements for the original policy comparison.',
           '- closeout_decision.json: human constraint, analyst stopping choice and recommendation distinguished.',
           '- sources/, source_manifest.json, source_asset_records.json: hashed official files and original basic API records with source-page provenance.',
           '- validation.json: checked source hashes, extracted values, unchanged contract and zero accepted affected-link sets.','',
           'Use .venv/bin/python src/exp/fetch_bounded_bridge_boundary_sources.py, local pdftotext -layout for the two PDFs, and pdftoppm to render the five cited PDF pages. Then run .venv/bin/python src/exp/report_bounded_bridge_boundary_closeout.py. The Ushiku introduction URL returned an HTML meta-refresh redirect; the official PDF URL was obtained independently through official-domain lookup, and both responses are preserved. No source field, raw dataset, national candidate relation, graph or AnaSOP is modified. No new dependency, Git or DVC operation is performed.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    expected=['S03_page_6','S03_page_49','S12_page_5','S12_page_8','S12_page_14']
    assert all((OUT/(name+'.png')).is_file() for name in expected)
    result={'status':'passed','investigation_status':'closed_at_two_case_evidence_limit',
            'existing_cases':2,'new_cases_generated':0,'official_manager_pdfs':2,'source_receipt_hashes_verified':4,
            'both_endpoint_evidence_obtained':0,'accepted_affected_link_sets':0,'accepted_maintenance_assignments':0,
            'source_value_discrepancies_retained':True,'national_unavailability_claimed':False,
            'question_scope_changed':False,'conditional_maintenance_model_adopted':False,
            'anasop_unchanged':True,'anasop_sha256':previous['anasop_sha256'],
            'visually_reviewed_page_hashes':{name:digest(OUT/(name+'.png')) for name in expected},
            'next_step_is_analyst_recommendation':True}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
