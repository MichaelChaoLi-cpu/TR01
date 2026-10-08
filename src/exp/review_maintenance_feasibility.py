"""Reproduce the evidence-based review after the bundled lexical scanner.

Standard library only. External observations are a dated desk review, not an
automated acquisition or a tested spatial join. Never writes raw data or AnaSOP.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/feasibility-check'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    sop = ROOT / 'docs/AnaSOP.md'
    before = hashlib.sha256(sop.read_bytes()).hexdigest()
    for name in ['README.md', 'dataset_availability.csv', 'variable_inventory.csv', 'question_feasibility.csv']:
        dest = OUT / ('automated_' + name)
        if not dest.exists():
            shutil.copyfile(OUT / name, dest)
    auto = read(OUT / 'automated_question_feasibility.csv')
    assert len(auto) == 5
    traffic_sample = (OUT / 'source-matching-sample/summary.json').exists()
    bridge_sample = (OUT / 'bridge-access-rural-sample/summary.json').exists()
    crosswalk_review = (OUT / 'bridge-crosswalk-review/validation.json').exists()
    national_bridge_validation = OUT / 'national-bridge-inventory/validation.json'
    national_bridge = national_bridge_validation.exists() and json.loads(national_bridge_validation.read_text()).get('acquisition_complete') is True
    candidate_validation_path = OUT / 'national-bridge-crosswalk/validation.json'
    national_crosswalk_validation = json.loads(candidate_validation_path.read_text()) if candidate_validation_path.exists() else {}
    national_crosswalk = national_crosswalk_validation.get('status') == 'passed'
    stratified_path = OUT / 'national-bridge-crosswalk/stratified-review/validation.json'
    stratified = stratified_path.exists() and json.loads(stratified_path.read_text()).get('status') == 'passed'
    boundary_path = OUT / 'national-bridge-crosswalk/boundary-evidence-closeout/validation.json'
    boundary_closed = boundary_path.exists() and json.loads(boundary_path.read_text()).get('investigation_status') == 'closed_at_two_case_evidence_limit'
    if national_crosswalk:
        assert national_bridge
        assert national_crosswalk_validation['inventory_records_retained'] == 730844
        assert national_crosswalk_validation['source_tiles'] == 160
        assert national_crosswalk_validation['accepted_maintenance_matches'] == 0
    judgments = [
        ('not-yet-testable', 'No complete common-budget comparison is supported.', 'Population, destinations and candidate connectivity are available.', 'Nationally usable traffic/condition coverage, costs, decision units and intervention effects are unverified.', 'Verify linkable policy inputs before allocation modeling.'),
        ('partly-testable', 'Descriptive distance and population patterns can be developed.', '176,964 census meshes and 146,540 age publication supports; medical and fire-facility diagnostics.', 'Age denominator and within-group access linkage, eligible facility categories and vehicle routing require decisions or validation.', 'Proceed through data-preprocessing for compatible demographic and accessibility supports.'),
        ('partly-testable', 'Conditional route-dependence diagnostics are possible; measured-traffic comparison is incomplete.', 'Candidate graph and origin/destination attachments permit explicitly assumed service-loss diagnostics.', 'Accepted traffic linkage, maintenance units, plausible disruption and scalable computation remain unverified.', 'Validate traffic-section and inspection-asset correspondence to N13 before national acquisition.'),
        ('not-yet-testable', 'Distributional policy comparisons lack comparable policy inputs.', 'Population and age supports can underpin future distributional outcomes.', 'Common intervention costs and effects and complete traffic/condition rules are absent.', 'Complete input feasibility before selecting allocation objectives or producing policy rankings.'),
        ('weakly-testable', 'Only assumed sensitivities are possible from current acquisitions.', 'A 2020 demographic starting state is available; official future projections were located externally.', 'No harmonized historical changes, projections or fiscal trajectories have been acquired; baseline allocation inputs are missing.', 'Acquire and harmonize demographic scenarios; label projections and assumed budget reductions explicitly.'),
    ]
    reviewed = []
    sop_text = sop.read_text()
    for row, judgment in zip(auto, judgments):
        assert row['question'] in sop_text
        status, rationale, evidence, gap, next_step = judgment
        reviewed.append({k: row[k] for k in ['id','kind','section','question']} | {'status':status, 'automated_status':row['status'], 'rationale':rationale, 'available_evidence':evidence, 'blocking_inputs':gap, 'next_step':next_step})
    write('question_feasibility.csv', reviewed)
    if bridge_sample:
        for row in reviewed:
            if row['id'] in ['RQ1','RQ3','RQ4']:
                row['available_evidence'] += ' Public traffic/geometric samples and coordinate-bearing basic bridge API records have now been acquired; no network assignment is accepted.'
        write('question_feasibility.csv', reviewed)
    if national_bridge:
        for row in reviewed:
            if row['id'] in ['RQ1','RQ3','RQ4']:
                row['available_evidence'] += ' A national public basic latest bridge inventory is acquired and reconciled to API counts; this is an unmatched source inventory, not all-road condition coverage.'
            if row['id'] == 'RQ3':
                row['next_step'] = 'Validate the national bridge-to-N13 candidate crosswalk and traffic correspondence before accepting maintenance units.'
        write('question_feasibility.csv', reviewed)
    if national_crosswalk:
        for row in reviewed:
            if row['id'] in ['RQ1','RQ3','RQ4']:
                row['available_evidence'] += ' National bridge-to-N13 proximity candidates retain all source assets and unresolved exceptions; no maintenance unit or affected-link extent is accepted.'
            if row['id'] == 'RQ3':
                row['next_step'] = 'Review national candidate identities and affected-link extents; validate traffic correspondence before accepting maintenance units.'
        write('question_feasibility.csv', reviewed)
    if stratified:
        assert national_crosswalk
        for row in reviewed:
            if row['id'] in ['RQ1','RQ3','RQ4']:
                row['available_evidence'] += ' A twelve-case stratified visual diagnostic identifies stacked-road, component-function, shared-feature and junction-piece ambiguities; ten provisional corridor associations and two withheld records remain unaccepted as maintenance units.'
            if row['id'] == 'RQ3':
                row['next_step'] = 'Obtain manager component/endpoint/carriageway evidence for traceable asset-to-graph mapping; validate traffic correspondence and intervention assumptions separately.'
        write('question_feasibility.csv', reviewed)
    if boundary_closed:
        assert stratified
        for row in reviewed:
            if row['id'] in ['RQ1','RQ3','RQ4']:
                row['available_evidence'] += ' Two official municipal plans corroborate existing sample names/routes/dimensions but do not provide sufficient two-endpoint mapping evidence in this check; the boundary investigation is closed without expanding samples or adopting intervention assumptions.'
            if row['id'] == 'RQ3':
                row['next_step'] = 'Do not expand bridge visual samples. Reopen affected-link mapping only for usable manager boundary evidence or an explicitly approved conditional model; prioritize the existing baseline-access and ageing preparation meanwhile.'
        write('question_feasibility.csv', reviewed)
    archives = []
    for folder in sorted((ROOT / 'data/raw').glob('*/*')):
        candidates = [folder] if list(folder.glob('*.zip')) else sorted(p for p in folder.iterdir() if p.is_dir())
        for candidate in candidates:
            count = len(list(candidate.glob('*.zip')))
            if count:
                archives.append({'source_directory':str(candidate.relative_to(ROOT)), 'archive_count':count, 'acquired':True, 'assessment':'Archive count only; prior source-specific audits provide content evidence. Not a new full integrity audit.'})
    write('archived_dataset_availability.csv', archives)
    shutil.copyfile(ROOT / 'data/exp/data-briefing/tables/variable_summary.csv', OUT / 'archived_population_poi_variable_inventory.csv')
    sources = [
        ('traffic', '2021 traffic census', 'https://www.mlit.go.jp/road/census/r3/', '47 prefectural basic-section and hourly-count tables are listed.', 'Main-road survey; not all municipal roads. Some volumes are estimated; retain observation flags. 2021/2024 mismatch and N13 matching untested.'),
        ('traffic_scope', 'Traffic census methodology', 'https://www.mlit.go.jp/road/census/r3/data/pdf/kasyorep.pdf', 'Defines road coverage and measured/estimated traffic distinctions.', 'Coverage cannot be extrapolated to every N13 ordinary-road edge.'),
        ('condition', 'National inspection database catalog', 'https://www.xroad.mlit.go.jp/database/221/', 'Official catalog lists inspection information and UI/API provision.', 'Catalog existence does not establish public bulk access or full-road usable condition coverage.'),
        ('condition_access', 'Inspection database operator FAQ', 'https://rirs.or.jp/tenken-db/pdf/faq.pdf', 'FAQ dated 2024-04-01 distinguishes public basic map information and paid detailed/API access.', 'Current permissions and acquisition terms must be reconfirmed. No account, purchase or API request was made.'),
        ('condition_schema', 'Bridge sample schema', 'https://rirs.or.jp/tenken-db/pdf/sampledata_bridge.pdf', 'Facility identity, origin-side coordinates, dimensions, inspection dates and health categories appear in the schema.', 'Schema is not acquired nationwide records. Link asset to all affected N13 edges; parallel roads and point placement require review. Traffic and cost coverage are not established.'),
        ('annual_condition', 'FY2024 maintenance report and inspection lists', 'https://www.mlit.go.jp/road/sisaku/yobohozen/yobohozen_maint_r06.html', 'Lists inspected assets for that fiscal year by manager class.', 'Annual inspected-asset lists alone are not the complete all-asset condition stock. Individual tables were not acquired or validated.'),
        ('maintenance_cost', 'Preventive maintenance policy and plan links', 'https://www.mlit.go.jp/road/sisaku/yobohozen.html', 'Official entry point for maintenance policy and manager plans.', 'No national unified linkable intervention-cost dataset was verified in this screening. Inspection unit cost is not repair cost.'),
        ('projection', 'Mesh population projection methods', 'https://www.mlit.go.jp/kokudoseisaku/content/r6_about_future_population.pdf', '2020-based 250m/500m/1km projections for 2025-2070 in five-year steps; cohort-component method.', 'Not acquired. Unknown-age imputation and demographic controls mean modeled base counts need reconciliation with published census supports. Projections are not observed decline.'),
    ]
    if bridge_sample:
        sources.append(('public_basic_bridge_api', 'Basic bridge API and actual bounded responses', 'https://rirs.or.jp/tenken-db/pdf/view/api_specification.pdf', 'Bounded requests returned coordinates and health records without credentials; ten latest Nikko facility IDs are unique.', 'Only sample coverage was tested; network assignment, nationally complete acquisition, costs and intervention effects remain unresolved. See bridge-access-rural-sample/README.md.'))
        sources = [
            (a, b, c, 'Bounded public basic bridge API access returned actual coordinates and health records without authentication.' if a == 'condition_access' else d,
             'The initial FAQ screening is superseded for basic API access by the tested specification and free/basic terms; paid details are separate. No account, purchase or access-application request was made.' if a == 'condition_access' else
             'Two annual-list files were subsequently acquired and parsed in the Tokyo source sample; annual inspected assets are not the complete all-asset stock.' if a == 'annual_condition' else e)
            for a, b, c, d, e in sources
        ]
    if national_bridge:
        sources.append(('national_basic_bridge_inventory', 'National public basic latest bridge inventory', 'https://rirs.or.jp/tenken-db/pdf/view/api_specification.pdf', 'Actual latest basic bridge records acquired in all 47 prefecture queries; source pages, unique IDs and national/prefecture counts audited.', 'Coverage reconciles to the exposed API, not all physical bridges or every road condition. No accepted maintenance crosswalk, intervention cost or effect. See national-bridge-inventory/README.md.'))
    write('external_sources.csv', [{'input':a,'title':b,'url':c,'verified_observation':d,'limitation_or_next_check':e,'review_date':'2026-10-06','nationwide_records_acquired':national_bridge and a == 'national_basic_bridge_inventory'} for a,b,c,d,e in sources])
    inputs = [
        ('Population and ageing', 'acquired; partly usable', 'Publication groups and local meshes differ; select denominator and access linkage.', 'data/exp/data-briefing/population-sample-audit/README.md'),
        ('Vehicle network', 'candidate only', 'Candidate topology is audited, not complete legal routing, direction or speed validation.', 'data/exp/national-network-rule-validation/README.md'),
        ('Eligible service destinations', 'acquired; selection pending', 'Historical years and suspended facilities; fire stations/branches are assumed ambulance bases, not observed response times.', 'data/exp/data-briefing/README.md'),
        ('Traffic volume', 'sample acquired; national usable coverage unverified' if traffic_sample else 'not acquired', 'Official main-road source located; accepted N13 crosswalk and nationwide coverage remain unresolved.', 'source-matching-sample/README.md' if traffic_sample else 'external_sources.csv'),
        ('Physical condition', 'coordinate-bearing public sample acquired; national usable coverage unverified' if bridge_sample else 'not acquired', 'Public basic access works in the bounded sample; accepted asset-level linkage and national coverage remain unresolved. N13 structure codes are not condition.', 'bridge-access-rural-sample/README.md' if bridge_sample else 'external_sources.csv'),
        ('Maintenance decision units', 'not selected', 'Road sections and physical assets can have many-to-many geometry relationships.', 'docs/AnaSOP.md'),
        ('Intervention cost', 'not verified', 'Need comparable asset/action costs; equal cost has not been approved.', 'external_sources.csv'),
        ('Intervention effect', 'not verified', 'Need defensible maintenance-to-usable-function relation; nonmaintenance does not imply universal closure.', 'docs/AnaSOP.md'),
        ('Demographic change', 'external source located; not acquired', 'Future projections require modeled-base harmonization; one census does not establish decline.', 'external_sources.csv'),
        ('Fiscal and economic context', 'not acquired or source-verified', 'No municipal economic/fiscal specification can be inferred from current acquisitions.', 'docs/AnaSOP.md'),
    ]
    if national_bridge:
        inputs = [(a, 'national basic bridge inventory acquired; accepted linkage unverified',
                   'Source count and identity reconciliation completed. Physical bridges are not all-road condition coverage; affected graph links remain unresolved.',
                   'national-bridge-inventory/README.md') if a == 'Physical condition' else (a,b,c,d)
                  for a,b,c,d in inputs]
    if national_crosswalk:
        inputs = [(a, 'national candidates generated; decision units unresolved',
                   'All 730,844 assets retained; 726,842 have a candidate within 200 m, 3,987 have none, and 15 coordinate anomalies remain withheld. Identity and affected-link extents are unaccepted.',
                   'national-bridge-crosswalk/README.md') if a == 'Maintenance decision units' else (a,b,c,d)
                  for a,b,c,d in inputs]
    if boundary_closed:
        inputs = [(a, 'candidates retained; bounded boundary check closed without accepted units',
                   'Two official manager plans corroborate sample identity/dimensions; sufficient endpoint evidence was not obtained. No national-unavailability claim or whole-feature repair assumption. Reopen only with usable boundary evidence or a separately approved conditional model.',
                   'national-bridge-crosswalk/boundary-evidence-closeout/README.md') if a == 'Maintenance decision units' else (a,b,c,d)
                  for a,b,c,d in inputs]
    write('required_input_matrix.csv', [{'input':a,'availability':b,'requirement':c,'evidence':d} for a,b,c,d in inputs])
    file_count = len(read(OUT / 'automated_dataset_availability.csv'))
    field_count = len(read(OUT / 'automated_variable_inventory.csv'))
    statuses = dict(Counter(r['status'] for r in reviewed))
    lines = [
        '# Maintenance policy feasibility review', '',
        'Review date: 2026-10-06. Research contract: `docs/AnaSOP.md`, Sections 1-2. Nationwide scope is retained. No research question or AnaSOP section was changed.', '',
        'The project can proceed with descriptive accessibility preparation and conditional road-dependence diagnostics. The full three-rule common-budget maintenance comparison is not yet supported by current acquisitions. Nationally usable traffic/condition coverage, accepted network linkage, comparable intervention costs and intervention effects remain blocking inputs. These gaps do not establish that the study is impossible.', '',
        '## Question-specific findings', '',
        '| Question | Reviewed status | What can currently be done |', '| --- | --- | --- |',
    ]
    lines += [f"| {r['id']}: {r['section']} | {r['status']} | {r['rationale']} |" for r in reviewed]
    lines += ['', 'The statuses assess components of each question, not full empirical identification. RQ3 cannot yet complete its measured-traffic comparison. RQ5 is weakly testable only through explicitly assumed sensitivities; no demographic change is observed in the current single-year census.', '',
        '## Local evidence and coverage', '',
        f'The bundled scanner found {file_count:,} tabular files and {field_count:,} column entries. These are repeated engineering-file columns, not unique scientific variables. The lexical scanner does not inspect compressed census tables, zipped spatial sources, shapefiles or graph arrays; its original report and tables are preserved with the `automated_` prefix. Its name matches do not validate construct meaning.', '',
        f'The archive supplement inventories {sum(r["archive_count"] for r in archives):,} acquired ZIP files. Population/POI briefing supplies 216 source-field summaries across nine logical tabular/spatial datasets; N13 road archives are an additional source family. `variable_inventory.csv` retains the scanner inventory; `archived_population_poi_variable_inventory.csv` supplements the compressed sources rather than treating scanner omissions as absent data.', '',
        'Prior audits identify 176,964 census meshes, 146,540 official age publication supports, and 200 unresolved candidate road entries containing 1,645 residents. None should be removed or assigned a zero access distance. Age counts cannot be assigned solely to a recipient mesh origin. The candidate graph contains 21,415,800 positive-length edges; topology diagnostics do not prove vehicle permissions or actual travel times.', '',
        'Population is from 2020, roads from September 2024, and facilities from different reference years. Source presence does not produce a synchronized observation year. All-road nationwide coverage must be distinguished from the narrower coverage of each proposed policy input.', '',
        '## Official source screening and matching requirements', '',
    ]
    lines += [f'- [{b}]({c}): {d} {e}' for a,b,c,d,e in sources]
    lines += ['', 'Required accepted joins remain unresolved: traffic survey sections to N13 edges; inspection asset identity to all affected road edges; age publication groups to member-origin accessibility; and modeled projection baselines to observed census supports. Spatial proximity alone cannot establish that an inspection asset and a nearby crossing road are the same intervention unit.', '',
        '## Recommended next step', '',
        'Proceed through data-preprocessing for supported demographic/accessibility inputs after resolving their remaining human choices. In parallel, undertake a small source-format and spatial-matching check for public traffic and condition records before nationwide acquisition or allocation modeling. Bridge inspection records are a promising candidate for that check, because the published schema has asset and condition fields; this is a recommendation, not an approved narrowing of the national all-road research object.', '',
        'If input coverage proves insufficient for the accepted central question, return to research-question-planning with the evidence and ask the human to choose between retaining the scope with further acquisition, a nationally distributed asset subset, or an explicitly conditional scenario study. Do not silently substitute modeled traffic, equal costs or universal road closure.', '',
        '## Reproduction and validation', '',
        'Run the bundled feasibility scanner, then `.venv/bin/python src/exp/review_maintenance_feasibility.py`. The review script reproduces tables from local inventories and dated desk-review observations; it does not refresh external websites or download policy datasets. See `validation.json` for contract identity, inventory counts and reviewed status counts. The script writes only this exploratory output folder, preserves raw data and does not run Git or DVC.', '',
    ]
    if traffic_sample:
        lines += ['## Subsequent source sample checks', '', '[Tokyo source matching sample](source-matching-sample/README.md) documents actual traffic/geometry and annual inspection formats. [Coordinate-bearing bridge and mountain traffic check](bridge-access-rural-sample/README.md) adds six Nikko survey geometries and ten latest public basic bridge records. Bounded free API requests actually returned location and health data without credentials. These discoveries improve input feasibility but do not change the five question labels: final linkage, national usable coverage, costs and intervention effects remain unresolved.', '']
    if crosswalk_review:
        lines += ['## Subsequent bridge crosswalk review', '', '[Ten-case visual crosswalk review](bridge-crosswalk-review/README.md) supports nine provisional road-corridor associations and withholds one old-bridge case. Four provisional associations use bridge/elevated geometry; five short bridges use ordinary-road geometry. Two distinct short assets share one 768 m source feature. All ten intervention extents remain unresolved and no maintenance assignment is accepted. This convenience sample cannot estimate national accuracy. National basic-inventory acquisition can be staged separately, preserving unmatched records; intervention units, effects and costs remain prerequisites for allocation modeling.', '']
    if national_bridge:
        lines += ['## National bridge inventory acquired', '', '[National public basic bridge inventory](national-bridge-inventory/README.md) adds source records for all 47 prefecture queries, with page hashes, count reconciliation, unique-ID checks and retained field-quality flags. Acquisition follows the human-authorized next step and does not narrow the national all-road contract. The next step for this input is national candidate crosswalk validation and exception handling; the earlier recommendation to acquire a small condition sample is superseded. Five question labels remain unchanged because usable joint coverage, maintenance units, intervention costs and effects are unresolved.', '']
    if national_crosswalk:
        lines += ['## National bridge candidates generated', '', '[National candidate crosswalk](national-bridge-crosswalk/README.md) retains 730,844 assets and screens 160 N13 archives. It records 27,107,426 candidate pairs: 726,842 assets have candidates within 200 m, 3,987 have none, and 15 coordinate anomalies remain withheld. The radius is diagnostic, not an acceptance rule. Of the assets with candidates, 119,274 share a nearest source feature with another asset. Individual API responses repeated all 15 coordinate anomalies; one official alternative position remains a separate unaccepted proposal because dimensions and temporal/datum correspondence need reconciliation. No affected-link extent or maintenance assignment is accepted. This supersedes the recommendation to generate national bridge candidates: the next bridge task is identity and intervention-extent validation. The five question labels and national all-road scope remain unchanged.', '']
    if stratified:
        lines += ['## Stratified identity and extent diagnostic', '', '[Twelve-case visual diagnostic](national-bridge-crosswalk/stratified-review/README.md) covers twelve prefectures and eight macroregions, with five shared-feature partner records. Ten provisional corridor associations or candidate sets and two withheld cases establish no national accuracy estimate. A named motorway asset is nearest to a lower municipal street; a staircase/attachment name requires component-function review; multiple nearby features can be coincident junction pieces rather than independent alternatives. All twelve affected-link extents remain unresolved. An analyst review protocol separates position, vehicle relevance, corridor identity, physical endpoints, source-to-graph coverage and intervention assumptions. The next condition-input task is manager boundary/component evidence and traceable asset-to-graph mapping, not automatic nearest-road acceptance. No AnaSOP contract, maintenance-unit definition or feasibility label changes.', '']
    if boundary_closed:
        lines += ['## Bounded boundary investigation closed', '', '[Two-case manager evidence closeout](national-bridge-crosswalk/boundary-evidence-closeout/README.md) ends this branch after two existing cases and two official municipal plans. Names/routes/dimensions are corroborated, but no sufficient complete endpoint/affected-link evidence was obtained; a construction-year discrepancy is retained. This does not establish nationwide unavailability. In response to the human instruction to respect the question boundary and converge promptly, no further visual sample is generated. Current analyst priority is the existing national baseline-access and ageing question, through preprocessing and its pending human choices. Reopen bridge-boundary work only for usable manager evidence or an explicitly approved conditional model. This supersedes earlier recommendations to continue sample or boundary exploration. The central common-budget maintenance contrast and national all-road scope remain accepted, but decision units, traffic correspondence, comparable costs and intervention effects are still gated. No alternative paper scope or assumption is silently adopted.', '']
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    after = hashlib.sha256(sop.read_bytes()).hexdigest()
    assert before == after
    assert sum(r['archive_count'] for r in archives) == 560
    assert statuses == {'not-yet-testable':2,'partly-testable':2,'weakly-testable':1}
    validation = {'status':'passed','anasop_sha256':before,'anasop_unchanged_during_review':True,'questions':5,'question_text_matches_contract':True,'automated_files':file_count,'automated_column_entries':field_count,'acquired_archives':560,'briefing_source_fields':216,'reviewed_status_counts':statuses,'external_sample_records_acquired':traffic_sample or bridge_sample,'nationwide_policy_records_acquired':national_bridge,'national_bridge_source_inventory_acquired':national_bridge,'national_bridge_candidates_generated':national_crosswalk,'accepted_maintenance_assignments':0,'national_traffic_inventory_acquired':False,'complete_allocation_inputs':False,'scope_revised':False,'raw_write_operations':0}
    validation['stratified_bridge_identity_review_completed'] = stratified
    validation['bounded_bridge_boundary_investigation_closed'] = boundary_closed
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))

if __name__ == '__main__':
    main()
