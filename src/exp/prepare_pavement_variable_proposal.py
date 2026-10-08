"""Run scoped skill discovery and prepare a reviewable naming/coding proposal.

No processed input, final decisions JSON or AnaSOP variable definitions are
written before the user confirms the selected names and construction proposal.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/national_road_correspondence/proposed_road_grade_aggregation.parquet'
OUT = ROOT / 'data/exp/data-preprocessing/pavement-inputs'
SOP = ROOT / 'docs/AnaSOP.md'

NAMES = {
    'nearest_graph_source_edge_id': ('Nearest Graph Source Edge ID', 'exclude redundant identity'),
    'anchored_inspection_records': ('Anchored Pavement Inspection Record Count', 'retain provenance'),
    'minimum_observed_grade': ('Minimum Anchored Pavement Inspection Category', 'retain provenance'),
    'maximum_observed_grade': ('Anchored Pavement Inspection Category', 'main condition-priority input'),
    'source_prefectures': ('Anchored Inspection Prefecture Count', 'retain provenance'),
    'Graph Source Edge ID': ('Graph Source Edge ID', 'main maintenance-unit identity'),
    'Registry Group Index': ('Registry Group Index', 'retain provenance'),
    'Baseline Source Road Length': ('Baseline Source Road Length', 'main physical-length input'),
    'Traffic Observation Key': ('Traffic Observation Key', 'shared measurement identity'),
    'Observed Traffic Volume': ('Observed Traffic Volume', 'main traffic-priority input'),
    'N13 Road Level': ('N13 Road Level', 'retain provenance'),
    'N13 Road State': ('N13 Road State', 'retain provenance'),
    'Candidate Road Source Record': ('Candidate Road Source Record', 'retain provenance'),
    'mixed_observed_grades': ('Mixed Anchored Pavement Inspection Grades', 'retain provenance'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = sha(SOP)
    module_path = ROOT / '.codex/skills/data-preprocessing/scripts/prepare_variable_list.py'
    spec = importlib.util.spec_from_file_location('variable_discovery', module_path)
    skill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skill)
    # Execute the bundled workflow on this newly accepted source only, rather
    # than rediscovering unrelated nationwide tiles and earlier analysis inputs.
    skill.discover_data_files = lambda root: [INPUT]
    skill.OUTPUT_DIR = str(OUT.relative_to(ROOT))
    sys.argv = [str(module_path), '--root', str(ROOT), '--pretty']
    skill.main()
    inventory = pd.read_csv(OUT / 'variable_list.csv', keep_default_na=False)
    source = pd.read_parquet(INPUT)
    assert len(source) == 70049 and set(source.columns) == set(NAMES)
    pd.testing.assert_series_equal(source['nearest_graph_source_edge_id'],
                                   source['Graph Source Edge ID'], check_names=False, check_dtype=False)
    rows = []
    for c in source:
        name, role = NAMES[c]
        rows.append({'source_dataset': str(INPUT.relative_to(ROOT)), 'original_name': c,
            'dtype': str(source[c].dtype), 'null_pct': float(source[c].isna().mean()*100),
            'sample_values': ', '.join(str(x) for x in source[c].head(3)),
            'feasibility_status': 'partly-testable', 'proposed_readable_name': name,
            'proposed_full_name': name, 'proposed_role': role,
            'proposed_include': 'no' if c == 'nearest_graph_source_edge_id' else 'yes',
            'proposed_coding': 'retain source values and dtype; no imputation, clipping or transformation',
            'human_confirmation': 'pending'})
    rows.append({'source_dataset': str(INPUT.relative_to(ROOT)), 'original_name': 'derived from Baseline Source Road Length',
        'dtype': 'float64', 'null_pct': 0., 'sample_values': ', '.join(str(x) for x in source['Baseline Source Road Length'].head(3)),
        'feasibility_status': 'partly-testable', 'proposed_readable_name': 'Pavement Relative Maintenance Cost',
        'proposed_full_name': 'Pavement Relative Maintenance Cost', 'proposed_role': 'main budget-accounting input',
        'proposed_include': 'yes', 'proposed_coding': 'copy physical road length; common relative cost per metre; no monetary interpretation',
        'human_confirmation': 'pending'})
    proposal = pd.DataFrame(rows)
    proposal.to_csv(OUT / 'variable_proposal.csv', index=False)
    body = ['# Pavement input variable proposal', '',
        'Sample and worst-observed-grade marker adopted by MILI-D-20261007-043. English readable names and the final selection below are proposed for review. No processed parquet or final Section 4 update has yet been produced.', '',
        '## Source group', '', f'`{INPUT.relative_to(ROOT)}`: 70,049 rows and 14 existing fields. The complete source variable list from the bundled discovery workflow is in `README.md` and `variable_list.csv`. Missingness below is recomputed on every source row, rather than only the discovery sample.', '',
        '| Original field | Proposed readable name | Include | Role | Dtype | Null % | Feasibility | Sample values |',
        '| --- | --- | --- | --- | --- | ---: | --- | --- |']
    for r in rows:
        body.append('| ' + ' | '.join(str(r[k]).replace('|','/') for k in
            ['original_name','proposed_readable_name','proposed_include','proposed_role','dtype','null_pct','feasibility_status','sample_values']) + ' |')
    body += ['', '## Construction proposal', '',
        '- Retain 13 source fields; omit the second, exactly identical road-ID column from the processed view. The original audit table retains it.',
        '- Add Pavement Relative Maintenance Cost equal to Baseline Source Road Length, in metres as a relative cost scale, following the already adopted length-cost assumption.',
        '- Use maximum anchored grade I-III as Anchored Pavement Inspection Category. It is an observed maintenance-need marker at the adopted anchors, not measured condition of every portion of a road.',
        '- Preserve existing numerical, string and Boolean types, source values and all 70,049 road units. No additional exclusion, imputation, clipping or log transformation.',
        '- Retain every original inspection record, excluded record and candidate relation in the immutable source audit. No rewriting of raw data or existing population/bridge inputs.',
        '- Planned output: `data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet`, 70,049 rows and 14 fields.',
        '- Add only the approved pavement construction subsection to AnaSOP Section 4, preserving existing demographic, network and independent bridge definitions.', '',
        '## Subsequent work', '',
        'Specify the three priority scores and budget selection/tie rules. Validate nationwide simultaneous finite impairment, including parallel roads and partial origin/destination legs. Then plan and calculate the main pavement protection outputs. Priority formulas, policy outcomes and publication figures are not produced by this input proposal.', '']
    (OUT / 'variable_proposal.md').write_text('\n'.join(body))
    assert sha(SOP) == before
    v = {'status': 'passed_scoped_variable_proposal', 'authorization': 'MILI-D-20261007-043',
        'bundled_discovery_executed': True, 'scoped_source_datasets': 1, 'source_fields_discovered': len(inventory),
        'source_rows_reviewed': len(source), 'proposed_retained_source_fields': 13, 'proposed_derived_fields': 1,
        'duplicate_identity_verified_exactly': True, 'names_and_selection_confirmation': 'pending',
        'processed_parquet_written': False, 'anasop_unchanged': True,
        'input_sha256': sha(INPUT), 'sop_sha256': before}
    (OUT / 'proposal_validation.json').write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(v,indent=2))


if __name__ == '__main__':
    main()
