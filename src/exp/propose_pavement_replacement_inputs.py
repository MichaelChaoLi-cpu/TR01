"""Prepare transparent, unapproved national scenario-input assignments for review.

Only calibration lookup proposals and readable-variable proposals are written.
No global road input, simulated grade or maintenance policy is constructed.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/pavement-scenario-domain-review'
STRATA = OUT / 'strata'
MIN_SUPPORT = 30
MUNICIPAL_MULTIPLIER = 0.10


def main():
    coverage = pd.read_csv(STRATA / 'road_class_width_coverage.csv')
    traffic = pd.read_csv(STRATA / 'traffic_calibration_observations.csv', dtype={'Traffic Observation Key': str})
    condition = pd.read_csv(STRATA / 'condition_calibration_markers.csv')
    rows = []

    def traffic_reference(category, width):
        candidates = [
            (f'class-{category}_width-{width}', traffic[traffic['Road Classification Code'].eq(category)&traffic['Road Width Code'].eq(width)]),
            (f'class-{category}', traffic[traffic['Road Classification Code'].eq(category)]),
            ('pooled-observed', traffic),
        ]
        for label, data in candidates:
            unique = data.drop_duplicates('Traffic Observation Key')
            if len(unique) >= MIN_SUPPORT:
                return label, len(unique), float(unique['Observed Traffic Volume'].median())
        raise RuntimeError('No traffic calibration pool meets proposed support')

    def condition_reference(category, width):
        candidates = [
            (f'class-{category}_width-{width}', condition[condition['Road Classification Code'].eq(category)&condition['Road Width Code'].eq(width)]),
            (f'class-{category}', condition[condition['Road Classification Code'].eq(category)]),
            (f'borrowed-width-{width}', condition[condition['Road Width Code'].eq(width)]),
            ('pooled-observed-marker', condition),
        ]
        for label, data in candidates:
            if len(data) >= MIN_SUPPORT:
                counts = data['Anchored Pavement Inspection Category'].value_counts()
                probabilities = np.array([counts.get(i,0) / len(data) for i in [1,2,3]])
                np.testing.assert_allclose(probabilities.sum(), 1)
                return label, len(data), probabilities
        raise RuntimeError('No condition calibration pool meets proposed support')

    for record in coverage.to_dict('records'):
        category = int(record['Road Classification Code'])
        width = int(record['Road Width Code'])
        # Class 3 (municipal etc.) has very sparse, selected traffic matches.
        # The multiplier is an explicit scenario assumption, not an estimate.
        if category in [3,5]:
            label, n, median = traffic_reference(2, width)
            multiplier = MUNICIPAL_MULTIPLIER
            traffic_basis = 'assumed-scaled-prefectural-reference'
        else:
            label, n, median = traffic_reference(category, width)
            multiplier = 1.0
            traffic_basis = 'assumed-borrowed-observed-median'
        clabel, cn, probabilities = condition_reference(category, width)
        rows.append({
            'Road Classification Code': category, 'Road Width Code': width,
            'Candidate Road Count': int(record['Candidate Road Count']),
            'Traffic Calibration Reference': label,
            'Traffic Distinct Calibration Observations': n,
            'Traffic Reference Median': median, 'Traffic Scenario Multiplier': multiplier,
            'Proposed Missing Traffic Volume': median * multiplier,
            'Missing Traffic Basis': traffic_basis,
            'Condition Calibration Reference': clabel,
            'Condition Calibration Road Markers': cn,
            'Proposed Condition I Probability': float(probabilities[0]),
            'Proposed Condition II Probability': float(probabilities[1]),
            'Proposed Condition III Probability': float(probabilities[2]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'stratified_assignment_proposal.csv', index=False)
    variables = [
        ('Graph Source Edge ID','existing identity','Unchanged original source-road identity.'),
        ('Baseline Source Road Length','existing geometric input','Unchanged accepted candidate-source length in reference metres.'),
        ('Road Classification Code','new categorical source input','Original official classification code; no ranking by code order.'),
        ('Road Width Code','new categorical source input','Original official width code; preserve unknown code 6.'),
        ('Observed Traffic Volume','existing nullable observed input','Retain accepted measured correspondence; remain missing outside it.'),
        ('Anchored Pavement Inspection Category','existing nullable observed marker','Retain accepted worst anchored category only on the original common sample; remain missing elsewhere.'),
        ('Scenario Traffic Volume','new complete scenario input','Observed value if available; otherwise the proposed class/width reference and multiplier.'),
        ('Scenario Pavement Need Category','new complete scenario marker','Observed anchored marker if available; otherwise a categorical draw from the proposed probabilities.'),
        ('Scenario Traffic Basis','new provenance input','Distinguish observed correspondence from each hypothetical traffic construction.'),
        ('Scenario Condition Basis','new provenance input','Distinguish observed anchored marker from each hypothetical categorical construction.'),
        ('Scenario Traffic Calibration Stratum','new provenance input','Record the calibration reference used for a missing traffic value.'),
        ('Scenario Condition Calibration Stratum','new provenance input','Record the calibration reference used for a missing condition marker.'),
        ('Scenario Deterioration Eligible','new domain input','True for every retained candidate vehicle-source record in the proposed national domain.'),
        ('Scenario Maintenance Eligible','new domain input','Same national domain as deterioration eligibility.'),
        ('Scenario Relative Maintenance Cost','new cost input','Baseline Source Road Length, once per original source-road unit; not official unique mileage or monetary expenditure.'),
        ('Scenario Priority Tie Key','new ranking input','Independent fixed-seed uniform key shared across all three rules; final ties use original source identity.'),
    ]
    pd.DataFrame(variables, columns=['Proposed Readable Name','Role','Proposed Construction']).to_csv(OUT / 'readable_input_proposal.csv', index=False)
    spec = {
        'status':'proposal_awaiting_human_confirmation',
        'strategy_authorization':'MILI-D-20261007-056',
        'specific_assignments_authorized':False,
        'candidate_road_records':21415800,
        'minimum_calibration_observations':MIN_SUPPORT,
        'municipal_and_other_missing_traffic_multiplier':MUNICIPAL_MULTIPLIER,
        'observed_values_preserved':True,'missing_observed_columns_left_missing':True,
        'traffic_fallback':'same class and width, then same class, then pooled observed; municipal/other use 10% of the corresponding prefectural reference',
        'condition_fallback':'same class and width, then same class, then borrowed width, then pooled observed markers; minimum 30 unique road markers',
        'condition_generation':'One categorical draw per missing road, using seed 20261007, source-ID order and the stated probabilities; no spatial-correlation or real-condition claim.',
        'tie_generation':'Independent seed 20261008 in source-ID order; common key across all rules, then source ID.',
        'traffic_priority_proposal':'Descending scenario volume: traffic-weighted impairment proxy is volume times road length, and length-relative cost cancels the length factor.',
        'condition_priority_proposal':'Category III, II, I; shared random tie key, then source ID. Do not rank ties by source-record length.',
        'access_priority_proposal':'Exact isolated restoration gain per cost at 25% all-road impairment; shared tie key then source ID.',
        'selection':'Adopted longest complete affordable prefix, no splitting or skipping; budgets 1%,5%,10%,20% of the expanded source-length cost.',
        'joint_evaluation':'All unselected eligible roads impaired simultaneously; primary increment 25%, fixed-set sensitivities 10% and 50%; original origins, facilities, population and groups unchanged.',
        'interpretation':'Hypothetical national policy comparison; class/width borrowing and municipal multiplier are assumptions, not validated estimates of missing observations.',
        'variables':16,'new_readable_names':12,'global_inputs_constructed':False,
    }
    (OUT / 'input_construction_proposal.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({'status':spec['status'],'strata':len(table),'variables':16,'new_names':12,'no_global_inputs_or_policy_results_created':True},indent=2))


if __name__ == '__main__':
    main()
