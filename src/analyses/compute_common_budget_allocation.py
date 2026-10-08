"""Human-adopted fixed-prefix, common-relative-budget policy comparison.

Single-road events are mutually exclusive and equally weighted. This produces
conditional average loss protection, not unique beneficiaries or annual risk.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/exp/common-budget-allocation'
AUTH = 'MILI-D-20261007-026'
OUTCOMES = {
    'Distance Burden Protection Share': 'Additional Medical Distance Burden',
    'Reachability Protection Share': 'Lost Medical Path Population',
    'Higher Ageing Reachability Protection Share': 'Higher Ageing Lost Medical Path Population',
    'Distance Disadvantaged Reachability Protection Share': 'Distance Disadvantaged Lost Medical Path Population',
    'Higher Ageing Distance Burden Protection Share': 'Higher Ageing Additional Medical Distance Burden',
    'Distance Disadvantaged Distance Burden Protection Share': 'Distance Disadvantaged Additional Medical Distance Burden',
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'data/processed/common-budget/conditional_group_losses_preprocessed.parquet'
    before = digest(path)
    u = pd.read_parquet(path)
    assert len(u) == 46688 and u['Graph Source Edge ID'].is_unique
    assert u['Relative Intervention Cost'].gt(0).all()
    c = u['Relative Intervention Cost']
    u['Traffic Per Relative Cost'] = u['Observed Traffic Volume'] / c
    u['Reachability Loss Per Relative Cost'] = u['Lost Medical Path Population'] / c
    u['Distance Burden Per Relative Cost'] = u['Additional Medical Distance Burden'] / c
    orders = {
        'Traffic priority': u.sort_values(['Traffic Per Relative Cost', 'Candidate Road Source Record'], ascending=[False, True], kind='stable'),
        'Condition priority': u.sort_values(['Bridge Inspection Category', 'Relative Intervention Cost', 'Candidate Road Source Record'], ascending=[False, True, True], kind='stable'),
        'Access priority': u.sort_values(['Reachability Loss Per Relative Cost', 'Distance Burden Per Relative Cost', 'Candidate Road Source Record'], ascending=[False, False, True], kind='stable'),
    }
    total = float(c.sum())
    denominators = {name: float(u[col].sum()) for name, col in OUTCOMES.items()}
    assert all(x > 0 for x in denominators.values()), 'An outcome denominator is zero; preserve undefined measures and review the panel scope.'
    rows, ranks = [], []
    for rule, frame in orders.items():
        cumulative_cost = np.r_[0., np.cumsum(frame['Relative Intervention Cost'].to_numpy())]
        cumulative_loss = {name: np.r_[0., np.cumsum(frame[col].to_numpy())] for name, col in OUTCOMES.items()}
        counts = []
        for percent in range(101):
            share = percent / 100.0
            budget = share * total
            k = len(frame) if percent == 100 else int(np.searchsorted(cumulative_cost, budget, side='right') - 1)
            spent = float(cumulative_cost[k])
            assert spent <= budget + 1e-10 * total
            row = {'Maintenance Priority': rule, 'Budget Share': share, 'Requested Relative Budget': budget,
                   'Actual Relative Spending': spent, 'Unused Relative Budget': max(0., budget-spent),
                   'Selected Maintenance Unit Count': k,
                   'Avoided Distance Burden Per Event': float(cumulative_loss['Distance Burden Protection Share'][k] / len(frame)),
                   'Avoided Lost Path Population Per Event': float(cumulative_loss['Reachability Protection Share'][k] / len(frame))}
            for name in OUTCOMES:
                row[name] = float(cumulative_loss[name][k] / denominators[name])
            # Independent direct-sum checks at representative budgets validate prefix accounting.
            if percent in [0, 1, 5, 10, 25, 50, 75, 90, 100]:
                direct_cost = float(frame.iloc[:k]['Relative Intervention Cost'].sum())
                np.testing.assert_allclose(direct_cost, spent, rtol=1e-11, atol=1e-8)
                assert k == len(frame) or spent + float(frame.iloc[k]['Relative Intervention Cost']) > budget
                for name, col in OUTCOMES.items():
                    expected = float(frame.iloc[:k][col].sum() / denominators[name])
                    np.testing.assert_allclose(row[name], expected, rtol=1e-11, atol=1e-12)
            rows.append(row)
            counts.append(k)
        assert np.all(np.diff(counts) >= 0)
        ranked = frame[['Graph Source Edge ID', 'Candidate Road Source Record']].copy()
        ranked['Maintenance Priority'] = rule
        ranked['Priority Position'] = np.arange(1, len(frame) + 1)
        ranks.append(ranked)
    curves = pd.DataFrame(rows)
    for rule, frame in curves.groupby('Maintenance Priority'):
        for name in OUTCOMES:
            assert frame[name].iloc[0] == 0
            np.testing.assert_allclose(frame[name].iloc[-1], 1., atol=1e-11)
            assert (np.diff(frame[name]) >= -1e-12).all()
            assert frame[name].between(-1e-12, 1+1e-12).all()
    assert digest(path) == before
    curves.to_parquet(OUT / 'protection_curves_construction.parquet', index=False)
    curves.to_csv(OUT / 'protection_curves.csv', index=False)
    pd.concat(ranks, ignore_index=True).to_parquet(OUT / 'priority_lists.parquet', index=False)
    validation = {'status': 'passed', 'authorization': AUTH, 'common_units': len(u), 'rules': list(orders),
                  'curve_rows': len(curves), 'budgets_per_rule': 101, 'budget_feasibility_passed': True,
                  'prefix_membership_and_direct_sums_verified': True, 'nested_selections': True,
                  'nondecreasing_outcome_curves': True, 'zero_and_full_budget_endpoints_passed': True,
                  'relative_total_cost': total, 'outcome_denominators_across_mutually_exclusive_events': denominators,
                  'maximum_unspent_share': float(curves['Unused Relative Budget'].max() / total),
                  'scientific_input_unchanged': True, 'source': str(path.relative_to(ROOT)), 'source_sha256': before,
                  'observed_cost_or_effect_claim': False, 'annual_failure_probability_claim': False,
                  'unique_beneficiary_claim': False, 'allocation_is_optimal': False}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
