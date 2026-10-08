"""Exact fixed-system population reweighting, adopted in research decision 036.

Reuse the accepted road-removal kernel once per maintenance unit across all years.
Save reproducible constructions in exp; data-preprocessing exports final inputs.
"""
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analyses'))
from road_service_loss import RoadServiceLoss
from road_service_loss_dynamic import DynamicRoadServiceLoss

OUT = ROOT / 'data/exp/demographic-stability'
PROC = ROOT / 'data/processed/national-network-rule-validation'
CACHE = ROOT / 'data/exp/road-service-loss-validation/dynamic'
YEARS = [2020, 2030, 2040, 2050]
GROUPS = ['All matched residents', 'Higher-ageing areas', 'Distance-disadvantaged areas']
LOSS = ['Lost Medical Path Population', 'Higher Ageing Lost Medical Path Population',
        'Distance Disadvantaged Lost Medical Path Population']
BURDEN = ['Additional Medical Distance Burden', 'Higher Ageing Additional Medical Distance Burden',
          'Distance Disadvantaged Additional Medical Distance Burden']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, value):
    part = path.with_suffix('.part')
    part.write_text(json.dumps(value, indent=2) + '\n')
    part.replace(path)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    unit_path = ROOT / 'data/processed/common-budget/conditional_maintenance_units_preprocessed.parquet'
    mesh_path = ROOT / 'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet'
    scenario_path = ROOT / 'data/processed/demographic-stability/population_scenarios_preprocessed.parquet'
    units = pd.read_parquet(unit_path)
    sample = pd.read_parquet(mesh_path)
    scenarios = pd.read_parquet(scenario_path)
    hashes = json.loads((CACHE / 'validation.json').read_text())['source_hashes_verified_unchanged'].copy()
    for path in [unit_path, mesh_path, scenario_path]:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for rel, expected in hashes.items():
        assert digest(ROOT / rel) == expected, rel
    config = {'authorization': 'MILI-D-20261007-036', 'sources': hashes,
              'script_sha256': digest(Path(__file__)), 'kernel_sha256': digest(ROOT / 'src/analyses/road_service_loss_dynamic.py')}
    config_path = OUT / 'configuration.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    else:
        save_json(config_path, config)
    # Explicitly exclude unmatched origins; zeros here are accumulator weights only.
    weights = np.zeros((len(sample), len(YEARS)))
    for j, year in enumerate(YEARS):
        frame = scenarios[scenarios['Population-Change Scenario'].eq(year)]
        assert frame['Mesh Code'].reset_index(drop=True).equals(sample['Mesh Code'])
        for column in ['Higher Ageing Area', 'Baseline Distance Disadvantaged Area']:
            np.testing.assert_array_equal(frame[column], sample[column])
        matched = frame['Forecast Matched Origin'].to_numpy()
        values = frame['Scenario Population'].to_numpy()
        assert np.isnan(values[~matched]).all()
        weights[matched, j] = values[matched]
    masks = [np.ones(len(sample), dtype=bool), sample['Higher Ageing Area'].to_numpy(),
             sample['Baseline Distance Disadvantaged Area'].to_numpy()]
    all_weights = np.column_stack([weights] + [weights * m[:, None] for m in masks[1:]]
                                 + [weights * (~m)[:, None] for m in masks[1:]]
                                 + [sample['Local Total Population'].to_numpy()])
    np.testing.assert_allclose(weights.sum(axis=0), [float(scenarios.loc[scenarios['Population-Change Scenario'].eq(y), 'Scenario Population'].sum()) for y in YEARS])
    points = pd.read_csv(ROOT / 'data/exp/national-population-entry/point_attachment_diagnostics.csv',
                        usecols=['point_id', 'dataset', 'category', 'routing_eligible', 'edge_id', 'along_m'],
                        dtype={'point_id': str, 'category': str}, keep_default_na=False,
                        low_memory=False, float_precision='round_trip')
    points['along_m'] = pd.to_numeric(points['along_m'], errors='coerce')
    origins = points.set_index('point_id').loc['mesh:' + sample['Mesh Code'].astype(str)]
    facilities = points[points['dataset'].eq('medical') & points['category'].isin(['1', '2']) & points['routing_eligible']]
    ref = RoadServiceLoss(np.load(PROC / 'edge_nodes_exact.npy', mmap_mode='r'),
                         np.load(PROC / 'edge_lengths.npy', mmap_mode='r'),
                         origins['edge_id'].to_numpy(), origins['along_m'].to_numpy(),
                         facilities['edge_id'].to_numpy(), facilities['along_m'].to_numpy(),
                         graph=load_npz(PROC / 'graph_exact.npz'))
    del points, origins, facilities
    gc.collect()
    fast = DynamicRoadServiceLoss(ref, replacements=np.load(CACHE / 'source_removal_replacement_weights.npy', mmap_mode='r'))
    np.testing.assert_allclose(fast.baseline, sample['Candidate Medical Network Distance'], rtol=1e-11, atol=1e-7)
    critical = np.load(CACHE / 'critical_source_road_screen.npy', mmap_mode='r')
    order = np.argsort(ref.origin_edges, kind='stable')
    origin_edges = ref.origin_edges[order]
    records = []
    max_l_error, max_t_error = 0., 0.
    for first in range(0, len(units), 500):
        last = min(first + 500, len(units))
        path = OUT / f'loss_batch_{first:06d}.parquet'
        if path.exists():
            records.append(pd.read_parquet(path))
            continue
        rows = []
        for _, unit in units.iloc[first:last].iterrows():
            edge = int(unit['Graph Source Edge ID'])
            if critical[edge]:
                ix, distances = fast.distances(edge, return_changes=True)
            else:
                left, right = np.searchsorted(origin_edges, edge, 'left'), np.searchsorted(origin_edges, edge, 'right')
                ix = order[left:right]
                distances = np.full(len(ix), np.inf)
            finite = np.isfinite(distances)
            lost, reach = ix[~finite], ix[finite]
            delta = distances[finite] - fast.baseline[reach]
            assert (delta >= -1e-7).all()
            delta[np.abs(delta) <= 1e-7] = 0
            losses = all_weights[lost].sum(axis=0)
            burdens = (all_weights[reach] * (delta / 1000)[:, None]).sum(axis=0)
            le = abs(losses[-1] - float(unit['Lost Medical Path Population']))
            te = abs(burdens[-1] - float(unit['Additional Medical Distance Burden']))
            max_l_error, max_t_error = max(max_l_error, le), max(max_t_error, te)
            assert le == 0
            np.testing.assert_allclose(burdens[-1], unit['Additional Medical Distance Burden'], rtol=1e-10, atol=1e-8)
            for g, complement in [(1, 3), (2, 4)]:
                np.testing.assert_allclose(losses[g*4:(g+1)*4] + losses[complement*4:(complement+1)*4], losses[:4], rtol=1e-11, atol=1e-8)
                np.testing.assert_allclose(burdens[g*4:(g+1)*4] + burdens[complement*4:(complement+1)*4], burdens[:4], rtol=1e-11, atol=1e-8)
            for j, year in enumerate(YEARS):
                row = {'Graph Source Edge ID': edge, 'Population-Change Scenario': year}
                for g in range(3):
                    row[LOSS[g]] = losses[g*4+j]
                    row[BURDEN[g]] = burdens[g*4+j]
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        records.append(frame)
        save_json(OUT / 'progress.json', {'status': 'running', 'completed_units': last, 'total_units': len(units), 'elapsed_seconds': time.perf_counter()-started})
        if first % 5000 == 0:
            print(f'Scenario loss accounting {last:,}/{len(units):,}; {time.perf_counter()-started:.1f} s', flush=True)
    losses = pd.concat(records, ignore_index=True)
    assert len(losses) == len(units)*4 and not losses.duplicated(['Graph Source Edge ID', 'Population-Change Scenario']).any()
    assert losses[LOSS + BURDEN].ge(0).all().all()
    losses.to_parquet(OUT / 'scenario_losses_construction.parquet', index=False)
    curves, ranks, sets, denominators = [], [], {}, {}
    original_ranks = pd.read_parquet(ROOT / 'data/exp/common-budget-allocation/priority_lists.parquet')
    total_cost = float(units['Relative Intervention Cost'].sum())
    for year in YEARS:
        u = units.drop(columns=['Lost Medical Path Population', 'Additional Medical Distance Burden']).merge(
            losses[losses['Population-Change Scenario'].eq(year)], on='Graph Source Edge ID', validate='one_to_one', sort=False)
        c = u['Relative Intervention Cost']
        u['Traffic Per Relative Cost'] = u['Observed Traffic Volume']/c
        u['Reachability Loss Per Relative Cost'] = u[LOSS[0]]/c
        u['Distance Burden Per Relative Cost'] = u[BURDEN[0]]/c
        orders = {
            'Traffic priority': u.sort_values(['Traffic Per Relative Cost', 'Candidate Road Source Record'], ascending=[False, True], kind='stable'),
            'Condition priority': u.sort_values(['Bridge Inspection Category', 'Relative Intervention Cost', 'Candidate Road Source Record'], ascending=[False, True, True], kind='stable'),
            'Access priority': u.sort_values(['Reachability Loss Per Relative Cost', 'Distance Burden Per Relative Cost', 'Candidate Road Source Record'], ascending=[False, False, True], kind='stable')}
        denominators[year] = dict(zip(GROUPS, u[LOSS].sum().astype(float)))
        for rule, ranked in orders.items():
            if rule != 'Access priority':
                np.testing.assert_array_equal(ranked['Graph Source Edge ID'], original_ranks.loc[original_ranks['Maintenance Priority'].eq(rule), 'Graph Source Edge ID'])
            costs = np.r_[0., np.cumsum(ranked['Relative Intervention Cost'].to_numpy())]
            sums = np.vstack([np.zeros(3), np.cumsum(ranked[LOSS].to_numpy(), axis=0)])
            counts = []
            for percent in range(101):
                budget = total_cost * percent/100
                k = len(ranked) if percent == 100 else int(np.searchsorted(costs, budget, 'right')-1)
                assert costs[k] <= budget + 1e-10*total_cost
                assert k == len(ranked) or costs[k+1] > budget
                counts.append(k)
                if percent in [0, 1, 5, 10, 20, 100]:
                    np.testing.assert_allclose(ranked.iloc[:k]['Relative Intervention Cost'].sum(), costs[k], rtol=1e-11, atol=1e-8)
                    np.testing.assert_allclose(ranked.iloc[:k][LOSS].sum(), sums[k], rtol=1e-11, atol=1e-8)
                for g, name in enumerate(GROUPS):
                    denominator = denominators[year][name]
                    q = float(sums[k, g]/denominator) if denominator > 0 else np.nan
                    curves.append({'Population-Change Scenario': year, 'Maintenance Priority': rule,
                                   'Population Group': name, 'Budget Share': percent/100,
                                   'Distribution of Protected Access': q,
                                   'Requested Relative Budget': budget, 'Actual Relative Spending': costs[k],
                                   'Unused Relative Budget': max(0., budget-costs[k]),
                                   'Selected Maintenance Unit Count': k, 'Event Loss Denominator': denominator})
                if rule == 'Access priority' and percent in [1, 5, 10, 20]:
                    sets[year, percent] = set(ranked.iloc[:k]['Graph Source Edge ID'])
            assert np.all(np.diff(counts) >= 0)
            r = ranked[['Graph Source Edge ID', 'Candidate Road Source Record']].copy()
            r['Population-Change Scenario'], r['Maintenance Priority'] = year, rule
            r['Priority Position'] = np.arange(1, len(ranked)+1)
            ranks.append(r)
    curve = pd.DataFrame(curves)
    for _, group in curve.groupby(['Population-Change Scenario', 'Maintenance Priority', 'Population Group'], sort=False):
        q = group['Distribution of Protected Access'].to_numpy()
        assert np.isfinite(q).all() and q[0] == 0 and abs(q[-1]-1)<1e-10
        assert (np.diff(q) >= -1e-12).all() and ((q>=-1e-12)&(q<=1+1e-12)).all()
    jaccard = []
    for year in YEARS:
        for percent in [1, 5, 10, 20]:
            current, baseline = sets[year, percent], sets[2020, percent]
            intersection, union = len(current & baseline), len(current | baseline)
            jaccard.append({'Population-Change Scenario': year, 'Budget Share': percent/100,
                            'Priority Stability': intersection/union if union else np.nan,
                            'Intersection Unit Count': intersection, 'Union Unit Count': union,
                            'Selected Maintenance Unit Count': len(current), 'Reference Selected Unit Count': len(baseline)})
    stability = pd.DataFrame(jaccard)
    assert stability['Priority Stability'].between(0,1).all()
    assert stability.loc[stability['Population-Change Scenario'].eq(2020),'Priority Stability'].eq(1).all()
    curve.to_parquet(OUT / 'scenario_protection_construction.parquet', index=False)
    stability.to_parquet(OUT / 'priority_stability_construction.parquet', index=False)
    pd.concat(ranks, ignore_index=True).to_parquet(OUT / 'priority_lists.parquet', index=False)
    curve[curve['Budget Share'].eq(.01)].to_csv(OUT / 'one_percent_protection.csv', index=False)
    stability.to_csv(OUT / 'priority_stability.csv', index=False)
    for rel, expected in hashes.items():
        assert digest(ROOT / rel) == expected, rel
    validation = {'status': 'passed', 'authorization': 'MILI-D-20261007-036', 'common_units': len(units),
                  'matched_origins': 175485, 'years': YEARS, 'baseline_reproduced': True,
                  'original_unit_losses_reproduced': True, 'maximum_original_population_loss_error': max_l_error,
                  'maximum_original_distance_burden_error': max_t_error, 'group_complements_reconciled': True,
                  'traffic_condition_lists_fixed_and_verified': True, 'direct_prefix_sums_verified': True,
                  'budgets_feasible_and_selections_nested': True, 'all_protection_denominators_positive': True,
                  'nondecreasing_protection_and_zero_full_endpoints_verified': True,
                  'relative_total_cost': total_cost, 'loss_denominators': denominators,
                  'source_hashes_unchanged': hashes, 'elapsed_seconds': time.perf_counter()-started}
    save_json(OUT / 'validation.json', validation)
    save_json(OUT / 'progress.json', {'status': 'complete', 'completed_units': len(units), 'total_units': len(units)})
    print(json.dumps({k:v for k,v in validation.items() if k != 'source_hashes_unchanged'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
