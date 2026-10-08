"""Audit intervention-domain restrictions and a diagnostic uniform-network reference.

The uniform-cost experiment isolates the domain restriction. It is not an adopted
replacement maintenance policy, calibrated deterioration model or final result.
All existing source, processed and publication outputs remain unchanged.
"""
from pathlib import Path
import sys
import json
import hashlib
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analyses'))
from pavement_inputs import load_inputs
from road_service_loss import RoadServiceLoss

OUT = ROOT / 'data/exp/pavement-scenario-domain-review'


def describe(costs, baseline, population):
    extra = costs - baseline
    assert np.isfinite(extra).all() and (extra >= -1e-7).all()
    extra[np.abs(extra) <= 1e-7] = 0
    zero = extra == 0
    return {
        'meshes': len(extra), 'zero_meshes': int(zero.sum()),
        'zero_mesh_share': float(zero.mean()),
        'zero_population': int(population[zero].sum()),
        'zero_population_share': float(population[zero].sum() / population.sum()),
        'positive_population': int(population[~zero].sum()),
        'population_total': int(population.sum()),
        'residual_burden_resident_reference_km': float(np.sum(population * extra) / 1000),
        'mean_excess_reference_km_per_resident': float(np.sum(population * extra) / 1000 / population.sum()),
        'maximum_excess_reference_km': float(extra.max() / 1000),
    }


def main():
    tick = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print('Loading accepted network, road sample and fixed origins', flush=True)
    data = load_inputs()
    baseline = data['sample']['Candidate Medical Network Distance'].to_numpy()
    population = data['population']
    lengths = data['lengths']
    common = data['roads']
    crosswalk = pd.read_parquet(ROOT / 'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet',
                               columns=['Graph Source Edge ID', 'Baseline Source Road Length'])
    assert crosswalk['Graph Source Edge ID'].is_unique
    total_length = float(np.sum(lengths))
    coverage = {
        'candidate_vehicle_road_records': len(lengths),
        'candidate_vehicle_road_record_length_km': total_length / 1000,
        'common_policy_roads': len(common),
        'common_policy_length_km': float(common['Pavement Relative Maintenance Cost'].sum() / 1000),
        'common_record_share': len(common) / len(lengths),
        'common_length_share': float(common['Pavement Relative Maintenance Cost'].sum() / total_length),
        'traffic_matched_roads': len(crosswalk),
        'traffic_matched_length_km': float(crosswalk['Baseline Source Road Length'].sum() / 1000),
        'traffic_matched_length_share': float(crosswalk['Baseline Source Road Length'].sum() / total_length),
        'length_interpretation': 'Sum of candidate source-road geometry records, not official unique physical pavement mileage.',
    }
    original_costs = np.load(ROOT / 'data/exp/pavement-policy-computation/cases/reference_0.25.npy')
    original = describe(original_costs, baseline, population)
    # Uniform scaling leaves route ordering unchanged: the entire physical
    # graph, partial origin/destination legs and same-edge direct costs scale.
    # Re-run Dijkstra independently rather than treating this identity as data.
    factor = 1.25
    data['graph'].data *= factor
    scaled_lengths = np.asarray(lengths) * factor
    print('Routing diagnostic: every candidate vehicle-road cost multiplied by 1.25', flush=True)
    ref = RoadServiceLoss(data['pairs'], scaled_lengths,
                          data['origin_edges'], data['origin_positions'] * factor,
                          data['facility_edges'], data['facility_positions'] * factor,
                          graph=data['graph'])
    costs = ref.distances()
    expected = factor * baseline
    error = float(np.max(np.abs(costs - expected)))
    np.testing.assert_allclose(costs, expected, rtol=1e-12, atol=1e-7)
    np.save(OUT / 'diagnostic_uniform_all_road_costs.npy', costs)
    diagnostic = describe(costs, baseline, population)
    pd.DataFrame([
        {'scenario': 'Adopted common-sample impairment', **original},
        {'scenario': 'Diagnostic uniform all-road impairment', **diagnostic},
    ]).to_csv(OUT / 'reference_domain_comparison.csv', index=False)
    pd.DataFrame([coverage]).to_csv(OUT / 'road_domain_coverage.csv', index=False)
    result = {
        'status': 'passed_domain_audit_and_uniform_scaling_diagnostic',
        'coverage': coverage, 'original_reference': original,
        'diagnostic_reference': diagnostic,
        'uniform_scaling_oracle_maximum_error_reference_m': error,
        'diagnostic_adopted_as_policy': False, 'maintenance_policies_reranked': False,
        'previous_outputs_recomputed': False,
        'source_and_processed_files_modified': False,
        'diagnostic_scope': 'Uniform all-road scaling isolates the scenario domain; it does not establish real-world impairment or repair effects.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds': time.perf_counter() - tick,
    }
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
