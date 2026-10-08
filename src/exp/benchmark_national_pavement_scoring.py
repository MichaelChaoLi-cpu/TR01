"""Runtime diagnostic for isolated restoration on the all-road reference.

Only a sampled subset is indexed for restoration; every road is impaired in
the reference. No assumed traffic/condition inputs or policy sets are generated.
This is a performance experiment, not the expanded production implementation.
"""
from pathlib import Path
import sys
import json
import time
import resource
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/analyses'))
from pavement_inputs import load_inputs
from pavement_impedance import PavementImpedance
from road_service_loss import RoadServiceLoss

OUT = ROOT/'data/exp/pavement-scenario-domain-review/runtime-benchmark'


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print('Loading fixed national network and origins', flush=True)
    d = load_inputs()
    loaded = time.perf_counter()
    rng = np.random.default_rng(20261007)
    # Uniform source-record sampling for the extrapolation; entry-road stress
    # samples are reported separately and not treated as representative roads.
    uniform = rng.choice(len(d['lengths']), size=3000, replace=False)
    origin = rng.choice(np.unique(d['origin_edges']), size=200, replace=False)
    facility = rng.choice(np.unique(d['facility_edges']), size=200, replace=False)
    eligible = np.unique(np.r_[uniform, origin, facility])
    e = PavementImpedance(d['pairs'], d['lengths'], d['origin_edges'], d['origin_positions'],
                         d['facility_edges'], d['facility_positions'], eligible, graph=d['graph'])
    # The constructor indexes only sampled pairs. Scale all other pair minima
    # and all graph entries, while flags make partial physical legs all-road.
    e.outside_min *= 1.25
    e.ref.graph.data *= 1.25
    e.eligible_flag[:] = True
    e.configure(.25)
    reference = e.distances()
    np.testing.assert_allclose(reference, d['sample']['Candidate Medical Network Distance']*1.25,
                               rtol=1e-12, atol=1e-7)
    setup = time.perf_counter()
    print('All-road reference reproduced; timing sampled restorations', flush=True)
    results = {}
    changed_example = None
    sampled_values = None
    for name, roads in [('uniform_roads', uniform), ('origin_entry_roads', origin),
                        ('medical_entry_roads', facility)]:
        timings = []
        changed_nodes = []
        count_affected = []
        for edge in roads:
            tick = time.perf_counter()
            indices, values = e.restore_one(int(edge), return_changes=True)
            # Include the benefit aggregation used by the production scorer.
            extra = reference[indices]-values
            assert (extra >= -1e-7).all()
            _gain = float(np.sum(d['population'][indices]*np.maximum(extra, 0))/1000)
            timings.append(time.perf_counter()-tick)
            changed_nodes.append(e.last_diagnostics['changed_nodes'])
            count_affected.append(len(indices))
            if changed_example is None and len(indices) and np.any(extra > 1e-7):
                changed_example = int(edge)
                sampled_values = reference.copy()
                sampled_values[indices] = values
        results[name] = {'roads': len(roads), 'total_seconds': float(sum(timings)),
                         'mean_seconds': float(np.mean(timings)),
                         'median_seconds': float(np.median(timings)),
                         'maximum_seconds': float(max(timings)),
                         'maximum_changed_nodes': int(max(changed_nodes)),
                         'maximum_affected_origins': int(max(count_affected))}
    scored = time.perf_counter()
    # One independent complete reroute verifies the sampled restoration domain.
    assert changed_example is not None
    edge = changed_example
    graph = d['graph'].copy()
    graph.data *= 1.25
    u, v = map(int, d['pairs'][edge])
    if u != v:
        for a, b in [(u, v), (v, u)]:
            left, right = graph.indptr[a:a+2]
            pos = int(left)+int(np.searchsorted(graph.indices[left:right], b))
            assert graph.indices[pos] == b
            graph.data[pos] = min(graph.data[pos], float(d['lengths'][edge]))
    lengths = np.asarray(d['lengths'])*1.25
    lengths[edge] = d['lengths'][edge]
    origin_positions = d['origin_positions']*np.where(d['origin_edges']==edge, 1., 1.25)
    facility_positions = d['facility_positions']*np.where(d['facility_edges']==edge, 1., 1.25)
    oracle = RoadServiceLoss(d['pairs'], lengths, d['origin_edges'], origin_positions,
                            d['facility_edges'], facility_positions, graph=graph)
    tick = time.perf_counter()
    full = oracle.distances()
    routing_seconds = time.perf_counter()-tick
    np.testing.assert_allclose(full, sampled_values, rtol=1e-12, atol=1e-7)
    estimate = results['uniform_roads']['mean_seconds']*len(d['lengths'])
    report = {'status': 'passed_sampled_all_road_runtime_diagnostic',
              'candidate_road_records': len(d['lengths']),
              'network_input_loading_seconds': loaded-start,
              'sample_engine_setup_and_reference_seconds': setup-loaded,
              'sampled_scoring_seconds': scored-setup, 'samples': results,
              'uniform_mean_linear_scoring_seconds': estimate,
              'uniform_mean_linear_scoring_hours': estimate/3600,
              'single_full_reroute_seconds': routing_seconds,
              '36_reroutes_only_linear_seconds': routing_seconds*36,
              'independent_restoration_oracle_max_error_reference_metres': float(np.max(np.abs(full-sampled_values))),
              'oracle_road': edge, 'elapsed_seconds': time.perf_counter()-start,
              'maximum_resident_set_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limits': ['Single benchmark pass; extrapolation is not a completion guarantee.',
                         'National all-pair indexing, input construction, sorting, storage and map aggregation are excluded.',
                         'Only sampled pairs are indexed; expanded production memory and runtime remain to validate.',
                         'This diagnostic creates no traffic, condition, rankings, selections or policy outcomes.'],
              'formal_policy_results_generated': False, 'source_and_processed_files_modified': False}
    (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
