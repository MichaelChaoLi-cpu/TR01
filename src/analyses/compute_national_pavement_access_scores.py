"""Resumable exact all-road isolated medical-access restoration scores.

Uses only fixed physical entries, original population and the all-road 25%
reference. No traffic/condition imputation, priority tie rule or maintenance
selection is executed. Outputs are intermediate experiment files, not figures.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import time

import numpy as np

from pavement_inputs import ROOT, load_inputs
from national_pavement_impedance import NationalPavementImpedance
from road_service_loss import RoadServiceLoss

OUT = ROOT/'data/exp/national-pavement-access-scoring'
TOL = 1e-7


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def full_oracle(data, selected, delta):
    """Independent full reroute; every unselected source is impaired."""
    factors = np.full(len(data['lengths']), 1+delta)
    factors[selected] = 1.
    graph = data['graph'].copy()
    graph.data *= 1+delta
    for edge in selected:
        u, v = map(int, data['pairs'][edge])
        if u == v:
            continue
        for a, b in [(u, v), (v, u)]:
            left, right = graph.indptr[a:a+2]
            pos = int(left)+int(np.searchsorted(graph.indices[left:right], b))
            assert graph.indices[pos] == b
            graph.data[pos] = min(graph.data[pos], float(data['lengths'][edge]))
    ref = RoadServiceLoss(data['pairs'], np.asarray(data['lengths'])*factors,
                          data['origin_edges'], data['origin_positions']*factors[data['origin_edges']],
                          data['facility_edges'], data['facility_positions']*factors[data['facility_edges']],
                          graph=graph)
    return ref.distances()


def run():
    start = time.perf_counter()
    small_file = ROOT/'data/exp/national-pavement-policy-validation/small-network/validation.json'
    small = json.loads(small_file.read_text())
    assert small['status'] == 'passed' and small['all_roads_eligible']
    assert small['kernel_sha256'] == digest(ROOT/'src/analyses/national_pavement_impedance.py')
    paths = ['src/analyses/compute_national_pavement_access_scores.py',
             'src/analyses/national_pavement_impedance.py', 'src/analyses/pavement_impedance.py',
             'src/analyses/pavement_inputs.py', 'src/analyses/road_service_loss.py',
             'data/processed/national-network-rule-validation/edge_nodes_exact.npy',
             'data/processed/national-network-rule-validation/edge_lengths.npy',
             'data/processed/national-network-rule-validation/graph_exact.npz',
             'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet',
             'data/exp/national-population-entry/point_attachment_diagnostics.csv']
    config = {'scope': 'all_accepted_candidate_vehicle_source_roads', 'increment': .25,
              'population_and_physical_entries': 'fixed original baseline',
              'candidate_records': 21415800, 'selected_origins': 175612,
              'reference_metres_tolerance': TOL,
              'source_hashes': {p: digest(ROOT/p) for p in paths},
              'traffic_and_condition_inputs_used': False,
              'priority_order_or_maintenance_selection_generated': False}
    config_file = OUT/'configuration.json'
    if config_file.exists():
        assert json.loads(config_file.read_text()) == config
    else:
        atomic_json(config_file, config)
    print('Loading national routing inputs', flush=True)
    data = load_inputs()
    engine = NationalPavementImpedance(data['pairs'], data['lengths'], data['origin_edges'],
        data['origin_positions'], data['facility_edges'], data['facility_positions'],
        graph=data['graph'], cache=OUT/'array-index',
        progress=lambda msg: print(msg, flush=True))
    engine.configure(.25)
    reference = engine.distances()
    baseline = data['sample']['Candidate Medical Network Distance'].to_numpy()
    np.testing.assert_allclose(reference, baseline*1.25, rtol=1e-12, atol=TOL)
    np.save(OUT/'all_road_reference_0.25.npy', reference)
    validation_file = OUT/'national_kernel_validation.json'
    if validation_file.exists():
        validation = json.loads(validation_file.read_text())
        assert validation['source_configuration_sha256'] == digest(config_file)
        assert validation['status'] == 'passed'
    else:
        print('Checking isolated restoration and joint selections by independent full rerouting', flush=True)
        examples = [np.array([4363868], dtype=np.int64),
                    np.array([int(data['origin_edges'][0]), int(data['facility_edges'][0])], dtype=np.int64)]
        errors = []
        for selected in examples:
            if len(selected) == 1:
                actual = engine.restore_one(int(selected[0]))
            else:
                engine.configure(.25, selected)
                actual = engine.distances()
            expected = full_oracle(data, selected, .25)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=TOL)
            errors.append(float(np.max(np.abs(actual-expected))))
            engine.configure(.25)
            np.testing.assert_allclose(engine.distances(), reference, rtol=1e-12, atol=TOL)
        atomic_json(validation_file, {'status': 'passed', 'all_road_reference_reproduced': True,
                    'independent_full_reroute_checks': len(examples),
                    'maximum_cost_error_reference_metres': max(errors),
                    'source_configuration_sha256': digest(config_file),
                    'sample_and_joint_entry_restore_checked': True,
                    'small_network_validation': str(small_file.relative_to(ROOT))})
    file = OUT/'isolated_restoration_gain.npy'
    if not file.exists():
        initial = np.lib.format.open_memmap(file, mode='w+', dtype=np.float64,
                                           shape=(len(data['lengths']),))
        initial[:] = np.nan
        initial.flush()
        del initial
    gains = np.load(file, mmap_mode='r+')
    assert len(gains) == len(data['lengths'])
    already = int(np.isfinite(gains).sum())
    score_start = time.perf_counter()
    computed = 0
    print(f'Starting exact scoring: {already:,}/{len(gains):,} already stored', flush=True)
    for edge in range(len(gains)):
        if np.isfinite(gains[edge]):
            continue
        indices, values = engine.restore_one(edge, return_changes=True)
        improvement = reference[indices]-values
        assert (improvement >= -TOL).all()
        improvement[np.abs(improvement) <= TOL] = 0.
        gains[edge] = np.sum(data['population'][indices]*improvement)/1000
        computed += 1
        if computed % 100000 == 0 or edge == len(gains)-1:
            gains.flush()
            duration = time.perf_counter()-score_start
            done = already+computed
            rate = computed/duration
            report = {'stage': 'isolated_access_scoring', 'completed_roads': done,
                      'total_roads': len(gains), 'percent': done/len(gains)*100,
                      'roads_per_second_this_run': rate,
                      'remaining_seconds_at_current_rate': (len(gains)-done)/rate,
                      'elapsed_seconds_this_run': time.perf_counter()-start,
                      'maximum_resident_set_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            atomic_json(OUT/'progress.json', report)
            print(f'Access scores {done:,}/{len(gains):,} ({report["percent"]:.1f}%); remaining scoring {report["remaining_seconds_at_current_rate"]/60:.1f} min', flush=True)
    gains.flush()
    assert np.isfinite(gains).all() and (gains >= 0).all()
    atomic_json(OUT/'validation.json', {'status': 'passed_all_road_access_scores',
        'candidate_records': len(gains), 'positive_gain_records': int(np.count_nonzero(gains > 0)),
        'population': int(data['population'].sum()), 'selected_origins': len(reference),
        'elapsed_seconds_this_run': time.perf_counter()-start,
        'scoring_seconds_this_run': time.perf_counter()-score_start,
        'source_configuration_sha256': digest(config_file), 'traffic_and_condition_inputs_used': False,
        'priority_orders_generated': False, 'maintenance_sets_generated': False,
        'joint_policy_outcomes_generated': False, 'publication_outputs_generated': False})
    print('All-road isolated access scoring complete', flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT/'computation.pid'
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('National access scoring is already running')
    lock.write_text(str(os.getpid()))
    try:
        run()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
