"""Check incremental loss computation against independent and national oracles."""
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
sys.path[:0] = [str(ROOT/'src/analyses'), str(ROOT/'src/exp')]
from road_service_loss import RoadServiceLoss, loss_summary
from road_service_loss_dynamic import DynamicRoadServiceLoss
from validate_road_service_loss_kernel import split_point_oracle

OUT = ROOT/'data/exp/road-service-loss-validation/dynamic'
PROC = ROOT/'data/processed/national-network-rule-validation'


def small_checks():
    rng = np.random.default_rng(20261007)
    count = 0
    for trial in range(100):
        pairs = rng.integers(0, 8, size=(14, 2))
        lengths = rng.integers(1, 12, 14).astype(float)
        origins = [(int(e), float(rng.integers(0, int(lengths[e])+1))) for e in rng.integers(0, 14, 5)]
        facilities = [(int(e), float(rng.integers(0, int(lengths[e])+1))) for e in rng.integers(0, 14, 3)]
        ref = RoadServiceLoss(pairs, lengths, [o[0] for o in origins], [o[1] for o in origins],
                              [f[0] for f in facilities], [f[1] for f in facilities])
        fast = DynamicRoadServiceLoss(ref)
        critical = fast.critical_source_roads()
        for closed in [None, *range(14)]:
            expected = split_point_oracle(pairs, lengths, origins, facilities, closed)
            np.testing.assert_allclose(fast.distances(closed), expected, rtol=1e-12, atol=1e-12)
            if closed is not None:
                indices, values = fast.distances(closed, return_changes=True)
                reconstructed = fast.baseline.copy(); reconstructed[indices] = values
                np.testing.assert_allclose(reconstructed, expected, rtol=1e-12, atol=1e-12)
                if not critical[closed]:
                    unchanged = fast.baseline.copy()
                    unchanged[ref.origin_edges == closed] = np.inf
                    np.testing.assert_allclose(unchanged, expected, rtol=1e-12, atol=1e-12)
            count += 1
        np.testing.assert_array_equal(fast.distances(), ref.distances())
    return count


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT.parent/'national_benchmark_validation.json').open() as handle:
        expected_sources = json.load(handle)['source_hashes_verified_unchanged']
    for path, expected_hash in expected_sources.items():
        with (ROOT/path).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected_hash, path
    independent = small_checks()
    print(f'Independent randomized oracle: {independent} checks passed', flush=True)
    sample = pd.read_parquet(ROOT/'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet')
    points = pd.read_csv(ROOT/'data/exp/national-population-entry/point_attachment_diagnostics.csv',
                         dtype={'point_id': str, 'category': str, 'population_raw': str, 'road_source_id': str},
                         keep_default_na=False, low_memory=False, float_precision='round_trip')
    points['along_m'] = pd.to_numeric(points['along_m'], errors='coerce')
    origins = points.set_index('point_id').loc['mesh:'+sample['Mesh Code'].astype(str)]
    facilities = points[points['dataset'].eq('medical') & points['category'].isin(['1', '2']) & points['routing_eligible']]
    pairs = np.load(PROC/'edge_nodes_exact.npy', mmap_mode='r')
    lengths = np.load(PROC/'edge_lengths.npy', mmap_mode='r')
    ref = RoadServiceLoss(pairs, lengths, origins['edge_id'].to_numpy(), origins['along_m'].to_numpy(),
                          facilities['edge_id'].to_numpy(), facilities['along_m'].to_numpy(), graph=load_npz(PROC/'graph_exact.npz'))
    del points, facilities; gc.collect()
    cache = OUT/'source_removal_replacement_weights.npy'
    metadata = OUT/'replacement_cache_metadata.json'
    if cache.exists() and metadata.exists():
        previous = json.loads(metadata.read_text())
        assert previous['source_hashes'] == expected_sources
        with cache.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == previous['replacement_weights_sha256']
    fast = DynamicRoadServiceLoss(ref, replacements=np.load(cache, mmap_mode='r') if cache.exists() else None)
    np.testing.assert_allclose(fast.baseline, sample['Candidate Medical Network Distance'], rtol=1e-11, atol=1e-7)
    if not cache.exists(): np.save(cache, fast.replacements)
    graph_hash = hashlib.sha256(ref.graph.data).hexdigest()
    node_hash = hashlib.sha256(fast.node_distances).hexdigest()
    print(f'National setup {fast.setup_seconds:.2f} s; exact baseline reproduced', flush=True)
    critical = fast.critical_source_roads()
    np.save(OUT/'critical_source_road_screen.npy', critical)
    print(f'Exact witness screen: {int(critical.sum()):,} of {len(critical):,} roads need conditional recomputation', flush=True)
    cases = pd.read_csv(OUT.parent/'technical_scenario_results.csv')['Closed Source Edge ID'].astype(int).tolist()
    # Additional tree-edge removals exercise affected subtrees, rather than
    # relying on uniformly sampled roads that frequently change no chosen path.
    origin_endpoints = pairs[origins['edge_id'].to_numpy()]
    desired_children = set(origin_endpoints[np.linspace(0, len(origins)-1, 6, dtype=int), 0].tolist())
    for child in desired_children:
        parent = int(fast.parents[child])
        if parent < 0 or parent == ref.n: continue
        match = (((pairs[:,0] == parent) & (pairs[:,1] == child)) |
                 ((pairs[:,1] == parent) & (pairs[:,0] == child)))
        edges = np.flatnonzero(match)
        if len(edges): cases.append(int(edges[np.argmin(lengths[edges])]))
    population = sample['Local Total Population'].to_numpy()
    checks = []
    for edge in dict.fromkeys(cases):
        existing = OUT.parent/f'scenario_distances_{edge}.npy'
        if existing.exists():
            expected = np.load(existing); reference_seconds = None
        else:
            start = time.perf_counter(); expected = ref.distances(edge)
            reference_seconds = time.perf_counter()-start
        actual = fast.distances(edge)
        changed_indices, changed_values = fast.distances(edge, return_changes=True)
        reconstructed = fast.baseline.copy(); reconstructed[changed_indices] = changed_values
        np.testing.assert_array_equal(actual, reconstructed)
        if not critical[edge]:
            unchanged = fast.baseline.copy(); unchanged[ref.origin_edges == edge] = np.inf
            np.testing.assert_array_equal(actual, unchanged)
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-11, atol=1e-7)
        a = loss_summary(fast.baseline, actual, population)
        b = loss_summary(fast.baseline, expected, population)
        assert a['Lost Medical Path Population'] == b['Lost Medical Path Population']
        assert np.isclose(a['Additional Medical Distance Burden'], b['Additional Medical Distance Burden'], rtol=1e-9, atol=1e-5)
        checks.append({'Closed Source Edge ID': edge, **fast.last_diagnostics,
                       'Reference Runtime Seconds': reference_seconds,
                       'Maximum Finite Distance Error': float(np.max(np.abs(actual[finite]-expected[finite]), initial=0)),
                       **a})
        print(json.dumps(checks[-1]), flush=True)
    assert hashlib.sha256(ref.graph.data).hexdigest() == graph_hash
    assert hashlib.sha256(fast.node_distances).hexdigest() == node_hash
    pd.DataFrame(checks).to_csv(OUT/'national_validation_cases.csv', index=False)
    receipt = {'status': 'passed', 'independent_dense_oracle_checks': independent,
               'national_baseline_meshes': len(sample), 'national_validation_cases': len(checks),
               'exact_path_status_agreement': True, 'maximum_finite_error_metres': max(x['Maximum Finite Distance Error'] for x in checks),
               'setup_seconds': fast.setup_seconds, 'state_restored_after_scenarios': True,
               'dynamic_runtime_seconds': [x['Runtime Seconds'] for x in checks],
               'exact_witness_screen_recompute_roads': int(critical.sum()),
               'exact_witness_screen_source_roads': len(critical),
               'sparse_updates_reconstruct_full_distances': True,
               'role': 'Exact accelerator engineering validation; purposive timings do not estimate national runtime.',
               'national_dependence_complete': False}
    with cache.open('rb') as handle:
        receipt['replacement_weights_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    receipt['source_hashes_verified_unchanged'] = expected_sources
    metadata.write_text(json.dumps({'source_hashes':expected_sources,
                                    'replacement_weights_sha256':receipt['replacement_weights_sha256']}, indent=2)+'\n')
    (OUT/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
