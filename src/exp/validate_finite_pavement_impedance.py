"""Engineering check of finite simultaneous pavement costs, not paper results."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src/analyses'), str(ROOT / 'src/exp')]
from road_service_loss import RoadServiceLoss
from validate_road_service_loss_kernel import split_point_oracle

OUT = ROOT / 'data/exp/feasibility-check/pavement-scope-review/finite_impedance_validation'


def distances(pairs, physical_lengths, origins, facilities, eligible, selected, delta):
    factor = np.ones(len(physical_lengths))
    factor[eligible] += delta
    factor[selected] = 1
    costs = physical_lengths * factor
    # Store physical positions unchanged; pass their source-specific cost
    # coordinates to the existing independent finite-path reference.
    origin_cost_positions = [position * factor[edge] for edge, position in origins]
    facility_cost_positions = [position * factor[edge] for edge, position in facilities]
    reference = RoadServiceLoss(pairs, costs, [edge for edge, _ in origins], origin_cost_positions,
                                [edge for edge, _ in facilities], facility_cost_positions)
    actual = reference.distances()
    expected = split_point_oracle(pairs, costs,
                                 [(edge, value) for (edge, _), value in zip(origins, origin_cost_positions)],
                                 [(edge, value) for (edge, _), value in zip(facilities, facility_cost_positions)], None)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    return actual


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / 'src/analyses/road_service_loss.py', ROOT / 'src/exp/validate_road_service_loss_kernel.py', ROOT / 'docs/AnaSOP.md']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    rng = np.random.default_rng(20261007)
    examples = [
        ([[0, 1], [0, 1]], [10., 12.], [(0, 0.)], [(1, 12.)], [0]),
        ([[0, 1]], [10.], [(0, 2.)], [(0, 8.)], [0]),
        ([[0, 0], [0, 1]], [10., 4.], [(0, 2.), (1, 0.)], [(0, 2.), (1, 4.)], [0, 1]),
        ([[0, 1], [1, 3], [0, 2], [2, 3]], [5., 5., 6., 6.], [(0, 0.)], [(1, 5.)], [0, 1]),
    ]
    for _ in range(50):
        pairs = rng.integers(0, 8, size=(14, 2))
        lengths = rng.integers(1, 12, size=14).astype(float)
        origins = [(int(edge), float(rng.uniform(0, lengths[edge]))) for edge in rng.integers(0, 14, 5)]
        facilities = [(int(edge), float(rng.uniform(0, lengths[edge]))) for edge in rng.integers(0, 14, 3)]
        eligible = np.flatnonzero(rng.random(14) < .6).tolist()
        examples.append((pairs, lengths, origins, facilities, eligible))
    checks = 0
    for pairs, lengths, origins, facilities, eligible in examples:
        pairs, lengths = np.asarray(pairs), np.asarray(lengths)
        baseline = distances(pairs, lengths, origins, facilities, eligible, eligible, 0.)
        checks += 1
        for delta in [.1, .25, .5]:
            previous = None
            for selected in [[], eligible[:len(eligible)//2], eligible]:
                actual = distances(pairs, lengths, origins, facilities, eligible, selected, delta)
                finite = np.isfinite(baseline)
                np.testing.assert_array_equal(np.isfinite(actual), finite)
                assert (actual[finite] >= baseline[finite] - 1e-10).all()
                if previous is not None:
                    assert (actual[finite] <= previous[finite] + 1e-10).all()
                previous = actual
                checks += 1
            np.testing.assert_allclose(previous, baseline, rtol=1e-12, atol=1e-12)
    # A parallel road must remain available when only one source deteriorates.
    pairs, lengths, origins, facilities, eligible = examples[0]
    assert np.isclose(distances(np.asarray(pairs), np.asarray(lengths), origins, facilities, eligible, [], .25)[0], 12.)
    # Joint route switching makes separately calculated maintenance gains nonadditive.
    pairs, lengths, origins, facilities, eligible = examples[3]
    values = [float(distances(np.asarray(pairs), np.asarray(lengths), origins, facilities, eligible, selected, .25)[0])
              for selected in [[], [0], [1], [0, 1]]]
    np.testing.assert_allclose(values, [12., 11.25, 11.25, 10.])
    assert not np.isclose(values[0]-values[3], (values[0]-values[1]) + (values[0]-values[2]))
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    receipt = {'status': 'passed_engineering_prototype', 'authorization': 'MILI-D-20261007-042',
               'small_networks': len(examples), 'independent_dense_oracle_checks': checks,
               'deterioration_fractions': [.1, .25, .5], 'finite_reachability_unchanged': True,
               'all_maintained_reproduces_baseline': True, 'nested_restoration_monotone': True,
               'parallel_source_minima_preserved': True, 'origin_and_destination_partial_legs_scaled': True,
               'same_source_direct_paths_scaled': True, 'joint_nonadditivity_demonstrated': True,
               'illustrative_diamond_costs': values, 'national_validation_completed': False,
               'paper_results_generated': False, 'input_hashes_unchanged': hashes}
    (OUT / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'input_hashes_unchanged'}, indent=2))


if __name__ == '__main__':
    main()
