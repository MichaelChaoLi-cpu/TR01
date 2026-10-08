"""Validate source-road loss against independently split-point dense networks."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
from road_service_loss import RoadServiceLoss, loss_summary


def split_point_oracle(pairs, lengths, origins, facilities, closed):
    """Independent all-pairs oracle: explicit origins/facilities on each road."""
    count = int(np.max(pairs))+1
    origin_nodes = list(range(count, count+len(origins)))
    facility_nodes = list(range(count+len(origins), count+len(origins)+len(facilities)))
    n = count+len(origins)+len(facilities)
    distances = np.full((n, n), np.inf)
    np.fill_diagonal(distances, 0)
    active_facilities = []
    for edge, ((u, v), length) in enumerate(zip(pairs, lengths)):
        if edge == closed:
            continue
        stops = [(0., int(u)), (float(length), int(v))]
        stops += [(position, origin_nodes[i]) for i, (road, position) in enumerate(origins) if road == edge]
        for i, (road, position) in enumerate(facilities):
            if road == edge:
                stops.append((position, facility_nodes[i]))
                active_facilities.append(facility_nodes[i])
        stops.sort()
        for (a, first), (b, second) in zip(stops, stops[1:]):
            distances[first, second] = distances[second, first] = min(distances[first, second], b-a)
    for intermediate in range(n):
        distances = np.minimum(distances, distances[:, intermediate, None]+distances[None, intermediate, :])
    return (distances[np.ix_(origin_nodes, active_facilities)].min(axis=1)
            if active_facilities else np.full(len(origins), np.inf))


def main():
    examples = [
        ("parallel replacement, detours and disconnected origins", [[0, 1], [0, 1], [1, 2], [0, 2], [2, 3]], [10., 18., 10., 35., 8.], [(2, 5.), (4, 4.), (0, 2.)], [(3, 35.)]),
        ("same-road direct path and source-specific facility removal", [[0, 1], [1, 2], [2, 0]], [10., 12., 20.], [(0, 4.), (1, 6.)], [(0, 4.), (0, 7.), (2, 10.)]),
        ("self-loop and co-located zero distance", [[0, 0], [0, 1]], [10., 4.], [(0, 2.), (1, 0.)], [(0, 2.), (1, 4.)]),
        ("all destinations become unavailable", [[0, 1], [1, 2]], [10., 10.], [(0, 5.), (1, 5.)], [(1, 5.)]),
    ]
    checks = []
    for name, pairs, lengths, origins, facilities in examples:
        pairs, lengths = np.asarray(pairs), np.asarray(lengths)
        kernel = RoadServiceLoss(pairs, lengths, [x[0] for x in origins], [x[1] for x in origins],
                                 [x[0] for x in facilities], [x[1] for x in facilities])
        baseline = kernel.distances()
        for closed in [None, *range(len(lengths))]:
            expected = split_point_oracle(pairs, lengths, origins, facilities, closed)
            actual = kernel.distances(closed)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            summary = loss_summary(baseline, actual, np.arange(1, len(origins)+1)*10)
            assert summary["Lost Medical Path Population"]+summary["Remaining Reachable Population"] == sum(np.arange(1, len(origins)+1)*10)
            checks.append({"case": name, "closed_source_edge": closed, "independent_dense_oracle": "passed"})
        np.testing.assert_array_equal(kernel.distances(), baseline)
    out = ROOT / "data/exp/road-service-loss-validation"
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "passed", "reference_cases": len(examples), "baseline_and_removal_checks": len(checks),
               "oracle": "Explicit point splitting and dense Floyd-Warshall; independent from the sparse seed computation",
               "checks": checks, "source_graph_restore_after_each_scenario": "passed"}
    (out / "kernel_validation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Passed: {len(checks)} independent baseline/removal checks, including parallel roads, zero distance, source entries, self-loops and loss of all destinations.")


if __name__ == "__main__":
    main()
