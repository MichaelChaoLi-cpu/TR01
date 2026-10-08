"""Exact reference computation for the adopted single-source-road loss scenario.

Fixed origin/facility entries on a removed source road become unusable. Parallel
source roads remain available. This is a conditional scenario, not a repair model.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.csgraph import dijkstra


def minimum_parallel_graph(pairs, lengths, node_count):
    pairs = np.asarray(pairs)
    keys = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)*node_count + np.maximum(pairs[:, 0], pairs[:, 1])
    order = np.argsort(keys)
    starts = np.r_[0, np.flatnonzero(np.diff(keys[order]))+1]
    unique = keys[order][starts]
    weights = np.minimum.reduceat(np.asarray(lengths)[order], starts)
    u, v = unique//node_count, unique % node_count
    keep = u != v
    return coo_matrix((np.r_[weights[keep], weights[keep]],
                       (np.r_[u[keep], v[keep]], np.r_[v[keep], u[keep]])),
                      shape=(node_count, node_count)).tocsr()


class RoadServiceLoss:
    def __init__(self, pairs, lengths, origin_edges, origin_positions,
                 facility_edges, facility_positions, graph=None):
        self.pairs, self.lengths = pairs, lengths
        self.origin_edges = np.asarray(origin_edges, dtype=np.int64)
        self.origin_positions = np.asarray(origin_positions, dtype=float)
        self.facility_edges = np.asarray(facility_edges, dtype=np.int64)
        self.facility_positions = np.clip(np.asarray(facility_positions, dtype=float), 0,
                                         lengths[self.facility_edges])
        self.n = int(np.max(pairs))+1 if graph is None else graph.shape[0]
        assert len(self.facility_edges) > 0
        assert np.isfinite(self.origin_positions).all()
        assert np.isfinite(self.facility_positions).all()
        assert (self.origin_edges >= 0).all() and (self.facility_edges >= 0).all()
        if graph is None:
            graph = minimum_parallel_graph(pairs, lengths, self.n)
        # Source-specific facility legs are retained before taking node minima.
        seed_nodes = np.r_[pairs[self.facility_edges, 0], pairs[self.facility_edges, 1]]
        self.seed_leg_costs = np.r_[self.facility_positions,
                                    lengths[self.facility_edges]-self.facility_positions]
        self.seed_leg_edges = np.tile(self.facility_edges, 2)
        self.seed_order = np.argsort(seed_nodes)
        ordered = seed_nodes[self.seed_order]
        self.seed_starts = np.r_[0, np.flatnonzero(np.diff(ordered))+1]
        self.seed_nodes = ordered[self.seed_starts]
        self.baseline_seed_costs = self.seed_costs(None)
        column = coo_matrix((self.baseline_seed_costs,
                             (self.seed_nodes, np.zeros(len(self.seed_nodes), dtype=int))),
                            shape=(self.n, 1)).tocsr()
        self.graph = bmat([[graph, column], [column.T, None]], format="csr")
        self.graph.sort_indices()
        self.seed_forward_positions = np.array([self.position(int(node), self.n)
                                                for node in self.seed_nodes])
        self.seed_reverse_positions = np.array([self.position(self.n, int(node))
                                                for node in self.seed_nodes])
        self.same_edge_facilities = {}
        for edge, position in zip(self.facility_edges, self.facility_positions):
            self.same_edge_facilities.setdefault(int(edge), []).append(float(position))
        self.same_edge_facilities = {key: np.sort(value) for key, value in self.same_edge_facilities.items()}

    def position(self, u, v):
        start, end = self.graph.indptr[u:u+2]
        index = int(start)+int(np.searchsorted(self.graph.indices[start:end], v))
        assert index < end and self.graph.indices[index] == v
        return index

    def seed_costs(self, closed_edge):
        costs = self.seed_leg_costs.copy()
        if closed_edge is not None:
            costs[self.seed_leg_edges == closed_edge] = np.inf
        return np.minimum.reduceat(costs[self.seed_order], self.seed_starts)

    def distances(self, closed_edge=None):
        changed = []
        if closed_edge is not None:
            assert 0 <= closed_edge < len(self.lengths)
            u, v = map(int, self.pairs[closed_edge])
            if u != v:
                # Recompute the pair weight from remaining source roads, rather
                # than accidentally removing parallel alternatives from the CSR.
                parallel = (((self.pairs[:, 0] == u) & (self.pairs[:, 1] == v)) |
                            ((self.pairs[:, 0] == v) & (self.pairs[:, 1] == u)))
                parallel[closed_edge] = False
                remaining = np.asarray(self.lengths)[parallel]
                weight = float(remaining.min()) if len(remaining) else np.inf
                for a, b in [(u, v), (v, u)]:
                    index = self.position(a, b)
                    changed.append((index, self.graph.data[index]))
                    self.graph.data[index] = weight
            seed_costs = self.seed_costs(closed_edge)
            self.graph.data[self.seed_forward_positions] = seed_costs
            self.graph.data[self.seed_reverse_positions] = seed_costs
        try:
            # Both directions are explicit; directed=True avoids constructing an
            # unnecessary transpose while retaining the accepted undirected metric.
            node_distances = dijkstra(self.graph, directed=True, indices=self.n)
            endpoints = self.pairs[self.origin_edges]
            result = np.minimum(node_distances[endpoints[:, 0]]+self.origin_positions,
                                node_distances[endpoints[:, 1]]+
                                self.lengths[self.origin_edges]-self.origin_positions)
            for i, (edge, position) in enumerate(zip(self.origin_edges, self.origin_positions)):
                edge = int(edge)
                if edge == closed_edge:
                    result[i] = np.inf
                elif edge in self.same_edge_facilities:
                    targets = self.same_edge_facilities[edge]
                    index = int(np.searchsorted(targets, position))
                    nearest = min(abs(position-targets[min(index, len(targets)-1)]),
                                  abs(position-targets[max(0, index-1)]))
                    result[i] = min(result[i], nearest)
            return result
        finally:
            for index, original in changed:
                self.graph.data[index] = original
            self.graph.data[self.seed_forward_positions] = self.baseline_seed_costs
            self.graph.data[self.seed_reverse_positions] = self.baseline_seed_costs


def loss_summary(baseline, scenario, population, tolerance=1e-7):
    baseline, scenario, population = map(np.asarray, (baseline, scenario, population))
    assert np.isfinite(baseline).all()
    assert (population >= 0).all()
    finite = np.isfinite(scenario)
    delta = scenario[finite]-baseline[finite]
    assert (delta >= -tolerance).all(), "Removing service cannot improve a fixed-entry shortest path"
    delta[np.abs(delta) <= tolerance] = 0
    return {"Lost Medical Path Population": int(population[~finite].sum()),
            "Additional Medical Distance Burden": float(np.sum(population[finite]*delta)/1000),
            "Disconnected Mesh Count": int((~finite).sum()),
            "Remaining Reachable Population": int(population[finite].sum()),
            "Mesh Count With Additional Distance": int(np.count_nonzero(delta > 0)),
            "Maximum Additional Distance": float(delta.max(initial=0)),
            "Numerical Distance Tolerance": tolerance}
