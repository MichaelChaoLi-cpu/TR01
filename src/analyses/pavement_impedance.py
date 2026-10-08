"""Exact finite pavement costs and isolated restoration on fixed source entries.

Physical locations remain fixed. Parallel roads use their minimum scenario cost.
Restoration scores use the all-unmaintained network; complete policies are routed
jointly. This module does not change the adopted sample or estimate repair effects.
"""
from __future__ import annotations

import heapq
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

from road_service_loss import RoadServiceLoss


class PavementImpedance:
    def __init__(self, pairs, lengths, origin_edges, origin_positions,
                 facility_edges, facility_positions, eligible, graph=None):
        self.ref = RoadServiceLoss(pairs, lengths, origin_edges, origin_positions,
                                   facility_edges, facility_positions, graph=graph)
        self.eligible = np.asarray(eligible, dtype=np.int64)
        assert len(np.unique(self.eligible)) == len(self.eligible)
        assert (np.asarray(lengths)[self.eligible] > 0).all()
        self.eligible_flag = np.zeros(len(lengths), dtype=bool)
        self.eligible_flag[self.eligible] = True
        self.selected_flag = np.zeros(len(lengths), dtype=bool)
        self.delta = 0.0
        ref = self.ref
        self.endpoints = pairs[ref.origin_edges]
        self.direct = np.full(len(ref.origin_edges), np.inf)
        for i, (edge, position) in enumerate(zip(ref.origin_edges, ref.origin_positions)):
            targets = ref.same_edge_facilities.get(int(edge))
            if targets is not None:
                k = int(np.searchsorted(targets, position))
                self.direct[i] = min(abs(position-targets[min(k, len(targets)-1)]),
                                     abs(position-targets[max(0, k-1)]))
        self.origin_node_index = self.grouped_index(self.endpoints.ravel())
        self.origin_edge_index = self.grouped_index(ref.origin_edges)
        self.facility_leg_index = self.grouped_index(ref.seed_leg_edges)
        # Index only endpoint pairs touched by an eligible road. Scan every
        # original road to retain noneligible parallel alternatives exactly.
        n = ref.n
        ep = pairs[self.eligible]
        ekeys = np.minimum(ep[:, 0], ep[:, 1]).astype(np.int64)*n + np.maximum(ep[:, 0], ep[:, 1])
        self.pair_keys, self.eligible_pair_group = np.unique(ekeys, return_inverse=True)
        self.outside_min = np.full(len(self.pair_keys), np.inf)
        for start in range(0, len(lengths) if len(self.pair_keys) else 0, 500000):
            end = min(start+500000, len(lengths))
            pp = pairs[start:end]
            keys = np.minimum(pp[:, 0], pp[:, 1]).astype(np.int64)*n + np.maximum(pp[:, 0], pp[:, 1])
            indices = np.searchsorted(self.pair_keys, keys)
            valid = indices < len(self.pair_keys)
            safe = np.minimum(indices, len(self.pair_keys)-1)
            valid &= self.pair_keys[safe] == keys
            valid &= ~self.eligible_flag[start:end]
            np.minimum.at(self.outside_min, indices[valid], np.asarray(lengths[start:end])[valid])
        self.pair_u, self.pair_v = self.pair_keys//n, self.pair_keys % n
        self.pair_forward = np.full(len(self.pair_keys), -1, dtype=np.int64)
        self.pair_reverse = np.full(len(self.pair_keys), -1, dtype=np.int64)
        for k, (u, v) in enumerate(zip(self.pair_u, self.pair_v)):
            if u != v:
                self.pair_forward[k] = ref.position(int(u), int(v))
                self.pair_reverse[k] = ref.position(int(v), int(u))
        self.edge_pair = dict(zip(map(int, self.eligible), map(int, self.eligible_pair_group)))
        self.node_distances = None
        self.baseline_origins = None
        self.last_diagnostics = {}

    @staticmethod
    def grouped_index(keys):
        order = np.argsort(keys, kind='stable')
        unique, starts, counts = np.unique(np.asarray(keys)[order], return_index=True, return_counts=True)
        return unique, starts, counts, order

    @staticmethod
    def group_members(index, key):
        unique, starts, counts, order = index
        k = np.searchsorted(unique, key)
        if k >= len(unique) or unique[k] != key:
            return np.empty(0, dtype=np.int64)
        return order[starts[k]:starts[k]+counts[k]]

    def factors(self, edges):
        return 1.0 + self.delta*(self.eligible_flag[edges] & ~self.selected_flag[edges])

    def configure(self, delta, selected=()):
        assert delta >= 0 and np.isfinite(delta)
        selected = np.asarray(selected, dtype=np.int64)
        assert self.eligible_flag[selected].all()
        self.delta = float(delta)
        self.selected_flag[:] = False
        self.selected_flag[selected] = True
        self.selected_count = len(selected)
        costs = np.asarray(self.ref.lengths)[self.eligible]*self.factors(self.eligible)
        pair_min = self.outside_min.copy()
        np.minimum.at(pair_min, self.eligible_pair_group, costs)
        valid = self.pair_forward >= 0
        self.ref.graph.data[self.pair_forward[valid]] = pair_min[valid]
        self.ref.graph.data[self.pair_reverse[valid]] = pair_min[valid]
        self.pair_min = pair_min
        self.seed_costs = self.ref.seed_leg_costs*self.factors(self.ref.seed_leg_edges)
        seed_min = np.minimum.reduceat(self.seed_costs[self.ref.seed_order], self.ref.seed_starts)
        self.ref.graph.data[self.ref.seed_forward_positions] = seed_min
        self.ref.graph.data[self.ref.seed_reverse_positions] = seed_min
        self.node_distances = None
        self.baseline_origins = None

    def origin_values(self, indices=None):
        take = slice(None) if indices is None else indices
        edges = self.ref.origin_edges[take]
        factors = self.factors(edges)
        positions = self.ref.origin_positions[take]
        endpoints = self.endpoints[take]
        result = np.minimum(self.node_distances[endpoints[:, 0]] + positions*factors,
                            self.node_distances[endpoints[:, 1]] +
                            (self.ref.lengths[edges]-positions)*factors)
        return np.minimum(result, self.direct[take]*factors)

    def distances(self):
        self.node_distances = dijkstra(self.ref.graph, directed=True, indices=self.ref.n)
        self.baseline_origins = self.origin_values()
        return self.baseline_origins.copy()

    def affected_origins(self, nodes, edge):
        unique, starts, counts, order = self.origin_node_index
        positions = np.searchsorted(unique, nodes)
        valid = positions < len(unique)
        safe = np.minimum(positions, len(unique)-1)
        valid &= unique[safe] == nodes
        chunks = [order[starts[k]:starts[k]+counts[k]]//2 for k in positions[valid]]
        on_edge = self.group_members(self.origin_edge_index, edge)
        if len(on_edge):
            chunks.append(on_edge)
        return np.unique(np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.int64)

    def restore_one(self, edge, return_changes=False):
        """Exact Dijkstra propagation after one source's costs decrease.

        Graph and node costs are restored after each call, so scores all use
        the same all-unmaintained reference rather than a greedy sequence.
        """
        assert self.selected_count == 0, 'Isolated scores require the all-unmaintained reference'
        assert self.node_distances is not None and self.baseline_origins is not None
        assert edge in self.edge_pair
        start_time = time.perf_counter()
        ref = self.ref
        q = self.edge_pair[edge]
        graph_changes, old_nodes = [], {}
        seeds = []
        u, v = map(int, ref.pairs[edge])
        new_weight = min(float(ref.lengths[edge]), float(self.pair_min[q]))
        if u != v and new_weight < self.pair_min[q]:
            for pos in [self.pair_forward[q], self.pair_reverse[q]]:
                graph_changes.append((pos, ref.graph.data[pos]))
                ref.graph.data[pos] = new_weight
            seeds.extend([(self.node_distances[u]+new_weight, v),
                          (self.node_distances[v]+new_weight, u)])
        legs = self.group_members(self.facility_leg_index, edge)
        old_seed_legs = self.seed_costs[legs].copy()
        self.seed_costs[legs] = ref.seed_leg_costs[legs]
        for node in set([u, v]):
            k = int(np.searchsorted(ref.seed_nodes, node))
            if k >= len(ref.seed_nodes) or ref.seed_nodes[k] != node:
                continue
            left = ref.seed_starts[k]
            right = ref.seed_starts[k+1] if k+1 < len(ref.seed_starts) else len(ref.seed_order)
            seed_min = float(self.seed_costs[ref.seed_order[left:right]].min())
            pos = ref.seed_forward_positions[k]
            if seed_min < ref.graph.data[pos]:
                for pos in [ref.seed_forward_positions[k], ref.seed_reverse_positions[k]]:
                    graph_changes.append((pos, ref.graph.data[pos]))
                    ref.graph.data[pos] = seed_min
                seeds.append((seed_min, node))
        self.selected_flag[edge] = True
        heap = []

        def improve(node, value):
            if value < self.node_distances[node]:
                if node not in old_nodes:
                    old_nodes[node] = self.node_distances[node]
                self.node_distances[node] = value
                heapq.heappush(heap, (float(value), int(node)))

        try:
            for value, node in seeds:
                improve(node, value)
            while heap:
                value, node = heapq.heappop(heap)
                if value != self.node_distances[node]:
                    continue
                left, right = ref.graph.indptr[node:node+2]
                for neighbour, weight in zip(ref.graph.indices[left:right], ref.graph.data[left:right]):
                    improve(int(neighbour), value+weight)
            affected = self.affected_origins(np.fromiter(old_nodes, dtype=np.int64), edge)
            values = self.origin_values(affected)
            self.last_diagnostics = {'changed_nodes':len(old_nodes), 'affected_origins':len(affected),
                                     'changed_matrix_entries':len(graph_changes),
                                     'seconds':time.perf_counter()-start_time}
            if return_changes:
                return affected, values
            result = self.baseline_origins.copy()
            result[affected] = values
            return result
        finally:
            for node, value in old_nodes.items():
                self.node_distances[node] = value
            for pos, value in graph_changes:
                ref.graph.data[pos] = value
            self.seed_costs[legs] = old_seed_legs
            self.selected_flag[edge] = False
