"""Exact incremental recomputation for one source-road removal.

An increased shortest-tree edge invalidates its descendant subtree. Distances
outside that subtree have an unchanged shortest-path witness. Boundary edges
seed Dijkstra within the invalidated subtree. The accepted loss scenario and
distance metric are identical to the independent full-recomputation reference.
"""
import heapq
import time
import numpy as np
from scipy.sparse.csgraph import dijkstra


def replacement_weights(pairs, lengths, node_count):
    """Remaining minimum pair weight for deletion of each original source road."""
    keys = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64) * node_count + np.maximum(pairs[:, 0], pairs[:, 1])
    order = np.lexsort((np.asarray(lengths), keys))
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    result = np.empty(len(lengths), dtype=float)
    result[order] = np.repeat(np.asarray(lengths)[order[starts]], counts)
    second = np.full(len(starts), np.inf)
    multiple = counts > 1
    second[multiple] = np.asarray(lengths)[order[starts[multiple] + 1]]
    result[order[starts]] = second
    return result


class DynamicRoadServiceLoss:
    def __init__(self, reference, replacements=None):
        self.ref = reference
        start = time.perf_counter()
        self.node_distances, self.parents = dijkstra(reference.graph, directed=True, indices=reference.n,
                                                     return_predecessors=True)
        assert self.parents[reference.n] == -9999
        valid = self.parents >= 0
        children = np.flatnonzero(valid).astype(np.int32)
        child_parents = self.parents[valid]
        self.children = children[np.argsort(child_parents, kind='stable')]
        counts = np.bincount(child_parents, minlength=reference.n + 1)
        self.child_offsets = np.r_[0, np.cumsum(counts)]
        self.replacements = (replacement_weights(reference.pairs, reference.lengths, reference.n)
                             if replacements is None else replacements)
        self.invalid = np.zeros(reference.n + 1, dtype=bool)
        self.direct = np.full(len(reference.origin_edges), np.inf)
        for i, (edge, position) in enumerate(zip(reference.origin_edges, reference.origin_positions)):
            targets = reference.same_edge_facilities.get(int(edge))
            if targets is not None:
                k = int(np.searchsorted(targets, position))
                self.direct[i] = min(abs(position - targets[min(k, len(targets)-1)]),
                                     abs(position - targets[max(0, k-1)]))
        self.endpoints = reference.pairs[reference.origin_edges]
        self.node_origin_index = self.grouped_index(self.endpoints.ravel())
        self.edge_origin_index = self.grouped_index(reference.origin_edges)
        self.baseline = self.origin_distances(None)
        self.setup_seconds = time.perf_counter() - start
        self.last_diagnostics = {}

    @staticmethod
    def grouped_index(keys):
        order = np.argsort(keys, kind='stable')
        unique, starts, counts = np.unique(np.asarray(keys)[order], return_index=True, return_counts=True)
        return unique, starts, counts, order

    @staticmethod
    def lookup_groups(index, keys):
        unique, starts, counts, order = index
        positions = np.searchsorted(unique, keys)
        positions = positions[positions < len(unique)]
        # The caller performs membership filtering before requesting slices.
        return [order[starts[k]:starts[k]+counts[k]] for k in positions]

    def affected_origins(self, nodes, closed):
        unique = self.node_origin_index[0]
        positions = np.searchsorted(unique, nodes)
        in_range = positions < len(unique)
        member = np.zeros(len(nodes), dtype=bool)
        member[in_range] = unique[positions[in_range]] == nodes[in_range]
        chunks = [part//2 for part in self.lookup_groups(self.node_origin_index, nodes[member])]
        unique = self.edge_origin_index[0]
        k = np.searchsorted(unique, closed)
        if k < len(unique) and unique[k] == closed:
            chunks.extend(self.lookup_groups(self.edge_origin_index, np.array([closed])))
        return np.unique(np.concatenate(chunks)) if chunks else np.empty(0, dtype=np.int64)

    def origin_distances(self, closed, indices=None):
        ref = self.ref
        take = slice(None) if indices is None else indices
        endpoints = self.endpoints[take]
        edge = ref.origin_edges[take]
        position = ref.origin_positions[take]
        result = np.minimum(self.node_distances[endpoints[:, 0]] + position,
                            self.node_distances[endpoints[:, 1]] + ref.lengths[edge] - position)
        np.minimum(result, self.direct[take], out=result)
        if closed is not None: result[edge == closed] = np.inf
        return result

    def distances(self, closed=None, return_changes=False):
        if closed is None: return self.baseline.copy()
        ref = self.ref
        assert 0 <= closed < len(ref.lengths)
        start = time.perf_counter()
        roots, changes, invalid_nodes = [], [], []
        u, v = map(int, ref.pairs[closed])
        if u != v:
            new_weight = float(self.replacements[closed])
            for a, b in [(u, v), (v, u)]:
                position = ref.position(a, b)
                old_weight = ref.graph.data[position]
                assert new_weight >= old_weight
                if new_weight > old_weight:
                    changes.append((position, old_weight)); ref.graph.data[position] = new_weight
                    if self.parents[b] == a: roots.append(b)
        # Only the closed source's endpoint seed groups can change.
        for node in set([u, v]):
            k = int(np.searchsorted(ref.seed_nodes, node))
            if k >= len(ref.seed_nodes) or ref.seed_nodes[k] != node: continue
            left = ref.seed_starts[k]
            right = ref.seed_starts[k+1] if k+1 < len(ref.seed_starts) else len(ref.seed_order)
            legs = ref.seed_order[left:right]
            retained = legs[ref.seed_leg_edges[legs] != closed]
            new_weight = float(ref.seed_leg_costs[retained].min()) if len(retained) else np.inf
            if new_weight > ref.baseline_seed_costs[k]:
                for position in [ref.seed_forward_positions[k], ref.seed_reverse_positions[k]]:
                    changes.append((position, ref.graph.data[position])); ref.graph.data[position] = new_weight
                if self.parents[node] == ref.n: roots.append(node)
        stack = list(set(roots))
        while stack:
            node = stack.pop()
            if self.invalid[node]: continue
            self.invalid[node] = True; invalid_nodes.append(node)
            stack.extend(self.children[self.child_offsets[node]:self.child_offsets[node+1]].tolist())
        nodes = np.asarray(invalid_nodes, dtype=np.int32)
        old_distances = self.node_distances[nodes].copy()
        try:
            self.node_distances[nodes] = np.inf
            heap = []
            # Seed every invalid node from paths wholly outside the invalidated
            # tree. Unreachable outside nodes and deleted edges cannot seed it.
            for node in nodes:
                left, right = ref.graph.indptr[node:node+2]
                neighbours = ref.graph.indices[left:right]
                weights = ref.graph.data[left:right]
                boundary = ~self.invalid[neighbours]
                value = float((self.node_distances[neighbours[boundary]] + weights[boundary]).min(initial=np.inf))
                if np.isfinite(value):
                    self.node_distances[node] = value; heapq.heappush(heap, (value, int(node)))
            while heap:
                value, node = heapq.heappop(heap)
                if value != self.node_distances[node]: continue
                left, right = ref.graph.indptr[node:node+2]
                for neighbour, weight in zip(ref.graph.indices[left:right], ref.graph.data[left:right]):
                    if not self.invalid[neighbour]: continue
                    candidate = value + weight
                    if candidate < self.node_distances[neighbour]:
                        self.node_distances[neighbour] = candidate
                        heapq.heappush(heap, (candidate, int(neighbour)))
            affected = self.affected_origins(nodes, closed)
            values = self.origin_distances(closed, affected)
            self.last_diagnostics = {'Invalidated Nodes': len(nodes), 'Changed Matrix Entries': len(changes),
                                     'Affected Origin Candidates': len(affected),
                                     'Runtime Seconds': time.perf_counter() - start}
            if return_changes: return affected, values
            result = self.baseline.copy()
            result[affected] = values
            return result
        finally:
            self.node_distances[nodes] = old_distances
            self.invalid[nodes] = False
            for position, old in changes: ref.graph.data[position] = old

    def critical_source_roads(self):
        """Conservative exact screen using unchanged baseline path witnesses.

        False means every non-closed origin retains a baseline shortest path.
        Closed-origin entry losses are accounted separately. True is a candidate
        for recomputation, not a prediction of positive loss.
        """
        ref = self.ref
        first = self.node_distances[self.endpoints[:,0]] + ref.origin_positions
        second = self.node_distances[self.endpoints[:,1]] + ref.lengths[ref.origin_edges] - ref.origin_positions
        via = np.minimum(first, second)
        chosen = self.endpoints[np.arange(len(first)), np.where(first <= second, 0, 1)]
        chosen = chosen[np.isfinite(via) & (via < self.direct)]
        used_child = np.zeros(ref.n+1, dtype=bool)
        for node in chosen:
            while node != ref.n and node >= 0 and not used_child[node]:
                used_child[node] = True
                node = self.parents[node]
        critical = np.zeros(len(ref.lengths), dtype=bool)
        for start in range(0, len(critical), 500000):
            end = min(start+500000, len(critical))
            u, v = ref.pairs[start:end].T
            selected_tree_pair = ((used_child[u] & (self.parents[u] == v)) |
                                  (used_child[v] & (self.parents[v] == u)))
            critical[start:end] = selected_tree_pair & (self.replacements[start:end] > ref.lengths[start:end]) & (u != v)
        # A used supernode leg can depend on a source-specific facility even
        # when its ordinary pair is not on the selected path witness.
        for edge in np.unique(ref.facility_edges):
            for node in set(map(int, ref.pairs[edge])):
                if not used_child[node] or self.parents[node] != ref.n: continue
                k = int(np.searchsorted(ref.seed_nodes, node))
                assert k < len(ref.seed_nodes) and ref.seed_nodes[k] == node
                left = ref.seed_starts[k]
                right = ref.seed_starts[k+1] if k+1 < len(ref.seed_starts) else len(ref.seed_order)
                legs = ref.seed_order[left:right]
                retained = legs[ref.seed_leg_edges[legs] != edge]
                remaining = ref.seed_leg_costs[retained].min() if len(retained) else np.inf
                if remaining > ref.baseline_seed_costs[k]: critical[edge] = True
        return critical
