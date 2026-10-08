"""Array-indexed all-road extension of the validated finite-cost kernel.

Every accepted source road belongs to the impairment domain. CSR positions are
looked up in bounded vector batches, without millions of Python dictionary
entries. Restoration and joint routing reuse the original exact kernel.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

from pavement_impedance import PavementImpedance


class SourcePairLookup:
    def __init__(self, groups):
        self.groups = groups

    def __contains__(self, edge):
        return isinstance(edge, (int, np.integer)) and 0 <= edge < len(self.groups)

    def __getitem__(self, edge):
        if edge not in self:
            raise KeyError(edge)
        return int(self.groups[edge])


def csr_positions(graph, nodes, targets):
    """Vector binary search on sorted CSR rows, for one bounded batch."""
    left = graph.indptr[nodes].astype(np.int64, copy=True)
    right = graph.indptr[nodes+1].astype(np.int64, copy=True)
    while True:
        active = np.flatnonzero(left < right)
        if not len(active):
            break
        mid = (left[active]+right[active])//2
        smaller = graph.indices[mid] < targets[active]
        left[active[smaller]] = mid[smaller]+1
        right[active[~smaller]] = mid[~smaller]
    assert (left < graph.indptr[nodes+1]).all()
    np.testing.assert_array_equal(graph.indices[left], targets)
    return left


class NationalPavementImpedance(PavementImpedance):
    def __init__(self, pairs, lengths, origin_edges, origin_positions,
                 facility_edges, facility_positions, eligible=None, graph=None,
                 cache=None, progress=None):
        report = progress or (lambda _: None)
        if eligible is not None:
            np.testing.assert_array_equal(eligible, np.arange(len(lengths)))
        assert np.isfinite(lengths).all() and (np.asarray(lengths) > 0).all()
        # Empty-domain construction retains the validated physical entry and
        # supernode logic while avoiding the original large eligible-pair dict.
        super().__init__(pairs, lengths, origin_edges, origin_positions,
                         facility_edges, facility_positions,
                         np.empty(0, dtype=np.int64), graph=graph)
        self.eligible = np.arange(len(lengths), dtype=np.int64)
        self.eligible_flag[:] = True
        self.selected_flag[:] = False
        signature = {
            'road_count': len(lengths), 'node_count': self.ref.n,
            'pairs_sha256': hashlib.sha256(np.asarray(pairs).tobytes()).hexdigest(),
            'augmented_indptr_sha256': hashlib.sha256(self.ref.graph.indptr.tobytes()).hexdigest(),
            'augmented_indices_sha256': hashlib.sha256(self.ref.graph.indices.tobytes()).hexdigest(),
        }
        cache = Path(cache) if cache is not None else None
        manifest = cache/'manifest.json' if cache else None
        names = ['pair_keys', 'eligible_pair_group', 'pair_forward', 'pair_reverse']
        if manifest and manifest.exists():
            assert json.loads(manifest.read_text()) == signature
            report('Loading array-indexed source-pair cache')
            for name in names:
                setattr(self, name, np.load(cache/f'{name}.npy', mmap_mode='r'))
        else:
            report('Grouping every original road by endpoint pair')
            keys = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)*self.ref.n
            keys += np.maximum(pairs[:, 0], pairs[:, 1])
            self.pair_keys, groups = np.unique(keys, return_inverse=True)
            del keys
            assert len(self.pair_keys) < np.iinfo(np.int32).max
            self.eligible_pair_group = groups.astype(np.int32)
            del groups
            u = self.pair_keys//self.ref.n
            v = self.pair_keys % self.ref.n
            self.pair_forward = np.full(len(u), -1, dtype=np.int64)
            self.pair_reverse = np.full(len(u), -1, dtype=np.int64)
            report('Indexing forward and reverse CSR positions in bounded batches')
            for start in range(0, len(u), 500000):
                stop = min(start+500000, len(u))
                indices = np.arange(start, stop)[u[start:stop] != v[start:stop]]
                self.pair_forward[indices] = csr_positions(self.ref.graph, u[indices], v[indices])
                self.pair_reverse[indices] = csr_positions(self.ref.graph, v[indices], u[indices])
            del u, v
            if cache:
                cache.mkdir(parents=True, exist_ok=True)
                for name in names:
                    target = cache/f'{name}.npy'
                    temporary = cache/f'{name}.part'
                    with temporary.open('wb') as handle:
                        np.save(handle, getattr(self, name))
                    temporary.replace(target)
                temporary = cache/'manifest.part'
                temporary.write_text(json.dumps(signature, indent=2)+'\n')
                temporary.replace(manifest)
        assert len(self.eligible_pair_group) == len(lengths)
        self.pair_u = self.pair_keys//self.ref.n
        self.pair_v = self.pair_keys % self.ref.n
        # There are no roads outside the all-road eligibility domain.
        self.outside_min = np.full(len(self.pair_keys), np.inf)
        self.edge_pair = SourcePairLookup(self.eligible_pair_group)
        report(f'All-road array index ready: {len(lengths):,} records')
