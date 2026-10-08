"""Independent dense-oracle checks for the array-indexed all-road extension."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src/analyses'), str(ROOT/'src/exp')]
from national_pavement_impedance import NationalPavementImpedance as PavementImpedance
from validate_road_service_loss_kernel import split_point_oracle

OUT = ROOT/'data/exp/national-pavement-policy-validation/small-network'


def oracle(pairs, lengths, origins, facilities, eligible, selected, delta):
    factors = np.ones(len(lengths))
    factors[eligible] += delta
    factors[selected] = 1
    return split_point_oracle(pairs, lengths*factors,
                             [(e, p*factors[e]) for e, p in origins],
                             [(e, p*factors[e]) for e, p in facilities], None)


def main():
    examples = [
        ([[0,1],[0,1]],[10.,12.],[(0,2.)],[(1,8.)],[0]),
        ([[0,1],[0,1]],[10.,12.],[(0,2.)],[(1,8.)],[0,1]),
        ([[0,0],[0,1]],[10.,4.],[(0,2.),(1,0.)],[(0,7.),(1,4.)],[0,1]),
        ([[0,1],[1,3],[0,2],[2,3]],[5.,5.,6.,6.],[(0,0.)],[(1,5.)],[0,1]),
        ([[0,1]],[10.],[(0,2.)],[(0,8.)],[]),
    ]
    rng = np.random.default_rng(20261007)
    for _ in range(100):
        pairs = rng.integers(0,10,(18,2))
        lengths = rng.uniform(.01,20,18)
        origins = [(int(e),float(rng.uniform(0,lengths[e]))) for e in rng.integers(0,18,8)]
        facilities = [(int(e),float(rng.uniform(0,lengths[e]))) for e in rng.integers(0,18,5)]
        eligible = np.flatnonzero(rng.random(18)<.5).tolist()
        examples.append((pairs,lengths,origins,facilities,eligible))
    full_checks = isolated_checks = 0
    max_error = 0.
    for pairs,lengths,origins,facilities,eligible in examples:
        pairs,lengths = np.asarray(pairs),np.asarray(lengths)
        eligible = np.arange(len(lengths)).tolist()
        engine = PavementImpedance(pairs,lengths,[e for e,_ in origins],[p for _,p in origins],
                                  [e for e,_ in facilities],[p for _,p in facilities],eligible)
        engine.configure(0)
        baseline = engine.distances()
        for delta in [.1,.25,.5]:
            previous = None
            for selected in [[],eligible[:len(eligible)//2],eligible]:
                engine.configure(delta,selected)
                actual = engine.distances()
                expected = oracle(pairs,lengths,origins,facilities,eligible,selected,delta)
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
                finite = np.isfinite(baseline)
                np.testing.assert_array_equal(np.isfinite(actual),finite)
                assert (actual[finite]>=baseline[finite]-1e-10).all()
                if previous is not None:
                    assert (actual[finite]<=previous[finite]+1e-10).all()
                previous = actual
                if finite.any():max_error = max(max_error,float(np.max(abs(actual[finite]-expected[finite]))))
                full_checks += 1
            np.testing.assert_allclose(previous,baseline,rtol=1e-12,atol=1e-10)
            engine.configure(delta)
            ref = engine.distances()
            graph_before = engine.ref.graph.data.copy()
            nodes_before = engine.node_distances.copy()
            seed_before = engine.seed_costs.copy()
            for e in eligible:
                actual = engine.restore_one(int(e))
                expected = oracle(pairs,lengths,origins,facilities,eligible,[e],delta)
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
                np.testing.assert_array_equal(engine.ref.graph.data,graph_before)
                np.testing.assert_array_equal(engine.node_distances,nodes_before)
                np.testing.assert_array_equal(engine.seed_costs,seed_before)
                finite = np.isfinite(ref)
                assert (actual[finite]<=ref[finite]+1e-10).all()
                if finite.any():max_error=max(max_error,float(np.max(abs(actual[finite]-expected[finite]))))
                isolated_checks += 1
    OUT.mkdir(parents=True,exist_ok=True)
    receipt={'status':'passed','networks':len(examples),'joint_dense_oracle_checks':full_checks,
             'isolated_restoration_dense_oracle_checks':isolated_checks,
             'maximum_cost_error':max_error,'parallel_source_alternatives_preserved':True,'all_roads_eligible':True,
             'partial_origin_destination_and_same_source_paths_scaled':True,
             'isolated_restoration_resets_all_state_exactly':True,
             'kernel_sha256':hashlib.sha256((ROOT/'src/analyses/national_pavement_impedance.py').read_bytes()).hexdigest(),
             'national_validation_completed':False,'paper_results_generated':False}
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
