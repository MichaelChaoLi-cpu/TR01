"""Load the accepted pavement domain and unchanged national medical origins."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
NETWORK = ROOT/'data/processed/national-network-rule-validation'


def load_inputs():
    roads = pd.read_parquet(ROOT/'data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet')
    sample = pd.read_parquet(ROOT/'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet')
    groups = pd.read_parquet(ROOT/'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet')
    groups = groups.set_index('Mesh Code').loc[sample['Mesh Code']].reset_index()
    for col in ['Local Total Population','Candidate Medical Network Distance']:
        np.testing.assert_array_equal(sample[col],groups[col])
    points = pd.read_csv(ROOT/'data/exp/national-population-entry/point_attachment_diagnostics.csv',
                         usecols=['point_id','dataset','category','routing_eligible','edge_id','along_m'],
                         dtype={'point_id':str,'category':str},keep_default_na=False,
                         low_memory=False,float_precision='round_trip')
    points['along_m'] = pd.to_numeric(points['along_m'],errors='coerce')
    origins = points.set_index('point_id').loc['mesh:'+sample['Mesh Code'].astype(str)]
    facilities = points[points['dataset'].eq('medical') & points['category'].isin(['1','2']) & points['routing_eligible']]
    pairs = np.load(NETWORK/'edge_nodes_exact.npy',mmap_mode='r')
    lengths = np.load(NETWORK/'edge_lengths.npy',mmap_mode='r')
    eligible = roads['Graph Source Edge ID'].to_numpy(dtype=np.int64)
    assert len(roads)==70049 and len(sample)==175612
    assert int(sample['Local Total Population'].sum())==125874788
    np.testing.assert_array_equal(roads['Pavement Relative Maintenance Cost'],lengths[eligible])
    result={'roads':roads,'sample':sample,'groups':groups,'pairs':pairs,'lengths':lengths,
            'eligible':eligible,'graph':load_npz(NETWORK/'graph_exact.npz'),
            'origin_edges':origins['edge_id'].to_numpy(dtype=np.int64),
            'origin_positions':origins['along_m'].to_numpy(dtype=float),
            'facility_edges':facilities['edge_id'].to_numpy(dtype=np.int64),
            'facility_positions':facilities['along_m'].to_numpy(dtype=float),
            'population':sample['Local Total Population'].to_numpy(dtype=np.int64)}
    for prefix in ['origin','facility']:
        edges,positions=result[prefix+'_edges'],result[prefix+'_positions']
        assert np.isfinite(positions).all()
        assert (positions>=-1e-7).all() and (positions<=lengths[edges]+1e-7).all()
    return result


def make_engine(inputs):
    from pavement_impedance import PavementImpedance
    return PavementImpedance(inputs['pairs'],inputs['lengths'],inputs['origin_edges'],inputs['origin_positions'],
                             inputs['facility_edges'],inputs['facility_positions'],inputs['eligible'],graph=inputs['graph'])


def fixed_observed_orders(roads):
    ids=roads['Graph Source Edge ID'].to_numpy(dtype=np.int64)
    cost=roads['Pavement Relative Maintenance Cost'].to_numpy()
    traffic=roads['Observed Traffic Volume'].to_numpy()/cost
    grade=roads['Anchored Pavement Inspection Category'].to_numpy()
    return {'Traffic':np.lexsort((ids,-traffic)), 'Condition':np.lexsort((ids,cost,-grade))}


def prefix_indices(order,cost,share):
    cumulative=np.cumsum(cost[order],dtype=np.float64)
    requested=float(share*cost.sum())
    count=int(np.searchsorted(cumulative,requested,side='right'))
    return order[:count]
