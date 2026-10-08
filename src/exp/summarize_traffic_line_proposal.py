"""Assemble a reviewable rule-based crosswalk proposal; no acceptance implied."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PROC=ROOT/'data/processed/national-network-rule-validation'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidates',required=True);args=parser.parse_args()
    out=ROOT/args.candidates;config=json.loads((out/'configuration.json').read_text())
    registry=pd.read_parquet(ROOT/config['registry_path'])
    inventory=pd.read_parquet((ROOT/config['registry_path']).parent/'observation_unit_inventory.parquet')
    frames=[]
    for path in sorted((out/'tiles').glob('*.npz')):
        receipt=json.loads(path.with_suffix('.json').read_text())
        with path.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==receipt['candidate_sha256']
        with np.load(path) as arrays:
            keep=arrays['Technical Proposal Qualified']
            if keep.any():frames.append(pd.DataFrame({key:arrays[key][keep] for key in arrays.files}))
    assert frames,'No proposed links are available; preserve the candidate receipts'
    qualified=pd.concat(frames,ignore_index=True)
    indices=qualified['Registry Group Index'].to_numpy()
    qualified['Traffic Observation Key']=registry['Traffic Observation Key'].to_numpy()[indices]
    qualified['Observed Traffic Volume']=registry['Observed Traffic Volume'].to_numpy()[indices]
    qualified=qualified.drop_duplicates(['Graph Source Edge ID','Traffic Observation Key'])
    counts=qualified.groupby('Graph Source Edge ID')['Traffic Observation Key'].nunique()
    single=counts[counts==1].index;multiple=counts[counts>1].index
    proposed=qualified[qualified['Graph Source Edge ID'].isin(single)].copy()
    ambiguous=qualified[qualified['Graph Source Edge ID'].isin(multiple)].copy()
    assert not proposed['Graph Source Edge ID'].duplicated().any()
    lengths=np.load(PROC/'edge_lengths.npy',mmap_mode='r');ordinals=np.load(PROC/'edge_source_ordinals.npy',mmap_mode='r')
    offsets=np.load(PROC/'tile_offsets.npy')
    for frame in [proposed,ambiguous]:
        edges=frame['Graph Source Edge ID'].to_numpy(dtype=np.int64)
        ti=np.searchsorted(offsets[:,1],edges,side='right')-1
        frame['Candidate Road Source Record']=[f'{int(offsets[i,0]):04d}:{int(ordinals[e])}' for i,e in zip(ti,edges)]
        frame['Baseline Source Road Length']=np.asarray(lengths[edges])
        frame['Result Role']='Unaccepted matching-rule proposal'
    proposed.to_parquet(out/'proposed_unique_crosswalk.parquet',index=False)
    ambiguous.to_parquet(out/'ambiguous_crosswalk.parquet',index=False)
    road_count=proposed.groupby('Traffic Observation Key').size()
    ambiguous_count=ambiguous.groupby('Traffic Observation Key')['Graph Source Edge ID'].nunique()
    inventory['Proposed Unique Source Roads']=inventory['Traffic Observation Key'].map(road_count).fillna(0).astype(int)
    inventory['Ambiguous Source Roads']=inventory['Traffic Observation Key'].map(ambiguous_count).fillna(0).astype(int)
    inventory.to_parquet(out/'proposal_observation_unit_coverage.parquet',index=False)
    inventory.to_csv(out/'proposal_observation_unit_coverage.csv',index=False)
    source=pd.read_parquet(ROOT/'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet')
    measured=source[source['Traffic Measurement Eligible']]
    unit_prefectures=measured.groupby('Traffic Observation Key')['Traffic Source Prefecture']
    assert unit_prefectures.nunique().eq(1).all(),'Shared observation unit crosses source prefectures'
    prefectures=unit_prefectures.first()
    table=registry.copy();table['Source Prefecture']=table['Traffic Observation Key'].map(prefectures)
    assert table['Source Prefecture'].notna().all()
    proposed['Source Prefecture']=table['Source Prefecture'].to_numpy()[proposed['Registry Group Index'].to_numpy()]
    coverage=proposed.groupby('Source Prefecture').agg(ProposedSourceRoads=('Graph Source Edge ID','size'),
                                              MatchedObservationUnits=('Traffic Observation Key','nunique'),
                                              SourceRoadLength=('Baseline Source Road Length','sum'))
    coverage=coverage.reindex([f'{p:02d}' for p in range(1,48)],fill_value=0)
    inventory['Source Prefecture']=inventory['Traffic Observation Key'].map(prefectures)
    total_units=inventory.groupby('Source Prefecture').size()
    complete_units=inventory.groupby('Source Prefecture')['Complete Observation Unit Geometry'].sum()
    coverage['MeasuredObservationUnits']=total_units.reindex(coverage.index,fill_value=0)
    coverage['CompleteGeometryObservationUnits']=complete_units.reindex(coverage.index,fill_value=0)
    coverage['UnmatchedObservationUnits']=coverage['MeasuredObservationUnits']-coverage['MatchedObservationUnits']
    assert coverage['MeasuredObservationUnits'].sum()==len(inventory)
    assert (coverage['UnmatchedObservationUnits']>=0).all()
    coverage.index.name='Source Prefecture'
    coverage.to_csv(out/'proposal_prefecture_coverage.csv')
    complete_receipt=json.loads((out/'validation.json').read_text())
    receipt={'status':'proposal_assembled','registry_mode':config['registry_mode'],
             'all_national_tiles':complete_receipt['all_national_tiles'],
             'qualified_source_roads':len(counts),'unique_proposed_source_roads':len(proposed),
             'ambiguous_source_roads':len(multiple),'proposed_observation_units':int(proposed['Traffic Observation Key'].nunique()),
             'all_observation_units_retained':len(inventory),
             'source_prefectures_retained':len(coverage),
             'source_graph_road_domain':len(lengths),'proposed_source_road_length_km':float(proposed['Baseline Source Road Length'].sum()/1000),
             'accepted_correspondences':0,'national_accuracy_estimate':None,
             'shared_measurement_limit':'Source road records share observation-unit counts; they are not independent traffic measurements.',
             'coverage_limit':'Unique rule-based proposals define candidate comparison coverage, not measured traffic on all graph roads.'}
    (out/'proposal_summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
