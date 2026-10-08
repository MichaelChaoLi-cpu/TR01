"""Apply in-mesh road representatives as a separately labeled diagnostic scenario."""
import json
import numpy as np
import pandas as pd
from validate_national_continuity_rule import OUT as TOPO,PROC
from test_population_in_mesh_entry import OUT
from check_national_paths import main as paths


def main():
    p=pd.read_csv(TOPO/'point_attachment_diagnostics.csv',dtype={'point_id':str,'category':str,'population_raw':str,'road_source_id':str},keep_default_na=False,low_memory=False,float_precision='round_trip')
    candidates=pd.read_csv(OUT/'population_entry_candidates.csv',dtype={'point_id':str,'proxy_road_source_id':str,'source_tile':str},keep_default_na=False,float_precision='round_trip')
    offsets={str(int(t)):int(o) for t,o,n in np.load(PROC/'tile_offsets.npy')}
    for col in ['along_m','offset_m']:
        p[col]=pd.to_numeric(p[col],errors='coerce')
    p['baseline_edge_id']=p.edge_id;p['baseline_along_m']=p.along_m;p['baseline_offset_m']=p.offset_m
    p['baseline_road_source_id']=p.road_source_id;p['routing_eligible']=p.attached_500m
    p['entry_method']=np.where(p.attached_500m,'mesh_center_nearest_road_500m','pending_no_eligible_entry')
    p['entry_longitude']=np.nan;p['entry_latitude']=np.nan;p['entry_is_observed_residence']=False
    index={v:i for i,v in enumerate(p.point_id)};applied=0
    for r in candidates.itertuples():
        if r.result!='in_mesh_road_proxy_available':continue
        i=index[r.point_id];assert p.at[i,'dataset']=='population' and not p.at[i,'attached_500m']
        eid=offsets[r.source_tile]+int(r.proxy_tile_edge_id)
        p.at[i,'edge_id']=eid;p.at[i,'along_m']=float(r.proxy_along_m);p.at[i,'offset_m']=float(r.proxy_center_offset_m)
        p.at[i,'road_source_id']=r.proxy_road_source_id;p.at[i,'routing_eligible']=True;p.at[i,'entry_method']='in_mesh_eligible_road_proxy'
        p.at[i,'entry_longitude']=float(r.proxy_longitude);p.at[i,'entry_latitude']=float(r.proxy_latitude);applied+=1
    ec=np.load(PROC/'edge_components_exact.npy',mmap_mode='r');valid=p.edge_id.ge(0)
    p.loc[valid,'component_exact']=ec[p.loc[valid,'edge_id'].to_numpy(dtype=int)]
    for target in ['medical','ems_base']:
        selector=(p.dataset.eq('medical')&p.category.isin(['1','2']) if target=='medical' else p.dataset.eq('fire_stations')&p.category.isin(['2','3']))&p.routing_eligible
        comps=set(p.loc[selector,'component_exact'])
        p[target+'_component_reachable_exact']=p.routing_eligible&p.component_exact.isin(comps)
    # Threshold flags remain truthful; eligibility in this scenario is separate.
    assert p.attached_500m.sum()==pd.read_csv(TOPO/'point_attachment_diagnostics.csv',usecols=['attached_500m']).attached_500m.sum()
    p.to_csv(OUT/'point_attachment_diagnostics.csv',index=False)
    (OUT/'scenario_metadata.json').write_text(json.dumps(dict(proxies_applied=applied,attachment_threshold_flags_unchanged=True,routing_eligibility_column='routing_eligible',observed_residence=False),indent=2)+'\n')
    paths(proc=PROC,out=OUT,attachment_column='routing_eligible')


if __name__=='__main__':main()
