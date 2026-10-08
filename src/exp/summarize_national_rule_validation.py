"""Validate preserved sources and compare the three national diagnostic scenarios."""
import csv,hashlib,json
import numpy as np
import pandas as pd
from validate_national_continuity_rule import OUT,PROC,BASE,QC,ROOT
from test_population_in_mesh_entry import OUT as ENTRY


def read(path):
    return pd.read_csv(path,dtype={'point_id':str,'category':str,'road_source_id':str,'population_raw':str},keep_default_na=False,low_memory=False,float_precision='round_trip')


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    oldpairs=np.load(BASE/'edge_nodes_exact.npy',mmap_mode='r');pairs=np.load(PROC/'edge_nodes_exact.npy',mmap_mode='r')
    remap=np.load(PROC/'node_remap_from_baseline.npy',mmap_mode='r')
    oldnodes=np.load(BASE/'nodes_exact.npy',mmap_mode='r');nodes=np.load(PROC/'nodes_exact.npy',mmap_mode='r')
    assert pairs.shape==oldpairs.shape
    for start in range(0,len(pairs),1000000):assert np.array_equal(pairs[start:start+1000000],remap[oldpairs[start:start+1000000]])
    for start in range(0,len(oldnodes),1000000):
        source=oldnodes[start:start+1000000];dest=nodes[remap[start:start+1000000]]
        assert np.array_equal(source['x'],dest['x']) and np.array_equal(source['y'],dest['y'])
    for name in ['edge_lengths.npy','edge_source_ordinals.npy','tile_offsets.npy','tiles']:assert (PROC/name).resolve()==(BASE/name).resolve()
    b=read(QC/'point_attachment_diagnostics.csv');t=read(OUT/'point_attachment_diagnostics.csv');e=read(ENTRY/'point_attachment_diagnostics.csv')
    assert len(b)==len(t)==len(e)==719791
    preserve=['point_id','dataset','category','longitude','latitude','population_raw','attached_500m']
    pd.testing.assert_frame_equal(b[preserve],t[preserve]);pd.testing.assert_frame_equal(b[preserve],e[preserve])
    anchors=['edge_id','road_source_id','along_m','offset_m']
    for col in anchors:
        # CSV round-trip normalization only; empty source numeric fields stay missing.
        if col in ['along_m','offset_m']:
            for frame in [b,t,e]:frame[col]=pd.to_numeric(frame[col],errors='coerce')
    pd.testing.assert_frame_equal(b[anchors],t[anchors])
    poi=b.dataset.ne('population');pd.testing.assert_frame_equal(b.loc[poi,anchors],e.loc[poi,anchors])
    retained=b.attached_500m;pd.testing.assert_frame_equal(b.loc[retained,anchors],e.loc[retained,anchors])
    changed=e.entry_method.eq('in_mesh_eligible_road_proxy');assert changed.sum()==1522 and (~b.loc[changed,'attached_500m']).all()
    assert e.loc[changed,'dataset'].eq('population').all()
    assert not e.entry_is_observed_residence.any()
    proxy=read(ENTRY/'population_entry_candidates.csv');proxy=proxy[proxy.result.eq('in_mesh_road_proxy_available')].set_index('point_id')
    pop=e[e.dataset.eq('population')];cells=pop.set_index('point_id').loc[proxy.index]
    x=proxy.proxy_longitude.to_numpy(float);y=proxy.proxy_latitude.to_numpy(float)
    assert (abs(x-cells.longitude.to_numpy(float))<=1/160+1e-12).all()
    assert (abs(y-cells.latitude.to_numpy(float))<=1/240+1e-12).all()
    lengths=np.load(PROC/'edge_lengths.npy',mmap_mode='r')
    edge=cells.edge_id.to_numpy(int);along=cells.along_m.to_numpy(float)
    assert (along>=-1e-6).all() and (along<=lengths[edge]+1e-6).all()
    offsets={str(int(tile)):int(start) for tile,start,n in np.load(BASE/'tile_offsets.npy')}
    for tile,group in proxy.groupby('source_tile'):
        z=np.load(BASE/'tiles'/f'{tile}.npz');idx=group.proxy_tile_edge_id.to_numpy(int)
        assert (z['levels'][idx]==0).all() and (z['states'][idx]==1).all() and (z['classes'][idx]!=4).all()
        actual=cells.loc[group.index,'edge_id'].to_numpy(int)
        assert np.array_equal(actual,offsets[str(tile)]+idx)
        assert list(cells.loc[group.index,'road_source_id'])==[f'{tile}:{fid}' for fid in z['fids'][idx]]
    routes=[]
    for folder,label in [(QC,'baseline'),(OUT,'continuity'),(ENTRY,'continuity_and_mesh_proxy')]:
        r=pd.read_csv(folder/'population_path_diagnostics.csv',dtype={'point_id':str})
        assert len(r)==353928 and not r.duplicated(['point_id','target']).any()
        routes.append(r.set_index(['point_id','target']).network_distance_m.rename(label))
    distances=pd.concat(routes,axis=1);assert distances.shape==(353928,3)
    assert distances.index.equals(routes[0].index)
    comparisons=[]
    for target in ['medical','ems_base']:
        d=distances.xs(target,level='target');a=d.baseline;c=d.continuity;v=d.continuity_and_mesh_proxy
        known=a.notna();assert c[known].notna().all();assert ((c[known]-a[known])<=1e-6).all()
        eligible_ids=set(b.loc[b.dataset.eq('population')&b.attached_500m,'point_id'])
        fixed=d.index.isin(eligible_ids)
        assert np.array_equal(c[fixed].isna(),v[fixed].isna())
        assert np.allclose(c[fixed].fillna(0),v[fixed].fillna(0),rtol=0,atol=1e-6)
        comparisons.append(dict(target=target,baseline_reachable=int(a.notna().sum()),continuity_reachable=int(c.notna().sum()),proxy_scenario_reachable=int(v.notna().sum()),newly_reachable_from_continuity=int((a.isna()&c.notna()).sum()),existing_routes_shortened=int((known&((a-c)>1e-6)).sum()),newly_reachable_from_mesh_proxy=int((c.isna()&v.notna()).sum()),existing_route_increases=0))
    pd.DataFrame(comparisons).to_csv(OUT/'scenario_comparison.csv',index=False)
    original=pd.read_csv(QC/'representative_route_checks.csv').rename(columns={'mesh_point_id':'point_id'})
    original=original[['path_id','point_id','target']].merge(distances.reset_index(),on=['point_id','target'],validate='one_to_one')
    assert len(original)==16;original.to_csv(OUT/'original_representative_routes_before_after.csv',index=False)
    for folder in [OUT,ENTRY]:
        pv=json.loads((folder/'path_validation.json').read_text());assert pv['status']=='passed' and pv['representative_routes']==16
    reviews=list(csv.DictReader((OUT/'independent_map_review_results.csv').open()))
    for r in reviews:assert hashlib.sha256((ROOT/r['evidence_card']).read_bytes()).hexdigest()==r['evidence_sha256']
    acquisition=json.loads((OUT/'map_acquisition.json').read_text());assert acquisition['failed_tiles']==0
    counts=pd.Series([r['status'] for r in reviews]).value_counts().to_dict()
    result=dict(status='passed',source_edges_checked=len(pairs),node_coordinates_preserved=True,source_lengths_and_ordinals_preserved=True,point_records_preserved=len(b),population_meshes=len(pop),raw_population_values_preserved=True,poi_and_original_qualifying_anchors_unchanged=True,original_500m_flags_unchanged=True,within_mesh_proxy_candidates=1522,pending_mesh_entries=200,all_proxy_source_edges_and_positions_valid=True,mesh_target_records_per_scenario=353928,route_monotonicity_passed=True,original_16_route_origins_compared=True,representative_geometries_checked_per_new_scenario=16,independent_visual_reviews=counts,independent_review_cards_hash_verified=True,scenario_comparison=comparisons)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join(f"| {r['target']} | {r['baseline_reachable']:,} | {r['continuity_reachable']:,} | {r['proxy_scenario_reachable']:,} | {r['existing_routes_shortened']:,} |" for r in comparisons)
    report=f'''# Nationwide structure continuity and mesh-entry validation

## Scope and decision status

This implements the approved diagnostic validation workflow (MILI-D-20261006-011), using the original nationwide N13 reference date of September 2024. The two variants are candidate engineering scenarios, not final accessibility variables. Source roads and population observations are preserved. No interior crossing, short-gap, speed, direction, legal-access, ferry, maintenance-condition or maintenance-allocation inference is added.

## Conservative topology rule

Join only an exact-coordinate pair with two incident endpoints globally, ground-adjacent levels {{0,1}} or {{-1,0}}, ordinary/bridge or ordinary/tunnel states, and matching road class. Keep prior unresolved/separated visual sites disconnected. Applied transitions: **{summary['applied_transitions']:,}**. Components: **{summary['baseline_components']:,} → {summary['candidate_graph']['components']:,}**. All **21,415,800** source edges and their **1,941,533.729 km** total projected length remain unchanged.

An independent deterministic screen selected 32 previously unseen sites, covering eight geographic bands and both bridge/tunnel transitions. Map and aerial cards supported 30 alignments; R004 and R028 remained unresolved and were withheld before the final rerun. The sample was frozen before these exceptions; the final rule is not an untouched holdout test. These judgments do not establish a national error rate. Mixed/unknown imagery dates cannot refute the trusted 2024 source solely through image absence.

## Within-mesh population-entry scenario

Among 1,722 centers failing the original 500 m attachment threshold, **1,522** have an eligible ordinary, non-expressway source-road point inside their own 1 km mesh. The search uses a 0.5 m interior guard and requires at least 1 m of clipped road. The closest point on these in-cell segments becomes a separately labeled road representative; **200** cells remain pending. This is not an observed residential location or evidence of a traversable off-road connector. No claim that a mesh lacks all real roads follows from an unsuccessful candidate search. Original mesh centers, raw population values and 500 m flags remain intact. All previously qualifying population anchors and all POI anchors are held fixed.

## Diagnostic comparison

Reachable counts refer to populated mesh records, not people. Medical targets use medical categories 1/2; EMS-base targets use fire station categories 2/3, under the user's station-as-ambulance-base assumption. Distances are bidirectional source-road lengths, excluding access connectors, with no travel speeds.

| Target | Baseline reachable meshes | Continuity reachable meshes | Continuity + mesh proxy | Previously finite routes shortened by continuity |
|---|---:|---:|---:|---:|
{table}

Connectivity gains alone do not confirm correct topology. The mesh-proxy scenario changes origin representation, so its distances cannot be interpreted as a pure topology effect. Population totals are not allocated to inferred residential points as empirical observations.

## Verification and reproducibility

`validation.json` verifies all source-edge mappings and preserved node coordinates, 719,791 original point records, proxy bounds and source attributes, original attachment flags, no distance increases or reachability loss for fixed anchors, and matching distances for unchanged anchors between the two candidate scenarios. Each new scenario has 353,928 mesh/target records and 16 reconstructed source-geometry routes. The original 16 diagnostic route origins are compared in `original_representative_routes_before_after.csv`.

Run locally from the repository root, in this order:

```sh
.venv/bin/python src/exp/validate_national_continuity_rule.py
.venv/bin/python src/exp/recheck_national_rule_paths.py
.venv/bin/python src/exp/test_population_in_mesh_entry.py
.venv/bin/python src/exp/evaluate_population_entry_scenario.py
.venv/bin/python src/exp/summarize_national_rule_validation.py
```

The frozen visual sample, review CSV, card hashes and tile manifests are retained here. `continuity_rule_audit.csv` records every screened transition and its disposition. Candidate graph arrays are in `data/processed/national-network-rule-validation/`; mesh-entry evidence and routing outputs are in `data/exp/national-population-entry/`. The baseline remains in `data/processed/national-network-quality/` and `data/exp/national-network-quality/`.

## Remaining interpretation limits

The graph is suitable for the next diagnostic review, not yet for claims about vehicle travel time or maintenance priorities. In-cell road representation needs a documented population-location assumption or finer settlement evidence before acceptance as a research variable. Legal vehicle access and unresolved entry cells still need attention. No AnaSOP section or final analysis output is changed by this provisional engineering validation.
'''
    (OUT/'README.md').write_text(report)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
