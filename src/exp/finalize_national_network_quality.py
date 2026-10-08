"""Validate national screening artifacts, annotate evidence and write the QC report."""
import csv,hashlib,json,math
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from pyproj import Geod
from screen_national_network_tiles import ROOT,OUT,CACHE,csvout
from assemble_national_network_quality import PROC


def main():
    s=json.loads((OUT/'summary.json').read_text());tiles=json.loads((OUT/'tile_screen_summary.json').read_text())
    original=json.loads((ROOT/'data/exp/road-quality-check/summary.json').read_text())
    assert len(tiles)==160 and sum(r['source_features'] for r in tiles)==original['features']
    assert sum(r['vehicle_edges'] for r in tiles)==s['vehicle_edges']
    assert sum(r['source_features']-r['vehicle_edges'] for r in tiles)==sum(sum(r['excluded'].values()) for r in tiles)
    pairs=np.load(PROC/'edge_nodes_exact.npy',mmap_mode='r');nodes=np.load(PROC/'nodes_exact.npy',mmap_mode='r')
    offsets=np.load(PROC/'tile_offsets.npy');lengths=np.load(PROC/'edge_lengths.npy',mmap_mode='r')
    source=np.load(PROC/'edge_source_ordinals.npy',mmap_mode='r');strict_ec=np.load(PROC/'edge_components_exact.npy',mmap_mode='r');shared_ec=np.load(PROC/'edge_components_shared.npy',mmap_mode='r')
    attached=np.load(PROC/'point_attachment_arrays.npz');allpi=[]
    for tile,start,count in offsets:
        z=np.load(CACHE/f'{tile}.npz');end=z['ends'];pp=pairs[start:start+count]
        for j in [0,1]:
            assert np.array_equal(nodes['x'][pp[:,j]],end[:,j,0]) and np.array_equal(nodes['y'][pp[:,j]],end[:,j,1])
            assert np.array_equal(nodes['level'][pp[:,j]],z['levels'])
        assert np.array_equal(lengths[start:start+count],z['lengths'])
        pi=np.flatnonzero((attached['edge_id']>=start)&(attached['edge_id']<start+count));ii=attached['edge_id'][pi]-start
        assert np.all((z['levels'][ii]==0)&(z['states'][ii]==1)&(z['classes'][ii]!=4))
        assert np.all(attached['along_m'][pi]>=0) and np.all(attached['along_m'][pi]<=z['lengths'][ii])
        allpi.extend(pi.tolist())
    assert len(allpi)==int(np.sum(attached['edge_id']>=0)) and len(set(allpi))==len(allpi)
    graphchecks={}
    for variant in ['exact','shared']:
        g=load_npz(PROC/f'graph_{variant}.npz')
        assert np.all(g.data>0) and (g!=g.T).nnz==0
        assert g.shape[0]==s['exact_graph' if variant=='exact' else 'shared_coordinate_diagnostic_graph']['nodes']
        graphchecks[variant]=dict(nodes=g.shape[0],directed_arcs=g.nnz,positive_symmetric_weights=True)
        del g
    # Refresh diagnostic display labels from the current source-checked inventory.
    # P14_008 is the welfare facility name; P14_001 is the prefecture name.
    inventory=pd.read_csv(OUT/'points.csv',dtype=str,keep_default_na=False)
    points=pd.read_csv(OUT/'point_attachment_diagnostics.csv',dtype={'point_id':str,'category':str,'population_raw':str,'road_source_id':str},keep_default_na=False,low_memory=False,float_precision='round_trip')
    assert inventory.point_id.is_unique and points.point_id.is_unique and set(inventory.point_id)==set(points.point_id)
    names=inventory.set_index('point_id').name;points['name']=points.point_id.map(names)
    points.to_csv(OUT/'point_attachment_diagnostics.csv',index=False)
    points.loc[~points.attached_500m].to_csv(OUT/'point_attachment_anomalies.csv',index=False)
    for target in ['medical','ems_base']:
        assert (~points[target+'_component_reachable_exact']|points[target+'_component_reachable_shared']).all()
    gaps=pd.read_csv(OUT/'near_dangling_endpoint_candidates.csv',dtype={'tile':str,'source_ids':str},keep_default_na=False)
    starts={str(tile):int(start) for tile,start,count in offsets}
    cachefids={str(tile):np.load(CACHE/f'{tile}.npz')['fids'] for tile,start,count in offsets}
    exact_same=[];shared_same=[]
    for r in gaps.itertuples():
        ids=[]
        for sid in r.source_ids.split(';'):
            tile,fid=sid.split(':');i=np.searchsorted(cachefids[tile],int(fid));assert cachefids[tile][i]==int(fid);ids.append(starts[tile]+int(i))
        exact_same.append(bool(strict_ec[ids[0]]==strict_ec[ids[1]]));shared_same.append(bool(shared_ec[ids[0]]==shared_ec[ids[1]]))
    gaps['same_component_exact']=exact_same;gaps['same_component_shared']=shared_same
    gaps['triage']='local_gap_candidate';gaps.loc[~gaps.same_component_exact,'triage']='candidate_between_strict_components'
    gaps.to_csv(OUT/'near_dangling_endpoint_candidates.csv',index=False)
    grades=pd.read_csv(OUT/'multilevel_endpoint_candidates.csv',keep_default_na=False)
    grades['triage']=['two_edge_structure_transition' if int(r.incidence)==2 and '1' in str(r.road_states).split(';') and any(v in str(r.road_states).split(';') for v in ['2','3']) else 'multi_branch_or_other_transition' for r in grades.itertuples()]
    grades.to_csv(OUT/'multilevel_endpoint_candidates.csv',index=False)
    rows=list(csv.DictReader((OUT/'map_review_queue.csv').open()));judgments=json.loads((OUT/'visual_judgments.json').read_text())
    assert len(rows)==51 and {r['review_id'] for r in rows}==set(judgments)
    reviewed=[]
    for r in rows:
        a=judgments[r['review_id']];assert a['status'] in ['connect_supported','separation_supported','terrain_or_point_representation','unresolved'] and a['basis']
        path=OUT/'cards'/f"{r['review_id']}.png";assert path.is_file()
        reviewed.append(dict(r,**a,reviewer='Codex visual source review',evidence_card=str(path.relative_to(ROOT)),evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evidence_limit='Mixed/unknown imagery dates; topology support, not dated ground truth or legal vehicle access'))
    csvout(OUT/'map_review_results.csv',reviewed)
    csvout(OUT/'map_review_unresolved.csv',[r for r in reviewed if r['status']=='unresolved'])
    manifests=list(csv.DictReader((OUT/'tile_manifest.csv').open()))
    for r in manifests:
        assert r['status']!='failed' and not r['error'];path=OUT/'tile_cache'/r['filename'];data=path.read_bytes()
        assert len(data)==int(r['bytes']) and hashlib.sha256(data).hexdigest()==r['sha256']
    pathcheck=json.loads((OUT/'path_validation.json').read_text());assert pathcheck['status']=='passed' and pathcheck['representative_routes']==16
    geometries=json.loads((OUT/'representative_routes.geojson').read_text())['features'];geod=Geod(ellps='GRS80');maximum_scale_error=0.
    for f in geometries:
        c=np.array(f['geometry']['coordinates']);true=geod.line_length(c[:,0],c[:,1]);approx=f['properties']['length_m']
        if true>1:maximum_scale_error=max(maximum_scale_error,abs(approx/true-1))
    assert maximum_scale_error<.001
    provenance=json.loads((ROOT/'data/exp/data-acquisition/n13_2024_source_confirmation.json').read_text());assert provenance['status']=='passed' and provenance['current_file_hashes_match_acquisition']
    counts=dict(Counter(r['status'] for r in reviewed))
    result=dict(status='passed',source_feature_accounting=original['features'],vehicle_endpoint_and_length_checks=s['vehicle_edges'],point_source_inventory_checks=len(points),attachment_eligible_edge_checks=len(allpi),graph_checks=graphchecks,map_record_coverage=len(rows),map_tile_hash_checks=len(manifests),visual_judgment_counts=counts,representative_route_geodesic_scale_error_max=maximum_scale_error,local_gap_candidates_between_strict_components=int((~gaps.same_component_exact).sum()),grade_triage_counts=grades.triage.value_counts().to_dict(),welfare_display_name_source='P14_008; P14_001 is prefecture label; earlier pilot artifacts retained',raw_data_modified=False,candidate_repairs_applied=False)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    report=f'''# Nationwide road, spatial attachment and route quality screening

Nationwide automated screening and a targeted visual review are complete. This establishes a diagnostic baseline; it does **not** accept either diagnostic graph as the final vehicle network. Raw road, census and facility archives were preserved. No candidate gap, multilevel endpoint or interior crossing was automatically repaired.

## Coverage and definitions

- Official N13-2024 source: 160 archives and {original['features']:,} source lines. Reference date: September 2024. Inventory and file hashes match the official source acquisition. The official horizontal positional standard deviation is 25 m; coordinate precision and 1.5/5 m computational gap thresholds are not equivalent to physical positioning accuracy.
- Vehicle-centerline diagnostic view: {s['vehicle_edges']:,} ordinary centerlines with positive length. Three zero-length ordinary lines are excluded from this view; other centerline types remain in raw data. Physical state codes are not pavement deterioration measures.
- Point inventory: {len(points):,} source records, including 176,964 published 2020 population mesh centers and six nationwide POI datasets. Source IDs and CRS provenance are retained.
- Attachment edges: level 0, ordinary physical state, non-expressway. Search is capped at 5 km, with 100 m and 500 m diagnostic thresholds. Both adjacent and containing tiles are searched within a 0.1-degree tile halo; offsets use local AEQD projections. No off-road connector is established by a nearby line.

## Main automated results

| Graph diagnostic | Level-coded exact endpoints | Shared-coordinate endpoints |
|---|---:|---:|
| Nodes | {s['exact_graph']['nodes']:,} | {s['shared_coordinate_diagnostic_graph']['nodes']:,} |
| Connected components | {s['exact_graph']['components']:,} | {s['shared_coordinate_diagnostic_graph']['components']:,} |
| Medical component-reachable meshes | {s['component_reachability']['medical']['exact']:,} | {s['component_reachability']['medical']['shared']:,} |
| EMS component-reachable meshes | {s['component_reachability']['ems_base']['exact']:,} | {s['component_reachability']['ems_base']['shared']:,} |

Neither graph splits interior crossings. The strict graph can break legitimate bridge or tunnel continuity; the shared-coordinate graph can create inappropriate connections at some mixed-level sites. Their contrast measures sensitivity to endpoint treatment, not an observed road error rate or a validated accessibility range. Islands and legitimate road ends can produce real disconnected components.

| Dataset | Records | Within 100 m | Within 500 m | Outside 500 m |
|---|---:|---:|---:|---:|
'''
    for r in s['attachments']:report+=f"| {r['dataset']} | {r['points']:,} | {r['within_100m']:,} | {r['within_500m']:,} | {r['points']-r['within_500m']:,} |\n"
    report+=f'''
Local closest-other-endpoint screening identified {len(gaps):,} candidates within 5 m; {result['local_gap_candidates_between_strict_components']:,} link source edges in different strict components. These are candidates, not verified errors. This screen considers each dangling endpoint's nearest other endpoint within a source tile; it is not an exhaustive inventory of all possible short-gap pairs. The earlier exact tile-boundary screen separately identified 27 cross-tile pairs within 5 m, including 13 within 1.5 m. Its universe includes all source centerline types, so eligibility must be checked before repair.

The multilevel ledger contains {len(grades):,} within-tile vehicle endpoint sites. Triage distinguishes two-edge structure transitions from multi-branch or other cases. The global adjacent-level ledger has {s['global_multilevel_adjacent_level_records']:,} records; multiple level pairs can describe one geographic site. These ledgers and counts have different units.

Interior crossing checks examined a deterministic sample of {s['structure_lines_sampled_for_crossings']:,} bridge/tunnel source lines, producing {s['sampled_crossing_records']:,} candidate records. They are not an exhaustive scan of all national road-pair intersections, and duplicate structure or pair locations can occur.

## Map and route review

The visual sample contains 51 records across eight explicit geographic bands and bridge endpoints, tunnel endpoints, bridge/tunnel crossings, near dangling endpoints and large point offsets. Bands are coordinate-based audit strata, not official administrative regions: latitude at least 41; latitude 37 to below 41; remaining longitude at least 138; remaining longitude at least 135; remaining longitude at least 132 and latitude at least 34; remaining longitude at least 132; remaining latitude at least 30; and the remaining southern islands. Higher/lower published local population counts provide an audit proxy for settlement context, with a 1,000-person mesh threshold; this is not an official urban/rural definition. Missing published mesh records are not zero population. Three records supplement the original 48 to ensure higher/lower population contexts are represented in each geographic band.

| Visual judgment | Records |
|---|---:|
| Connection supported | {counts['connect_supported']} |
| Separation supported | {counts['separation_supported']} |
| Terrain or point representation explains a plausible large offset | {counts['terrain_or_point_representation']} |
| Unresolved | {counts['unresolved']} |

These are Codex visual observations, not new human-approved topology rules. N037 and N040 concern the same bridge-to-tunnel transition under different structure strata; record counts are not unique geographic-site counts. Available maps and photographs have mixed or unknown individual dates. An absent corridor in older imagery does not refute its presence in N13-2024. None of these observations triggered a national repair.

The strict graph produced 353,928 mesh-target diagnostic distance records. Sixteen high-detour routes, one per geographic band and target, were reconstructed from source centerlines and checked for partial-edge length and continuity. Route geometry checks passed, and graph reachability agrees with component membership. The largest local-projection discrepancy against geodesic length among reviewed route segments is {maximum_scale_error:.6%}. The floating-point boundary issue encountered during the first path run was corrected by exact float parsing and bounded, nonnegative partial-edge costs; the successful run passed explicit nonnegative-weight checks.

High detour ratios remain flags for topology review, not evidence of poor real-world service. The denominator is chord distance to the nearest eligible target point, which can be across water or disconnected roads. Numerical path continuity does not establish real vehicle access, permissible turns or source correctness.

## Interpretation limits and actionable findings

1. Review bridge/tunnel continuity rules first. The large component-count contrast shows that endpoint treatment matters nationally; selected supported two-edge transitions warrant a separately validated rule, while multi-branch cases need participating-link evidence.
2. Review large point offsets by context. Some population mesh centers fall in water, steep terrain or far from their residents' actual locations. Do not create a vehicle connector across water merely to attach a mesh center. Small-island facilities and mountain clinics can legitimately lie beyond the road eligibility threshold.
3. Review medical destination suitability. The large-offset list includes mountain clinics and facility names indicating military or industrial contexts. Name context is a screening clue, not verified public access; no facilities were removed by keywords. Facility public access and historical status still require a research decision before formal accessibility interpretation.
4. Treat national distances as diagnostic. All edges are bidirectional; speeds, one-way restrictions, turns, access gates, ferry links, service status and EMS dispatch delays are not modeled. The station/branch ambulance deployment assumption is user-specified. No maintenance costs, deterioration or budget scenario is estimated here.

The new nationwide welfare name display uses P14_008 (facility name). P14_001 is the prefecture name and had been used as a display label in the earlier pilot. Earlier pilot artifacts remain intact; this label issue does not affect coordinates, categories or routing.

## Deliverables

- `attachment_summary.csv`: dataset-level road attachment summary.
- `point_attachment_anomalies.csv`: every point outside the 500 m diagnostic threshold, including search-cap failures.
- `near_dangling_endpoint_candidates.csv`: local gap candidates with global component context.
- `multilevel_endpoint_candidates.csv` and `global_multilevel_endpoint_sites.csv`: source-traceable endpoint candidates and global adjacent-level coordinates.
- `sampled_interior_crossing_candidates.csv`: sampled structure-crossing candidates.
- `population_connectivity_candidates.csv`: inaccessible or variant-sensitive mesh candidates.
- `map_review_results.csv`, `map_review_unresolved.csv`, `cards/`, `sheets/`: map evidence and judgments.
- `population_path_diagnostics.csv`, `representative_route_checks.csv`, `representative_routes.geojson`, `representative_routes.png`: diagnostic routes and review material.
- `validation.json`: accounting, graph, attachment and evidence validation results.

## Reproduction

Run from the repository root, in this order:

```sh
.venv/bin/python src/exp/confirm_road_source.py
.venv/bin/python src/exp/prepare_national_quality_points.py
.venv/bin/python src/exp/screen_national_network_tiles.py
.venv/bin/python src/exp/assemble_national_network_quality.py
.venv/bin/python src/exp/check_national_paths.py
.venv/bin/python src/exp/plot_national_route_checks.py
.venv/bin/python src/exp/stratify_national_map_review.py
.venv/bin/python src/exp/fetch_national_quality_maps.py
.venv/bin/python src/exp/finalize_national_network_quality.py
```

The tile screening cache is resumable and assumes unchanged raw inputs and screening code. Delete the derived national tile cache before rebuilding after a rule or input change. `visual_judgments.json` stores manual judgments; changed sample IDs or evidence require renewed visual review. All required table operations use local scripts and libraries. No AnaSOP sections were changed by this engineering check.

Sources: [MLIT N13-2024](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html), [GSI tile catalogue](https://maps.gsi.go.jp/development/ichiran.html). The original national geometry report is in `data/exp/road-quality-check/README.md`; its zero-length, repeated-vertex and duplicate-geometry findings remain separate source diagnostics.
'''
    (OUT/'README.md').write_text(report)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
