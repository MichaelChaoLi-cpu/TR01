"""Render the engineering pilot report from reproducible saved outputs."""
import json
from collections import Counter
import pandas as pd
from build_vehicle_network_pilot import OUT,PROC
s=json.loads((OUT/'summary.json').read_text());v=json.loads((OUT/'validation.json').read_text())
a=pd.read_csv(PROC/'point_attachments.csv',dtype={'category':str,'mesh_code':str})
a[(a.dataset=='population')&~a.attached_500m].to_csv(OUT/'unattached_population_meshes.csv',index=False)
r=pd.read_csv(OUT/'route_summary.csv');grades=pd.read_csv(OUT/'grade_transition_candidates.csv')
actions=pd.read_csv(OUT/'geometry_actions.csv')
types=pd.read_csv(OUT/'road_type_counts.csv')
poi_count=int((a.dataset!='population').sum());meshes=int((a.dataset=='population').sum())
ems=int(((a.dataset=='fire_stations')&a.category.isin(['2','3'])).sum())
med=int(((a.dataset=='medical')&a.category.isin(['1','2'])).sum())
variant_table='\n'.join(f"| {q['variant']} | {q['nodes']:,} | {q['components']:,} | {q['largest_component_length_share']:.2%} |" for q in s['network_variants'])
attachment_table='\n'.join(f"| {q['dataset']} | {q['points']:,} | {q['within_100m']:,} | {q['within_500m']:,} | {q['median_offset_m']:.1f} | {q['p95_offset_m']:.1f} |" for q in s['attachment_summary'])
route_table='\n'.join(f"| {q.variant} | {q.target} | {int(q.reachable_meshes)} / {int(q.attached_500m)} |" for q in r.itertuples())
text=f'''# Vehicle network pilot: Shimabara Peninsula and surrounding coasts

## Assessment

The local pipeline runs end to end: candidate road extraction, projected distance measurement, endpoint graph construction, population/POI attachment and shortest-distance diagnostics. It is not yet a validated national vehicle network. Layer transitions and interior crossings require review before a national connection rule can be adopted.

This is an engineering pilot, not a final research sample, population estimate, accessibility result or maintenance analysis. No final variable definition or AnaSOP section was changed.

## Scope

- Bounding box in JGD2011 longitude/latitude: {s['bbox']}.
- Road tiles: 4830 and 4930. The box spans the shared latitude boundary, peninsula roads, settlements and surrounding coasts.
- Working projection: EPSG:6670, JGD2011 / Japan Plane Rectangular CS II, in metres.
- Candidate vehicle edges: **{s['stats']['candidate_edges']:,}**.
- Population mesh centres: **{meshes:,}**; census 2020 total-population field retained as published.
- Facility points: **{poi_count:,}**, across all six previously selected categories.

## Provisional rules and geometry treatment

- Retain ordinary centerlines (type 1) as candidate vehicle roads. Exclude garden paths, footpaths and stairs: {s['stats']['excluded_non_ordinary_centerline']:,} intersecting source features. This does not establish vehicle legality or adequate road width for every retained road.
- Exclude construction/other/unknown physical road states when present. Do not treat physical road state as deterioration or maintenance need.
- Clip lines to the pilot box. Point-only boundary touches provide no in-box road segment and are omitted; multipart intersections become separate edges. Box clipping introduces artificial ends and prevents routes outside the box.
- Remove consecutive repeated coordinates in the derived pilot only: {len(actions):,} source features affected. No zero-length or exact duplicate removal was triggered inside this pilot; nationwide anomalies remain unchanged elsewhere. Exact duplicate removal, if needed, requires both identical/reversed coordinates and identical properties.
- Merge **{s['stats']['submillimetre_seam_pairs_merged']}** cross-tile, same-level endpoint pairs separated by less than 1 mm on the shared tile boundary. This is a provisional numerical-precision treatment; the larger 1.5 m/5 m candidates were not automatically snapped.
- Compare exact vs precision-treated graphs, each with strict level separation vs shared-coordinate endpoint joining. No rule is selected as the final national model.
- Do not split interior crossings automatically. All retained edges are treated as bidirectional for this diagnostic; one-way travel, turn restrictions, vehicle limitations, ferries and travel speeds are not supplied by the source or modeled here.
- Facility and population attachment candidates are level-0, ordinary-state, non-expressway edges. Report both 100 m and 500 m attachment screens. An offset threshold does not establish a traversable connector: rivers, walls, hills and entrance locations are unverified.

## Connectivity diagnostics

| Graph variant | Nodes | Components | Largest component share of line length |
| --- | ---: | ---: | ---: |
{variant_table}

The different land masses and the clipped study boundary explain some disconnections. Component counts are not counts of data errors. Strict level separation can disconnect a valid bridge or ramp transition; shared-coordinate merging can create an invalid connection. Both need interpretation.

Endpoint review ledger: **{v['grade_transition_sites']}** multi-level sites. Candidate classes: {json.dumps(grades.candidate_class.value_counts().to_dict())}. These labels prioritize review and do not confirm topology.

Interior intersection screen:

- Same-level interior crossings: **{s['intersection_candidates'].get('interior_crossing_same_level',0)}**.
- Different-level interior crossings: **{s['intersection_candidates'].get('interior_crossing_different_level',0)}**.
- Other non-endpoint touches/overlaps: **{s['intersection_candidates'].get('non_endpoint_touch_or_overlap_same_level',0)}**.

Same-level crossings are not automatically at-grade junctions. Examples are capped at 200; aggregate counts cover the full pilot. The endpoint-level ledger is exhaustive within this pilot.

![Component comparison](network_components.png)

Blue represents the largest component; orange represents all other components. The maps are engineering diagnostics, not manuscript figures.

## Spatial attachment

| Source | Points | Within 100 m | Within 500 m | Median offset (m) | 95th percentile offset (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
{attachment_table}

All {poi_count:,} facility points pass the provisional 500 m screen. **577 of {meshes}** population centres pass; **15** exceed it and are saved in `unattached_population_meshes.csv`. A mesh centre is a proxy location, not the observed location of residents. These are attachment diagnostics, not population-weighted service coverage.

POI coordinates were transformed from their declared source CRS to JGD2011 and then projected. Missing fire `.prj` declarations were resolved using the official JGD2000 specification. Coordinate operations and their reported accuracy are saved separately; the JGD2000-to-JGD2011 operation reports 1 m accuracy, while projection accuracy may be unreported. This treatment does not establish sub-metre source positional accuracy.

Source attributes and census suppression codes were retained. No elderly share was constructed: suppressed age-specific counts remain incompatible with direct local-total division until aggregation is resolved.

## Distance-path diagnostics

For a reproducibility check only, use {med} hospital/clinic candidate points and {ems} fire station/branch candidate bases. Dental clinics and fire headquarters remain in attachment outputs but are not origins/destinations for these two path checks. Historical suspension/service availability was not filtered, so these candidate points do not establish current service operation.

The user's station/branch ambulance-deployment assumption is retained. Vehicle availability, dispatch delays and response times are not observed.

Points enter at the projected position along a line, with partial-edge distances to each endpoint. Same-edge direct paths are also considered. The graph retains the shortest weight for parallel endpoint pairs. Distances exclude the straight off-road attachment offset; they are provisional network lengths, not door-to-door travel costs.

| Graph variant | Candidate destination/origin | Meshes with a finite path / attached meshes |
| --- | --- | ---: |
{route_table}

Shared-endpoint joining gives 13 additional finite-path meshes for each of the two destination/origin checks, compared with strict level separation. This difference concerns the 577 attached mesh centres, not national access inequality. In total, {v['variant_sensitive_mesh_target_rows']} mesh-target rows change in reachability or path length; they are retained for review. The three precision seam merges do not alter finite-path mesh counts in this pilot.

## Validation and reproducibility

Validation: **{v['status']}**.

- All {v['source_hashes_matching_acquisition']} source hashes match the acquisition manifests; raw archives remain unchanged.
- Edge/node/attachment counts and attachment positions pass consistency checks.
- Known-route checks verify partial-edge distances, layer-separated endpoints, shared-endpoint distances, no automatic interior-crossing node, and minimum parallel-edge weights.
- Adding shared-endpoint connections does not increase any common finite shortest distance.

```sh
.venv/bin/python src/exp/build_vehicle_network_pilot.py
.venv/bin/python src/exp/validate_vehicle_network_pilot.py
.venv/bin/python src/exp/report_vehicle_network_pilot.py
```

Experimental graph/attachment files are in `data/processed/vehicle-network-pilot/`; diagnostics are in this directory. These files are provisional engineering artifacts, not final analysis-ready variables. Exact source IDs identify tile and 1-based feature position. The stored layer-aware graph is one comparison artifact, not an endorsed national topology.

Dependencies added with approval: shapely, pyproj and scipy. Python remains 3.12.13. No Git or DVC operation was performed. Framelog: MILI-D-20261006-008 records the accepted pilot-first workflow; agent-chosen area, cutoffs and connection variants are not attributed to the human.

## Next evidence needed

Review the 151 multi-level shared endpoints and prioritize the 204 same-level interior crossing candidates using official map/imagery and road context. Resolve the 15 distant mesh-centre attachments, distinguishing centre-location artifacts from network/data problems. Adopt the national connection and attachment rules only after this review. This pilot does not resolve missing traffic volumes, maintenance condition/cost, fiscal capacity or historical population change.

## Source and implementation references

- [Official N13 road specifications](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html)
- [Official fire facility specifications](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P17.html)
- [Shapely spatial index documentation](https://shapely.readthedocs.io/en/stable/strtree.html)
- [pyproj transformation documentation](https://pyproj4.github.io/pyproj/stable/api/transformer.html)
'''
(OUT/'README.md').write_text(text)
print('Report written:',OUT/'README.md')
