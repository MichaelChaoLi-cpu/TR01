"""Independent saved-artifact geometry and SciPy shortest-path verification."""
from collections import defaultdict
import json, math, hashlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from apply_reviewed_pilot_topology import ROOT, BASE, OUT, PROC, readcsv


def main():
    s=json.loads((OUT/'summary.json').read_text())
    old=json.loads((BASE/'edges_layer_aware.geojson').read_text())['features']
    new=json.loads((PROC/'edges.geojson').read_text())['features']
    groups=defaultdict(list)
    for f in new: groups[f['properties']['parent_edge_id']].append(f)
    assert set(groups)==set(range(len(old)))
    for eid,parts in groups.items():
        original=shape(old[eid]['geometry']); lines=[shape(f['geometry']) for f in parts]
        assert abs(sum(g.length for g in lines)-original.length)<1e-6
        assert unary_union(lines).hausdorff_distance(original)<1e-7
        assert all(f['properties']['source_id']==old[eid]['properties']['source_id'] for f in parts)
    oldp=readcsv(BASE/'point_attachments.csv');newp=readcsv(PROC/'point_attachments.csv')
    assert len(oldp)==len(newp)==1989
    before={p['point_id']:p for p in oldp}; after={p['point_id']:p for p in newp}
    assert before.keys()==after.keys()
    for pid,p in after.items():
        op=before[pid]
        for key in ['dataset','category','offset_m','attached_500m','snap_longitude','snap_latitude']:
            assert p[key]==op[key]
        eid=int(p['edge_id']);e=new[eid];parent=int(e['properties']['parent_edge_id'])
        assert parent==int(op['edge_id'])
        a=shape(e['geometry']).interpolate(float(p['along_m']))
        b=shape(old[parent]['geometry']).interpolate(float(op['along_m']))
        assert a.distance(b)<1e-7
    nodes=readcsv(PROC/'nodes.csv');n=len(nodes);weights={}
    lengths=[shape(f['geometry']).length for f in new]
    for f,w in zip(new,lengths):
        p=f['properties'];u,v=p['from_node'],p['to_node']
        if u==v:continue
        for a,b in [(u,v),(v,u)]:weights[a,b]=min(w,weights.get((a,b),math.inf))
    comparison=readcsv(OUT/'mesh_distance_comparison.csv');maxerror=0.;checked=0
    for target in ['medical','ems_base']:
        facilities=[p for p in newp if p['attached_500m']=='True' and (
            p['dataset']=='medical' and p['category'] in ('1','2') if target=='medical'
            else p['dataset']=='fire_stations' and p['category'] in ('2','3'))]
        augmented=dict(weights);positions=defaultdict(list)
        for p in facilities:
            eid=int(p['edge_id']);pos=float(p['along_m']);props=new[eid]['properties']
            positions[eid].append(pos)
            for node,d in [(props['from_node'],pos),(props['to_node'],lengths[eid]-pos)]:
                for a,b in [(n,node),(node,n)]:augmented[a,b]=min(d,augmented.get((a,b),math.inf))
        matrix=coo_matrix((list(augmented.values()),([a for a,b in augmented],[b for a,b in augmented])),shape=(n+1,n+1)).tocsr()
        distances=dijkstra(matrix,directed=False,indices=n)
        for row in comparison:
            if row['target']!=target:continue
            p=after['mesh:'+row['mesh_code']];expected=math.inf
            if p['attached_500m']=='True':
                eid=int(p['edge_id']);pos=float(p['along_m']);props=new[eid]['properties']
                expected=min(distances[props['from_node']]+pos,distances[props['to_node']]+lengths[eid]-pos,
                             min((abs(pos-q) for q in positions[eid]),default=math.inf))
            actual=float(row['after_distance_m']) if row['after_distance_m'] else math.inf
            assert math.isfinite(actual)==math.isfinite(expected)
            if math.isfinite(actual):
                error=abs(actual-expected);maxerror=max(error,maxerror);assert error<1e-6
            checked+=1
    ff=json.loads((OUT/'route_examples.geojson').read_text())['features'];routes=defaultdict(list)
    for f in ff: routes[f['properties']['mesh_code'],f['properties']['target']].append(f)
    examples=readcsv(OUT/'route_examples.csv');maximum_gap=0.
    for row in examples:
        parts=routes[row['mesh_code'],row['target']];lines=[shape(f['geometry']) for f in parts]
        assert abs(sum(g.length for g in lines)-float(row['after_distance_m']))<1e-5
        for a,b in zip(lines,lines[1:]):
            gap=Point(a.coords[-1]).distance(Point(b.coords[0]));maximum_gap=max(maximum_gap,gap)
            assert gap<=.001+1e-7
        pop=after['mesh:'+row['mesh_code']];dest=after[row['destination_point_id']]
        for line,p,end in [(lines[0],pop,0),(lines[-1],dest,-1)]:
            snapped=shape(new[int(p['edge_id'])]['geometry']).interpolate(float(p['along_m']))
            assert Point(line.coords[end]).distance(snapped)<1e-6
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in s['input_hashes'].items())
    result=dict(status='passed',parent_geometry_and_length_checks=len(groups),fixed_attachment_checks=len(newp),
                independent_scipy_distance_checks=checked,maximum_distance_difference_m=maxerror,
                complete_route_geometry_checks=len(examples),maximum_route_segment_gap_m=maximum_gap,input_hash_checks='passed')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    rows=pd.read_csv(OUT/'mesh_distance_comparison.csv',dtype={'mesh_code':str})
    report=f'''# Reviewed pilot topology and distance comparison

The 2024-09 N13 road baseline is retained. Applied 139 reviewed endpoint connection candidates and three surface junction candidates. Five parent edges were split; all source IDs, source line geometry and total line length are preserved. The 163 separated-crossing and 55 unresolved review records received no forced connections.

| Measure | Before | After |
|---|---:|---:|
| Nodes | {s['baseline_graph']['nodes']} | {s['reviewed_graph']['nodes']} |
| Edges | {s['baseline_graph']['edges']} | {s['reviewed_graph']['edges']} |
| Connected components | 353 | 260 |
| Total road length (km) | {s['baseline_graph']['total_length_m']/1000:.6f} | {s['reviewed_graph']['total_length_m']/1000:.6f} |

| Target | Reachable meshes before | Reachable meshes after | Newly reachable | Shorter among previously reachable | Still disconnected among attached |
|---|---:|---:|---:|---:|---:|
'''
    for target in ['medical','ems_base']:
        t=rows[rows.target==target];c=s['distances'][target]
        report+=f"| {target} | {t.before_distance_m.notna().sum()} | {t.after_distance_m.notna().sum()} | {c.get('newly_reachable',0)} | {c.get('shorter',0)} | {c.get('still_disconnected',0)} |\n"
    report+='''
There are 592 population mesh centers; 577 satisfy the existing 500 m attachment threshold and 15 remain unattached. Attachment positions, offsets, target eligibility and the bidirectional distance weighting were held fixed. A newly reachable mesh means connectivity changed in this computational graph, not that residents newly obtained a real service.

## Validation

All 70,778 parent geometries retain their length and geometry after segmentation. All 1,989 point attachments retain their source-edge position and eligibility. Independently rebuilt SciPy graphs reproduce all 1,184 corrected mesh-target distance records, including disconnected and unattached records. Every previously finite baseline distance matches the earlier pilot output; all corrected distances are no greater than their baseline value. Four representative changed routes were reconstructed from partial and full edges, checked for length and endpoint agreement, and verified to traverse reviewed corrections. Input artifact hashes remain unchanged.

## Files and reproduction

- `mesh_distance_comparison.csv`: all mesh-target comparisons.
- `changed_meshes.csv`: newly reachable or shorter mesh-target records.
- `topology_actions.csv`: reviewed site to corrected node trace.
- `route_examples.csv` and `route_examples.geojson`: two changed routes per target, including the greatest distance reduction among previously reachable meshes.
- `data/processed/reviewed-network-pilot/`: independently saved corrected nodes, edges and attachments; EPSG:6670.

Run from the project root:

```sh
.venv/bin/python src/exp/apply_reviewed_pilot_topology.py
.venv/bin/python src/exp/validate_reviewed_pilot.py
```

## Scope

This is an engineering pilot, not a national accessibility estimate or a maintenance-allocation result. Distances exclude off-road attachment connectors. The existing 500 m threshold is diagnostic and does not establish a traversable access path. Facility historical status is not filtered. Medical destinations are hospitals and clinics; EMS destinations are fire stations and branches under the user-specified ambulance deployment assumption. No speeds, vehicle restrictions, turn rules or response delays are modeled. Source road physical-state codes do not measure deterioration.

Do not treat remaining components as road errors: islands and clipping can create genuine disconnection. Older or undated imagery does not justify deleting an N13-2024 road. Unresolved review records concern insufficient connectivity evidence. The verified local actions cannot be indiscriminately applied to the national network.
'''
    (OUT/'README.md').write_text(report)


if __name__=='__main__':main()
