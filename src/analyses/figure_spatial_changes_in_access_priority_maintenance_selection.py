#!/usr/bin/env python3
"""Spatial Changes in Access-Priority Maintenance Selection.

Plan: adopted four-year 2x2 spatial supplement at 1% relative-cost budget.
Framework: AnaSOP Sections 5-7; existing fixed-prefix access-priority sets,
with display-only membership classes relative to forecast-matched 2020.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tr01-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from PIL import Image
from figure_national_candidate_medical_distance_and_analysis_sample_coverage import (
    basemap_views, style_map, LAND, BOUNDARY, BASEMAP_DIR)

ROOT = Path(__file__).resolve().parents[2]
TITLE = 'Spatial Changes in Access-Priority Maintenance Selection'
SLUG = 'spatial_changes_in_access_priority_maintenance_selection'
EXP = ROOT / 'data/exp/figure-table-generation' / SLUG
OUT = ROOT / f'data/results/figures/Figure_{SLUG}.png'
YEARS = [2020, 2030, 2040, 2050]
COLORS = {'Retained / reference': '#2166AC', 'Exit': '#D95F02', 'New': '#8E44AD'}
FIGURE_MM = (190, 240)
DATA = ROOT / 'data/processed/demographic-stability'
UNITS = ROOT / 'data/processed/common-budget/conditional_maintenance_units_preprocessed.parquet'
GEOMETRY = ROOT / 'data/exp/figure-table-generation/road_service_dependence_and_measured_traffic/source_road_display_geometry.parquet'
RANKS = ROOT / 'data/exp/demographic-stability/priority_lists.parquet'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert json.loads((ROOT / 'data/exp/demographic-stability/validation.json').read_text())['status'] == 'passed'
    sources = [UNITS, RANKS, GEOMETRY, DATA / 'scenario_losses_preprocessed.parquet',
               DATA / 'scenario_protection_preprocessed.parquet', DATA / 'priority_stability_preprocessed.parquet', BOUNDARY]
    sources += [BASEMAP_DIR / f'neighbour_context/ne_110m_admin_0_countries.{ext}' for ext in ['shp', 'shx', 'dbf', 'prj']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    geometry_receipt = json.loads(GEOMETRY.with_name('geometry_validation.json').read_text())
    assert hashes[str(GEOMETRY.relative_to(ROOT))] == geometry_receipt['output_sha256']
    u = pd.read_parquet(UNITS)
    losses = pd.read_parquet(DATA / 'scenario_losses_preprocessed.parquet')
    ranks = pd.read_parquet(RANKS)
    curves = pd.read_parquet(DATA / 'scenario_protection_preprocessed.parquet')
    stability = pd.read_parquet(DATA / 'priority_stability_preprocessed.parquet')
    total_cost = float(u['Relative Intervention Cost'].sum())
    requested_budget = .01 * total_cost
    sets, accounting = {}, []
    for year in YEARS:
        ranked = ranks[ranks['Population-Change Scenario'].eq(year) & ranks['Maintenance Priority'].eq('Access priority')].sort_values('Priority Position')
        np.testing.assert_array_equal(ranked['Priority Position'], np.arange(1, len(u)+1))
        inputs = u.drop(columns=['Lost Medical Path Population', 'Additional Medical Distance Burden']).merge(
            losses[losses['Population-Change Scenario'].eq(year)], on='Graph Source Edge ID', validate='one_to_one')
        inputs['Reachability Per Cost'] = inputs['Lost Medical Path Population']/inputs['Relative Intervention Cost']
        inputs['Burden Per Cost'] = inputs['Additional Medical Distance Burden']/inputs['Relative Intervention Cost']
        independent = inputs.sort_values(['Reachability Per Cost', 'Burden Per Cost', 'Candidate Road Source Record'],
                                         ascending=[False, False, True], kind='stable')
        np.testing.assert_array_equal(independent['Graph Source Edge ID'], ranked['Graph Source Edge ID'])
        row = curves.loc[curves['Population-Change Scenario'].eq(year) & curves['Maintenance Priority'].eq('Access priority')
                         & curves['Population Group'].eq('All matched residents') & curves['Budget Share'].eq(.01)].iloc[0]
        k = int(row['Selected Maintenance Unit Count'])
        sets[year] = set(ranked.iloc[:k]['Graph Source Edge ID'])
        costs = independent['Relative Intervention Cost'].to_numpy()
        spent = float(costs[:k].sum())
        np.testing.assert_allclose(spent, row['Actual Relative Spending'], rtol=1e-11, atol=1e-8)
        assert spent <= requested_budget and (k == len(u) or spent+costs[k] > requested_budget)
        accounting.append({'year': year, 'selected': k, 'actual_spending': spent,
                           'actual_share_percent': spent/total_cost*100,
                           'unused_budget': requested_budget-spent})
    reference = sets[2020]
    displayed_ids = sorted(set.union(*sets.values()))
    geometry = pd.read_parquet(GEOMETRY, filters=[('Graph Source Edge ID', 'in', displayed_ids)])
    assert len(geometry) == len(displayed_ids) and geometry['Graph Source Edge ID'].is_unique
    geometry = geometry.sort_values('Graph Source Edge ID').reset_index(drop=True)
    np.testing.assert_array_equal(geometry['Graph Source Edge ID'], displayed_ids)
    ids = geometry['Graph Source Edge ID'].to_numpy()
    longitude, latitude = geometry['Display Longitude'].to_numpy(), geometry['Display Latitude'].to_numpy()
    curves_geometry = shapely.from_wkb(geometry['Geometry WKB'].to_numpy())
    assert (shapely.get_type_id(curves_geometry) == 1).all()
    _, coordinates, offsets = shapely.to_ragged_array(curves_geometry)
    ends = offsets[0]
    segments = [coordinates[ends[i]:ends[i+1]] for i in range(len(ids))]
    regions = [
        ('Main islands', latitude >= 30, (128.2, 147., 29.8, 46.4), None),
        ('Nansei', (latitude < 30) & (longitude < 134), (122.5, 131.7, 23.75, 30.15), [.16, .68, .36, .25]),
        ('Ogasawara', (latitude < 30) & (longitude >= 134), (140.95, 142.55, 24.45, 27.45), [.07, .36, .16, .23])]
    np.testing.assert_array_equal(sum(mask.astype(int) for _, mask, _, _ in regions), np.ones(len(ids)))
    bounds_all = shapely.bounds(curves_geometry)
    for _, mask, (west, east, south, north), _ in regions:
        bounds = bounds_all[mask]
        assert (bounds[:,0] >= west).all() and (bounds[:,2] <= east).all()
        assert (bounds[:,1] >= south).all() and (bounds[:,3] <= north).all()
    views = basemap_views(regions)
    sns.set_theme(style='white', context='paper', font='Arial')
    plt.rcParams.update({'font.size': 7, 'text.color': 'black', 'axes.labelcolor': 'black', 'path.simplify': False})
    fig = plt.figure(figsize=(FIGURE_MM[0]/25.4, FIGURE_MM[1]/25.4), dpi=600, facecolor='white')
    panels = [fig.add_axes(position) for position in [[.08,.56,.40,.38],[.57,.56,.40,.38],
                                                    [.08,.125,.40,.38],[.57,.125,.40,.38]]]
    display_records, inset_checks, note_checks = [], [], []
    for letter, year, ax, record in zip('abcd', YEARS, panels, accounting):
        current = sets[year]
        classes = {name: group for name, group in [
            ('Retained / reference', current & reference), ('Exit', reference-current), ('New', current-reference)]}
        assert set.union(*classes.values()) == current | reference
        assert sum(map(len, classes.values())) == len(current | reference)
        check = stability[stability['Population-Change Scenario'].eq(year) & stability['Budget Share'].eq(.01)].iloc[0]
        assert len(classes['Retained / reference']) == int(check['Intersection Unit Count'])
        assert len(current | reference) == int(check['Union Unit Count'])
        record.update(retained=len(classes['Retained / reference']), exiting=len(classes['Exit']), new=len(classes['New']))
        targets = []
        for name, mask, bounds, position in regions:
            target = ax if position is None else ax.inset_axes(position, zorder=8)
            style_map(target, bounds, inset=position is not None)
            target.tick_params(labelsize=5.5 if position else 6.5, pad=2.0,
                               length=2., width=.5, colors='black')
            target.grid(True, color='#9AADB8', linewidth=.35 if position else .45,
                        alpha=.75, linestyle=(0,(4,3)), zorder=1.0)
            if position is None:
                target.set_xticks([130, 138, 146]); target.set_yticks([30, 34, 38, 42, 46])
            elif name == 'Nansei':
                target.set_xticks([124, 130]); target.set_yticks([25, 29])
            else:
                target.set_xticks([142]); target.set_yticks([25,27])
            land, outlines, context = views[name]
            for path in context:
                target.add_patch(PathPatch(path, facecolor='#F0F1EF', edgecolor='#AAB4BA', linewidth=.2, zorder=.5))
            for path in land:
                target.add_patch(PathPatch(path, facecolor=LAND, edgecolor='none', zorder=.7))
            target.add_collection(LineCollection(outlines, colors='#84929C', linewidth=.18, zorder=1.5))
            for category, group in classes.items():
                index = np.flatnonzero(mask & np.isin(ids, list(group)))
                target.add_collection(LineCollection([segments[i] for i in index], colors=COLORS[category],
                                                      linewidth=.80 if category != 'New' else 1.15,
                                                      rasterized=True, zorder=3 if category == 'Retained / reference' else 4))
                display_records.append({'year':year, 'view':name, 'category':category, 'units':len(index)})
            if position is not None:
                target.text(.5,1.04,name,transform=target.transAxes,ha='center',va='bottom',fontsize=5.8,color='black')
                targets.append(target)
        ax.text(-.13,1.015,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',ha='left',va='bottom')
        ax.text(.96,.965,str(year),transform=ax.transAxes,fontsize=10,fontweight='bold',ha='right',va='top')
        note = ax.text(.96,.06,f'Selected: {len(current):,}\nRetained: {record["retained"]:,}; exit: {record["exiting"]:,}; new: {record["new"]:,}',
                       transform=ax.transAxes,ha='right',va='bottom',fontsize=5.8,color='black',linespacing=1.4)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        main_curves = curves_geometry[np.isin(ids, list(current | reference)) & regions[0][1]]
        for child in targets:
            corners = ax.transData.inverted().transform(child.get_window_extent(renderer).get_points())
            occluded = int(shapely.intersects(main_curves, shapely.box(*corners.ravel())).sum())
            inset_checks.append({'year':year,'main_curves_obscured_by_inset':occluded})
            assert occluded == 0, f'Inset obscures {occluded} main curves in {year}; bounds={corners.tolist()}'
        corners = ax.transData.inverted().transform(note.get_window_extent(renderer).get_points())
        occluded = int(shapely.intersects(main_curves, shapely.box(*corners.ravel())).sum())
        note_checks.append({'year':year,'main_curves_obscured_by_note':occluded})
        assert occluded == 0, f'Note obscures {occluded} main curves in {year}; bounds={corners.tolist()}'
    legend = fig.legend(handles=[Line2D([],[],color=c,linewidth=1.5,label=name) for name,c in COLORS.items()],
                        loc='lower center',bbox_to_anchor=(.53,.055),ncol=3,frameon=False,
                        fontsize=7.3,handlelength=2.5,columnspacing=1.7,title='Selection relative to matched 2020',title_fontsize=7.5)
    fig.text(.08,.026,'Bridge-associated units; access priority; 1% relative-cost budget.',fontsize=7,color='black')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside=[]
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box=item.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
                outside.append(item.get_text())
    assert not outside, outside
    for record in accounting:
        for category,key in [('Retained / reference','retained'),('Exit','exiting'),('New','new')]:
            assert sum(r['units'] for r in display_records if r['year']==record['year'] and r['category']==category) == record[key]
    fig.savefig(OUT,dpi=600,facecolor='white')
    fig.savefig(EXP/'preview.png',dpi=180,facecolor='white')
    plt.close(fig)
    pd.DataFrame(accounting).to_csv(EXP/'selection_accounting.csv',index=False)
    csv_rows=[]
    for year in YEARS:
        current=sets[year]
        for edge in sorted(current | reference):
            category='Retained / reference' if edge in current & reference else 'Exit' if edge in reference else 'New'
            csv_rows.append({'Graph Source Edge ID':edge,'Population-Change Scenario':year,'Display Selection Status':category})
    pd.DataFrame(csv_rows).to_csv(EXP/'display_selection_membership.csv',index=False)
    for rel,h in hashes.items():
        assert digest(ROOT/rel) == h
    with Image.open(OUT) as image:
        dimensions,dpi=list(image.size),list(image.info['dpi'])
    assert all(abs(v-600)<.1 for v in dpi)
    caption='''# Spatial Changes in Access-Priority Maintenance Selection

Four national panels show access-priority maintenance selection under the matched 2020 reference and the 2030, 2040 and 2050 census-anchored official population scenarios, at a common requested budget of 1% of summed relative intervention cost across the 46,688 conditional units. This percentage refers to relative cost, not the fraction of facilities selected; relative cost is bridge length multiplied by width, rather than observed repair spending. The original indivisible, fixed-prefix algorithm may leave unused budget, and differing unit costs allow the selected count to vary across years.

Blue represents selection in both the scenario year and matched 2020 (the reference-selected set in panel a). Orange identifies units selected in matched 2020 but outside the scenario-year prefix. Purple identifies newly selected units relative to matched 2020. Units selected in neither year are not highlighted. Each highlighted unit uses its original source-road curve; line widths are display choices rather than road width or disruption footprints. Common geographic bounds, pale-blue sea, official prefecture boundaries and the same Nansei and Ogasawara insets apply across all four panels. Every highlighted unit is accounted for once in the main or island view; insets do not obscure highlighted main-island curves. Small counts of new selections may be difficult to distinguish at national scale; the accompanying accounting file retains exact identities and counts.

Selection reuses the approved population-scenario comparison: 175,485 matched origins, the unchanged 46,688-unit event domain and fixed system inputs. Original local census population is multiplied by the official mesh future/reference ratio after summing municipality components, using the official 2024 [MLIT one-kilometre projection](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-mesh1000r6.html). The 127 unmatched origins with 278 census residents remain outside this figure's matched scenario analysis only. Original area membership, roads, medical destinations, traffic, inspection categories, relative costs, full event prevention and uniform mutually exclusive event weights remain fixed. Road locations identify conditional maintenance units, not protected residents or verified bridge footprints. Exiting the selected prefix does not imply that a road disappears or has no maintenance need. These maps locate conditional selection changes and do not establish observed future maintenance outcomes.
'''
    OUT.with_suffix('.caption.md').write_text(caption)
    validation={'status':'passed','authorization':'MILI-D-20261007-037','title':TITLE,
                'image':str(OUT.relative_to(ROOT)),'dimensions':dimensions,'dpi':dpi,'figure_size_mm':list(FIGURE_MM),
                'panels':4,'years':YEARS,'requested_budget_share':.01,'requested_relative_budget':requested_budget,
                'common_units':len(u),'original_road_curves':True,'full_curves_within_assigned_view':True,
                'independent_ranking_and_budget_prefix_verified':True,'category_counts_reproduce_jaccard':True,
                'all_displayed_units_accounted_once_per_panel':True,'selection_accounting':accounting,
                'inset_occlusion_checks':inset_checks,'note_occlusion_checks':note_checks,
                'all_text_within_canvas':True,'source_hashes_unchanged':hashes,'output_status':'generated_awaiting_human_review'}
    (EXP/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='source_hashes_unchanged'},indent=2))


if __name__ == '__main__':
    main()
