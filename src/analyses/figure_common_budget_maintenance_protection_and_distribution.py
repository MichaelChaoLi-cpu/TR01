#!/usr/bin/env python3
"""Common-Budget Maintenance Protection and Distribution.

Plan: three budget-protection panels on the adopted conditional common domain.
Framework: AnaSOP Sections 5-7, area-proportional costs, fixed-prefix priority
lists and uniformly weighted single-road events. No simultaneous-closure model.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tr01-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SLUG = 'common_budget_maintenance_protection_and_distribution'
TITLE = 'Common-Budget Maintenance Protection and Distribution'
SRC = ROOT / 'data/processed/common-budget/conditional_protection_curves_preprocessed.parquet'
OUT = ROOT / f'data/results/figures/Figure_{SLUG}.png'
EXP = ROOT / 'data/exp/figure-table-generation' / SLUG
COLOURS = {'Traffic priority': '#2166AC', 'Condition priority': '#D95F02', 'Access priority': '#1B9E77'}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    before = digest(SRC)
    frame = pd.read_parquet(SRC)
    assert len(frame) == 303 and not frame.isna().any().any()
    allocation = json.loads((ROOT / 'data/exp/common-budget-allocation/validation.json').read_text())
    groups = json.loads((ROOT / 'data/exp/data-preprocessing/common-budget/construction_validation.json').read_text())
    assert allocation['status'] == 'passed' and groups['status'] == 'passed'
    sns.set_theme(style='white', context='paper', font='Arial')
    plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'axes.linewidth': .55, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})
    fig = plt.figure(figsize=(190/25.4, 160/25.4), dpi=600, facecolor='white')
    axes = [fig.add_axes(pos) for pos in [[.12, .59, .34, .34], [.60, .59, .34, .34], [.12, .12, .34, .34]]]
    legend_ax = fig.add_axes([.60, .12, .34, .34])
    legend_ax.set_axis_off()
    ylabels = ['Additional distance burden\navoided (%)', 'Lost-path population\nprotected (%)', 'Group lost-path population\nprotected (%)']
    for label, ax, ylabel in zip('abc', axes, ylabels):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel('Budget share (%)', labelpad=4)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.tick_params(length=2.5, width=.5, pad=2)
        ax.grid(color='#D5DADF', linewidth=.4, alpha=.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(.55)
            spine.set_color('#30363B')
        ax.text(-.13, 1.055, label, transform=ax.transAxes, fontweight='bold', fontsize=10, va='top', color='black')
    curve_paths = []
    for rule, colour in COLOURS.items():
        f = frame[frame['Maintenance Priority'].eq(rule)].sort_values('Budget Share')
        x = f['Budget Share'].to_numpy() * 100
        for ax, col in zip(axes[:2], ['Distance Burden Protection Share', 'Reachability Protection Share']):
            line, = ax.plot(x, f[col].to_numpy()*100, color=colour, linewidth=1.2, zorder=3)
            curve_paths.append((ax, line))
        for col, ls, width in [('Higher Ageing Reachability Protection Share', '-', 1.1),
                               ('Distance Disadvantaged Reachability Protection Share', (0, (4, 2.5)), 1.45)]:
            line, = axes[2].plot(x, f[col].to_numpy()*100, color=colour, linestyle=ls, linewidth=width, zorder=3)
            if ls != '-':
                line.set_path_effects([pe.Stroke(linewidth=2.15, foreground='white'), pe.Normal()])
            curve_paths.append((axes[2], line))
    note_c = axes[2].text(.38, .09, 'Access-group curves\noverlap from 2% budget.', transform=axes[2].transAxes,
                         ha='left', va='bottom', fontsize=6.1, color='black')
    rule_handles = [Line2D([], [], color=c, lw=1.4, label=r) for r, c in COLOURS.items()]
    rule_legend = legend_ax.legend(handles=rule_handles, loc='upper left', bbox_to_anchor=(.03, 1.02),
                                  frameon=False, fontsize=8, title='Priority rule (a-c)', title_fontsize=8,
                                  handlelength=2.6, labelspacing=.8, borderaxespad=0)
    legend_ax.add_artist(rule_legend)
    group_handles = [Line2D([], [], color='black', lw=1.1, label='Higher-ageing areas'),
                     Line2D([], [], color='black', lw=1.45, linestyle=(0, (4, 2.5)), label='Longer-baseline-distance areas')]
    group_legend = legend_ax.legend(handles=group_handles, loc='upper left', bbox_to_anchor=(.03, .55),
                                   frameon=False, fontsize=7.5, title='Population area (c)', title_fontsize=8,
                                   handlelength=3, labelspacing=.8, borderaxespad=0)
    legend_ax.text(.03, .08, 'Conditional units: 46,688\nShared traffic units: 15,815',
                   transform=legend_ax.transAxes, ha='left', va='bottom', fontsize=7, color='black', linespacing=1.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    canvas = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > canvas.x1+.5 or box.y1 > canvas.y1+.5:
                outside.append(text.get_text())
    tick_overlap = 0
    for ax in axes:
        for texts in [ax.get_xticklabels(), ax.get_yticklabels()]:
            boxes = [t.get_window_extent(renderer) for t in texts if t.get_visible()]
            tick_overlap += sum(Bbox.overlaps(a, b) for a, b in zip(boxes[:-1], boxes[1:]))
    note_hits = 0
    for note, target in [(note_c, axes[2])]:
        box = note.get_window_extent(renderer).expanded(1.03, 1.08)
        for ax, line in curve_paths:
            if ax is target:
                display_path = line.get_path().transformed(line.get_transform())
                note_hits += int(display_path.intersects_bbox(box, filled=False))
    assert not outside and tick_overlap == 0 and note_hits == 0
    assert digest(SRC) == before
    fig.savefig(OUT, dpi=600, facecolor='white')
    plt.close(fig)
    with Image.open(OUT) as im:
        size, dpi = im.size, im.info['dpi']
    caption = f'''# {TITLE}

Conditional maintenance protection under common requested relative budgets. The two-by-two layout places panel a at the upper left, panel b at the upper right and panel c at the lower left; the lower-right cell contains legends and sample notes, without adding another analytical panel. Panel a shows the share of additional finite-route medical distance burden avoided; panel b shows the share of lost-path population protected; panel c shows group-specific shares of lost-path population protected. Colours identify traffic, condition and access priorities. Solid and dashed curves in panel c identify higher-ageing and longer-baseline-distance areas; the access-group curves coincide from the evaluated 2% budget onward. Line segments join evaluations at one-percentage-point budget intervals. The separate group-style legend applies to panel c only.

The common domain contains 46,688 one-bridge/one-source-road conditional units across all 47 prefectures, sharing 15,815 measured 2021 daytime 12-hour traffic observation units. The source bridge inventory retains all 730,844 records. The adopted anchor scenario retains a unique nearby candidate within 50 m with no alternative within nearest distance plus 5 m, and withholds class/level conflicts, tunnel-nearest, old-asset, source-quality and shared-nearest-feature cases. This is an assumed road association, not independently verified bridge identity or a complete physical bridge footprint. Condition categories and dimensions are observed heterogeneous latest inspection inputs; the candidate graph is from 2024 and baseline population/medical registration from 2020.

Relative intervention cost equals bridge length times width. Budget share is the requested budget divided by total common-domain relative cost. Fixed ordered lists select their longest affordable complete prefix, retain indivisible units and may leave unused budget; the maximum unspent share across these evaluations is {100*allocation['maximum_unspent_share']:.2f}% of total cost. These rules are priority policies rather than proven optimal allocations. Traffic priority ranks observed volume per cost; condition priority orders categories IV, III, II and I, then lower cost within category; access priority ranks lost-path population per cost, then additional distance burden per cost. Thus access priority explicitly targets the evaluated medical reachability outcome and is not an equity criterion by definition.

Selecting a unit is assumed to prevent its own single-source-road loss event completely. Events are mutually exclusive and equally weighted. Protection is the selected units' loss divided by the loss over all common-domain events, separately for each outcome or group. Repeated population across distinct events is not a count of unique beneficiaries. These are conditional single-event loss averages, not annual benefits, measured maintenance effects, real expenditure or a scenario closing all unselected roads simultaneously. Both outcomes retain fixed origin and destination entries and parallel road alternatives; disconnected origins receive no arbitrary finite distance penalty. Loss of a fixed origin entry is included in reachability loss and is not automatically downstream network isolation.

Area groups use the upper quartile of the original 145,771 selected official publication supports, with all cutoff ties retained: ageing share at least {100*groups['ageing_cutoff']:.3f}% among known ages, and population-weighted baseline medical distance at least {groups['baseline_distance_cutoff_metres']/1000:.3f} km. They contain {groups['higher_ageing_population']:,} and {groups['distance_disadvantaged_population']:,} originally selected residents, respectively, and overlap by {groups['overlap_population']:,} residents. These are area-composition labels, not identified individual older residents. Group loss is recomputed from affected origins on the unchanged 175,612-mesh, 125,874,788-resident baseline, and each group/complement sum reconciles to the accepted national road-loss total. Group shares have their own denominators and are not additive.

Display: 190 x 160 mm; 600 dpi PNG; no figure title. Analytical decision: MILI-D-20261007-026; layout revision: MILI-D-20261007-027. This figure remains pending human output review.
'''
    OUT.with_suffix('.caption.md').write_text(caption)
    frame.to_csv(EXP / 'display_values.csv', index=False)
    v = {'status': 'passed', 'title': TITLE, 'authorization': 'MILI-D-20261007-026',
         'image': str(OUT.relative_to(ROOT)), 'dimensions': list(size), 'dpi': list(dpi), 'figure_size_mm': [190, 160],
         'layout': 'two-by-two with three analytical panels and a lower-right legend cell',
         'layout_authorization': 'MILI-D-20261007-027',
         'source': str(SRC.relative_to(ROOT)), 'source_sha256': before, 'source_unchanged': True,
         'panels': 3, 'rules': 3, 'curve_rows': len(frame), 'common_units': 46688,
         'text_outside_canvas': outside, 'tick_label_overlap_count': int(tick_overlap),
         'curve_intersections_with_note_boxes': int(note_hits), 'title_on_canvas': False,
         'status_of_output_review': 'generated_awaiting_human_review'}
    (EXP / 'validation.json').write_text(json.dumps(v, indent=2)+'\n')
    print(json.dumps(v, indent=2))


if __name__ == '__main__':
    main()
