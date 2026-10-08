"""Design previews for the three adopted national pavement figures.

Real basemap geometry only. No policy values, selections or illustrative data
are used. Previews stay in data/exp and do not change acceptance status.
"""
from pathlib import Path
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tr01-matplotlib')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analyses'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import seaborn as sns
from PIL import Image

from figure_national_candidate_medical_distance_and_analysis_sample_coverage import (
    basemap_views, style_map, LAND,
)

OUT = ROOT / 'data/exp/figure-table-generation/national-pavement-layout-design'
RULES = ['Traffic', 'Condition', 'Access']
COLORS = ['#0072B2', '#D55E00', '#009E73']
CMAP = LinearSegmentedColormap.from_list(
    'blue_green_yellow_red', ['#2166AC', '#1A9850', '#FEE08B', '#D73027'])
REGIONS = [
    ('Main islands', None, (128.2, 146.4, 29.8, 46.1), None),
    ('Nansei islands', None, (122.5, 131.7, 23.75, 30.15), [.055, .735, .34, .225]),
    ('Ogasawara', None, (140.95, 142.55, 24.45, 27.45), [.055, .395, .15, .22]),
    ('Minamitorishima', None, (153.945, 154.02, 24.265, 24.325), [.265, .465, .15, .10]),
]


def letter(ax, label):
    ax.text(-.10, 1.035, label, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', color='black')


def pending(ax, *, x=.5, y=.5):
    ax.text(x, y, 'Results pending', transform=ax.transAxes, ha='center',
            va='center', fontsize=8, color='#59636D',
            bbox={'facecolor': 'white', 'edgecolor': '#CBD2D8',
                  'linewidth': .5, 'boxstyle': 'round,pad=.4'})


def save(fig, slug, mm, extra=None):
    fig.text(.5, .008, 'Layout preview - policy results not computed',
             ha='center', fontsize=6.5, color='black')
    fig.canvas.draw()
    # A physical-canvas check catches misplaced panel labels and axis captions.
    renderer = fig.canvas.get_renderer()
    clipped = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text() or item.get_text() == '':
            continue
        box = item.get_window_extent(renderer)
        # Text on hidden Matplotlib tick positions is excluded.
        if item.axes is not None and not item.axes.get_visible():
            continue
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            clipped.append(item.get_text())
    final = OUT / f'{slug}_layout_600dpi.png'
    preview = OUT / f'{slug}_preview.png'
    fig.savefig(final, dpi=600)
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    with Image.open(final) as im:
        assert min(im.info['dpi']) > 599
        dimensions = list(im.size)
    receipt = {'kind': 'layout_design_only', 'policy_data_used': False,
               'policy_results_generated': False, 'dpi': 600,
               'figure_size_mm': mm, 'dimensions': dimensions,
               'text_outside_canvas': clipped, 'preview': str(preview.relative_to(ROOT))}
    if extra:
        receipt.update(extra)
    return receipt


def protection():
    fig, axes = plt.subplots(2, 2, figsize=(190/25.4, 180/25.4))
    fig.subplots_adjust(left=.11, right=.96, bottom=.20, top=.94,
                        wspace=.32, hspace=.40)
    groups = ['All residents', 'Higher-ageing areas', 'Distance-disadvantaged areas']
    for ax, name in zip(axes.flat, groups):
        ax.set_xlim(0, 21); ax.set_xticks([1, 5, 10, 20])
        ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Requested pavement budget (%)')
        ax.set_ylabel('Access burden protected (%)')
        ax.text(.04, .965, name, transform=ax.transAxes, fontsize=8, va='top')
        ax.grid(axis='y', color='#DDE2E5', linewidth=.5)
        sns.despine(ax=ax)
        pending(ax)
    ax = axes[1, 1]
    ax.set_xlim(0, 3); ax.set_ylim(3, 0); ax.set_aspect('equal')
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor='#F7F9FA',
                                   edgecolor='white', linewidth=1.2))
    ax.set_xticks([.5, 1.5, 2.5], ['10%', '25%', '50%'])
    ax.set_yticks([.5, 1.5, 2.5], RULES)
    ax.tick_params(length=0)
    ax.set_xlabel('Unmaintained-road impedance increment')
    ax.text(.5, 1.06, 'All residents; 1% budget', transform=ax.transAxes,
            ha='center', fontsize=8)
    sns.despine(ax=ax, left=True, bottom=True)
    pending(ax)
    for label, target in zip('abcd', axes.flat):
        target.text(-.10, 1.065, label, transform=target.transAxes,
                    fontsize=11, fontweight='bold', va='bottom', color='black')
    cax = fig.add_axes([.64, .125, .25, .012])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(
        norm=Normalize(0, 1), cmap=LinearSegmentedColormap.from_list(
            'protection_design', ['#FAFCFE', '#80B1D3'])),
        cax=cax, orientation='horizontal')
    bar.set_ticks([])
    bar.set_label('Protected (%); range pending', fontsize=7)
    handles = [Line2D([], [], color=c, marker=m, linewidth=1.4,
                      markersize=4, label=r+' priority')
               for r, c, m in zip(RULES, COLORS, ['o', 's', '^'])]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .048),
               ncol=3, frameon=False, fontsize=8)
    return save(fig, 'national_protection', [190, 180])


def map_panel(ax, views, label, note):
    for name, _, bounds, position in REGIONS:
        target = ax if position is None else ax.inset_axes(position)
        style_map(target, bounds, inset=position is not None)
        target.tick_params(labelsize=4 if position else 6,
                           length=1.5 if position else 2.5,
                           pad=1.5 if position else 2.5)
        if position is None:
            target.set_xticks([130, 135, 140, 145])
            target.set_yticks([30, 35, 40, 45])
        elif name in ['Ogasawara', 'Minamitorishima']:
            target.set_xticks([(bounds[0]+bounds[1])/2])
            if name == 'Minamitorishima':
                target.set_yticks([(bounds[2]+bounds[3])/2])
        land, outlines, context = views[name]
        for path in context:
            target.add_patch(PathPatch(path, facecolor='#F0F1EF',
                                       edgecolor='#AAB4BA', linewidth=.2, zorder=.5))
        for path in land:
            target.add_patch(PathPatch(path, facecolor=LAND, edgecolor='none', zorder=.8))
        target.add_collection(LineCollection(outlines, colors='#73818D',
                                             linewidth=.25, zorder=3))
        if position is not None:
            target.text(.02, 1.04, name, transform=target.transAxes,
                        fontsize=4.5, va='bottom', color='black')
    letter(ax, label)
    ax.text(.97, .98, note, transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
    pending(ax, x=.70, y=.15)


def maps(views, *, allocation):
    fig, axes = plt.subplots(2, 2, figsize=(190/25.4, 210/25.4))
    fig.subplots_adjust(left=.075, right=.975, bottom=.14, top=.96,
                        wspace=.18, hspace=.15)
    notes = ['Traffic priority', 'Condition priority', 'Access priority'] if allocation else [
        'No maintenance', 'Traffic priority', 'Condition priority', 'Access priority']
    for label, ax, note in zip('abcd', axes.flat, notes):
        map_panel(ax, views, label, note)
    if allocation:
        ax = axes[1, 1]
        ax.set_xlim(0, 3); ax.set_ylim(3, 0); ax.set_aspect('equal')
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor='#F7F9FA',
                                       edgecolor='white', linewidth=1.2))
        ax.set_xticks([.5, 1.5, 2.5], RULES)
        ax.set_yticks([.5, 1.5, 2.5], RULES)
        ax.tick_params(labelsize=7, length=0)
        sns.despine(ax=ax, left=True, bottom=True)
        letter(ax, 'd')
        ax.text(.5, 1.06, 'Selection overlap', transform=ax.transAxes,
                ha='center', fontsize=8)
        pending(ax)
        ax.text(.5, -.095, 'Intersection / union', transform=ax.transAxes,
                ha='center', fontsize=7.5)
    cax = fig.add_axes([.12, .068, .80, .018])
    norm = Normalize(0, 100 if allocation else 1)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP),
                       cax=cax, orientation='horizontal')
    if allocation:
        bar.set_ticks([0, 25, 50, 75, 100])
        bar.set_label('Panels a-c: maintained candidate-road length within mesh (%)', fontsize=7.5)
    else:
        bar.set_ticks([])
        bar.set_label('Residual access burden (resident reference-km); range pending', fontsize=7.5)
    bar.ax.tick_params(labelsize=7, pad=2)
    slug = 'national_allocation' if allocation else 'national_residual_burden'
    return save(fig, slug, [190, 210], {'maps': 3 if allocation else 4,
                                      'basemap': 'existing official prefecture display geometry',
                                      'thematic_values_plotted': False})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context='paper', style='ticks', font='DejaVu Sans')
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'text.color': 'black'})
    receipts = {'protection': protection()}
    views = basemap_views(REGIONS)
    receipts['residual_burden'] = maps(views, allocation=False)
    receipts['allocation'] = maps(views, allocation=True)
    (OUT/'design_validation.json').write_text(json.dumps(receipts, indent=2)+'\n')
    for name, receipt in receipts.items():
        print(name, receipt['preview'], receipt['text_outside_canvas'])


if __name__ == '__main__':
    main()
