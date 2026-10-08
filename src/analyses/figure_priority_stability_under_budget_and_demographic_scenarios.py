#!/usr/bin/env python3
"""Priority Stability Under Budget and Demographic Scenarios.

AnaSOP Sections 5-7: adopted census-anchored spatial population sensitivity,
matched 2020 reference and selected-unit Jaccard at four requested budgets.
Human revision MILI-D-20261008-006 moves the former protection heatmap to
Reachability Protection Under Demographic Scenarios, without recomputation.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tr01-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SLUG = 'priority_stability_under_budget_and_demographic_scenarios'
TITLE = 'Priority Stability Under Budget and Demographic Scenarios'
OUT = ROOT / f'data/results/figures/Figure_{SLUG}.png'
EXP = ROOT / 'data/exp/figure-table-generation' / SLUG
SOURCE = ROOT / 'data/processed/demographic-stability/priority_stability_preprocessed.parquet'
YEARS = [2020, 2030, 2040, 2050]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    before = digest(SOURCE)
    stability = pd.read_parquet(SOURCE)
    computation = json.loads((ROOT / 'data/exp/demographic-stability/validation.json').read_text())
    assert computation['status'] == 'passed'
    assert len(stability) == 16 and stability['Priority Stability'].between(0, 1).all()
    sns.set_theme(style='white', context='paper', font='DejaVu Sans')
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.linewidth': .6, 'text.color': 'black'})
    fig = plt.figure(figsize=(190 / 25.4, 105 / 25.4), facecolor='white')
    ax = fig.add_axes([.12, .24, .85, .68])
    lines = []
    for budget, colour, marker in zip([.01, .05, .10, .20],
                                      ['#2166AC', '#D95F02', '#1B9E77', '#7B3294'],
                                      ['o', 's', '^', 'D']):
        frame = stability[stability['Budget Share'].eq(budget)].sort_values('Population-Change Scenario')
        np.testing.assert_array_equal(frame['Population-Change Scenario'], YEARS)
        line, = ax.plot(frame['Population-Change Scenario'], frame['Priority Stability'] * 100,
                        color=colour, marker=marker, label=f'{budget * 100:g}%',
                        markersize=4, linewidth=1.35, markeredgewidth=.6)
        lines.append(line)
    ax.set_xlim(2019, 2051)
    ax.set_ylim(65, 102)
    ax.set_xticks(YEARS)
    ax.set_yticks([70, 80, 90, 100])
    ax.set_xlabel('Population scenario year', labelpad=5)
    ax.set_ylabel('Selected-unit Jaccard similarity (%)', labelpad=6)
    ax.tick_params(length=3, width=.55, pad=3)
    ax.grid(axis='y', color='#D5DADF', linewidth=.45)
    note = ax.text(.03, .19, 'Access priority; matched 2020 reference',
                   transform=ax.transAxes, va='top', fontsize=8, color='black')
    legend = fig.legend(lines, [line.get_label() for line in lines],
                        title='Requested budget', loc='lower center', bbox_to_anchor=(.545, .01),
                        ncol=4, frameon=False, fontsize=8, title_fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = legend.get_window_extent(renderer)
    assert box.y1 < ax.get_tightbbox(renderer).y0
    note_box = note.get_window_extent(renderer)
    for line in lines:
        assert not any(note_box.contains(*point) for point in ax.transData.transform(
            np.column_stack([line.get_xdata(), line.get_ydata()])))
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            if bounds.width and bounds.height:
                assert bounds.x0 >= -1 and bounds.y0 >= -1 and bounds.x1 <= fig.bbox.width + 1 and bounds.y1 <= fig.bbox.height + 1, text.get_text()
    fig.savefig(OUT, dpi=600, facecolor='white')
    fig.savefig(EXP / 'preview.png', dpi=150, facecolor='white')
    plt.close(fig)
    assert digest(SOURCE) == before
    with Image.open(OUT) as image:
        dimensions, dpi = list(image.size), list(image.info['dpi'])
    assert all(abs(value - 600) < .1 for value in dpi)
    OUT.with_suffix('.caption.md').write_text("""# Priority Stability Under Budget and Demographic Scenarios

Selected-unit Jaccard similarity for access priority relative to the forecast-matched 2020 reference, at requested budgets of 1%, 5%, 10% and 20% of total relative intervention cost. Jaccard is intersection count divided by union count and measures selected-set overlap rather than ranking correlation or accessibility protection. The vertical axis spans 65-102% for readability. Lines connect four adopted population scenarios without adding intermediate evaluations. The former reachability-protection heatmap is reported separately in Reachability Protection Under Demographic Scenarios.

Official spatial population-change factors for 2030, 2040 and 2050 reweight original census population on 175,485 matched origins, with 125,874,510 census residents at the matched 2020 reference. The 127 unmatched original origins with 278 residents are excluded only from this matched demographic comparison. Genuine projected zeros remain zero. All 46,688 conditional bridge-associated units, relative costs, network, medical destinations and original area-group definitions remain fixed. Access priority is reranked under each population scenario using the adopted loss-per-cost rules and original final ties.

This is a conditional bridge demographic sensitivity; it does not forecast traffic, future elderly-resident locations, annual failure risk, realized maintenance effects or pavement outcomes. Projection source: [MLIT nationwide one-kilometre population projection](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-mesh1000r6.html).
""")
    result = {'status': 'passed', 'title': TITLE, 'authorization': 'MILI-D-20261007-036',
              'presentation_revision': 'MILI-D-20261008-006', 'image': str(OUT.relative_to(ROOT)),
              'dimensions': dimensions, 'dpi': dpi, 'figure_size_mm': [190, 105],
              'panels': 1, 'panel_letters': [], 'line_points': 16, 'legend_position': 'figure_bottom',
              'legend_overlap_check_passed': True, 'all_text_inside_canvas': True,
              'notes_black': True, 'no_note_or_legend_vertex_overlap': True,
              'source_hashes_unchanged': {str(SOURCE.relative_to(ROOT)): before},
              'matched_origins': 175485, 'unmatched_origins': 127, 'unmatched_census_population': 278,
              'common_units': 46688, 'former_heatmap_relocated_to': 'Reachability Protection Under Demographic Scenarios',
              'all_planned_metrics_follow_approved_framework': True,
              'output_status': 'generated_awaiting_human_review'}
    (EXP / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved: {OUT.relative_to(ROOT)} (600 dpi, 190 x 105 mm; single panel)')


if __name__ == '__main__':
    main()
