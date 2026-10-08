"""Pavement Maintenance Protection Under Budget Constraints.

AnaSOP Sections 5-7: exact joint finite-cost evaluation of fixed priorities,
original population and area supports, complete-prefix budgets and fixed-set
impedance sensitivities. Data and conditional scenario assumptions are unchanged.
"""
from pathlib import Path
import json
import os

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tr01-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
SLUG='pavement_maintenance_protection_under_budget_constraints'
EXP=ROOT/'data/exp/figure-table-generation'/SLUG
OUT=ROOT/'data/results/figures'/f'Figure_{SLUG}.png'
RULES=['Traffic','Condition','Access']
COLORS={'Traffic':'#0072B2','Condition':'#D55E00','Access':'#009E73'}


def main():
    EXP.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((ROOT/'data/exp/data-preprocessing/pavement-policy/construction_validation.json').read_text())
    assert receipt['status']=='passed_readable_pavement_policy_construction'
    df=pd.read_parquet(ROOT/'data/processed/pavement-policy/pavement_group_outcomes_preprocessed.parquet')
    sns.set_theme(context='paper',style='ticks',font='DejaVu Sans')
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'text.color':'black'})
    fig,axes=plt.subplots(2,2,figsize=(190/25.4,170/25.4))
    fig.subplots_adjust(left=.10,right=.92,bottom=.16,top=.94,wspace=.33,hspace=.34)
    labels=['All residents','Higher-ageing areas','Distance-disadvantaged areas']
    groups=['All Residents','Higher Ageing Area','Baseline Distance Disadvantaged Area']
    for ax,group,label in zip(axes.flat,groups,labels):
        for rule in RULES:
            data=df[df['Pavement Impedance Increment'].eq(.25)&df['Pavement Population Group'].eq(group)&df['Pavement Maintenance Priority'].eq(rule)].sort_values('Pavement Budget Share')
            assert len(data)==4
            ax.plot(data['Pavement Budget Share']*100,data['Pavement Access Protection Share']*100,
                    color=COLORS[rule],marker={'Traffic':'o','Condition':'s','Access':'^'}[rule],
                    markersize=4,linewidth=1.4,label=rule+' priority')
        ax.set_xlim(0,21);ax.set_xticks([1,5,10,20]);ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100])
        ax.set_xlabel('Requested pavement budget (%)');ax.set_ylabel('Access burden protected (%)')
        ax.text(.04,.965,label,transform=ax.transAxes,ha='left',va='top',fontsize=8,color='black')
        ax.grid(axis='y',color='#DDE2E5',linewidth=.5);sns.despine(ax=ax)
    ax=axes[1,1]
    subset=df[df['Pavement Budget Share'].eq(.01)&df['Pavement Population Group'].eq('All Residents')]
    matrix=subset.pivot(index='Pavement Maintenance Priority',columns='Pavement Impedance Increment',values='Pavement Access Protection Share').loc[RULES,[.10,.25,.50]]*100
    assert matrix.shape==(3,3) and matrix.notna().all().all()
    cmap=LinearSegmentedColormap.from_list('pale_protection',['#FAFCFE','#80B1D3'])
    sns.heatmap(matrix,ax=ax,cmap=cmap,vmin=0,vmax=float(matrix.to_numpy().max()),annot=True,fmt='.2f',
                annot_kws={'fontsize':8,'color':'black'},linewidths=.6,linecolor='white',
                cbar_kws={'label':'Protected (%)','fraction':.05,'pad':.04})
    ax.set_xticklabels(['10%','25%','50%'],rotation=0);ax.set_yticklabels(RULES,rotation=0)
    ax.set_xlabel('Unmaintained-road impedance increment');ax.set_ylabel('')
    ax.text(.5,1.03,'All residents; 1% budget',transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color='black')
    for label,ax in zip('abcd',axes.flat):
        ax.text(-.16,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=11,va='top',color='black')
    handles,labels=axes[0,0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=8)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    legend_box=legend.get_window_extent(renderer)
    lower_panel_boxes=[ax.get_tightbbox(renderer) for ax in axes[1,:]]
    assert legend_box.y1 < min(box.y0 for box in lower_panel_boxes)
    assert fig.bbox.contains(legend_box.x0,legend_box.y0) and fig.bbox.contains(legend_box.x1,legend_box.y1)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT,dpi=600)
    fig.savefig(EXP/'preview.png',dpi=150)
    plt.close(fig)
    caption='''# Pavement Maintenance Protection Under Budget Constraints

Panels a-c compare aggregate, higher-ageing-area and baseline-distance-disadvantaged-area protection at requested budgets of 1, 5, 10 and 20 percent, under a 25 percent unmaintained-road impedance increment. Panel d compares aggregate protection at a 1 percent budget under 10, 25 and 50 percent increments, keeping the baseline priority lists and selections fixed. Protection is the fraction of population-weighted excess generalized medical-access cost avoided relative to the same group's all-unmaintained reference. Group denominators differ and the two area groups overlap. Lines connect evaluated budget points without adding intermediate evaluations. Results concern 70,049 conditional pavement units on a fixed national network and selected population, with length-relative costs and assumed finite impairment; they are not observed repair effects, travel-time savings or individual elderly outcomes.
'''
    OUT.with_suffix('.caption.md').write_text(caption)
    with Image.open(OUT) as im:
        assert min(im.info['dpi'])>599
        dimensions=list(im.size)
    (EXP/'validation.json').write_text(json.dumps({'status':'passed','panels':4,'layout':'two_by_two',
          'dpi':600,'figure_size_mm':[190,170],'dimensions':dimensions,'curve_points':36,'sensitivity_cells':9,
          'framework_followed':True,'legend_position':'figure_bottom','legend_overlap_check_passed':True,
          'human_acceptance_pending':True},indent=2)+'\n')
    print(f'Saved {OUT.relative_to(ROOT)} (600 dpi, 190 x 170 mm)')


if __name__=='__main__':main()
