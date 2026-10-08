"""National Spatial Allocation of Pavement Maintenance.

AnaSOP: native-mesh maintained candidate-road length shares for three fixed
1% budget selections, plus count-based Jaccard overlap of source-road sets.
Map colours use a shared linear 0-5 percent range with upper saturation;
stored shares and the panel d 0-100 percent overlap scale are unchanged.
"""
from pathlib import Path
import json
import os

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tr01-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_figure_checks import check_publication_layout
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from figure_national_candidate_medical_distance_and_analysis_sample_coverage import (
    mesh_raster_indices,native_raster,basemap_views,style_map,LAND)

ROOT=Path(__file__).resolve().parents[2]
SLUG='national_spatial_allocation_of_pavement_maintenance'
EXP=ROOT/'data/exp/figure-table-generation'/SLUG
OUT=ROOT/'data/results/figures'/f'Figure_{SLUG}.png'


def main():
    receipt=json.loads((ROOT/'data/exp/data-preprocessing/national-pavement-policy/construction_validation.json').read_text())
    assert receipt['status']=='passed_national_pavement_policy_construction' and receipt['joint_policy_cases']==36
    EXP.mkdir(parents=True,exist_ok=True)
    construction=json.loads((ROOT/'data/exp/national-pavement-allocation-computation/validation.json').read_text())
    assert construction['status']=='passed_national_mesh_allocation_construction'
    file=ROOT/'data/processed/national-pavement-allocation/national_pavement_mesh_allocation_preprocessed.parquet'
    values=[];frames=[];accounting=[]
    for rule in ['Traffic','Condition','Access']:
        df=pd.read_parquet(file,filters=[('Pavement Maintenance Priority','=',rule)])
        assert df['Mesh Code'].is_unique and df['Candidate Source Road Length'].gt(0).all()
        share=df['Mesh Maintained Road Length Share'].to_numpy()*100
        assert np.isfinite(share).all() and ((share>=0)&(share<=100+1e-10)).all()
        frames.append(df);values.append(share)
    for df in frames[1:]:np.testing.assert_array_equal(df['Mesh Code'],frames[0]['Mesh Code'])
    sample=frames[0];lat=sample['Latitude'].to_numpy();lon=sample['Longitude'].to_numpy()
    row,col=mesh_raster_indices(sample)
    regions=[('Main islands',lat>=30,(128.2,146.4,29.8,46.1),None),
             ('Nansei islands',(lat<30)&(lon<134),(122.5,131.7,23.75,30.15),[.14,.735,.34,.225]),
             ('Ogasawara',(lat<30)&(lon>=134)&(lon<150),(140.95,142.55,24.45,27.45),[.055,.395,.15,.22]),
             ('Minamitorishima',(lat<30)&(lon>=150),(153.945,154.02,24.265,24.325),[.265,.465,.15,.10])]
    coverage=sum(mask.astype(int) for _,mask,_,_ in regions);assert (coverage==1).all()
    regions=[r for r in regions if r[1].any()]
    basemaps=basemap_views(regions)
    vmax=5.0
    norm=Normalize(vmin=0,vmax=vmax,clip=True)
    cmap=LinearSegmentedColormap.from_list('blue_green_yellow_red',['#2166AC','#1A9850','#FEE08B','#D73027'])
    cmap.set_bad((0,0,0,0))
    sns.set_theme(context='paper',style='white',font='DejaVu Sans')
    fig,axes=plt.subplots(2,2,figsize=(190/25.4,210/25.4))
    fig.subplots_adjust(left=.075,right=.975,bottom=.12,top=.96,wspace=.18,hspace=.15)
    panel_names=['Traffic priority','Condition priority','Access priority']
    panel_receipts=[]
    for panel,ax,v,note in zip('abcd',axes.flat,values,panel_names):
        for name,mask,bounds,position in regions:
            target=ax if position is None else ax.inset_axes(position)
            style_map(target,bounds,inset=position is not None)
            target.tick_params(labelsize=6,length=1.5 if position else 2.5,pad=1.5 if position else 2.5)
            if position is None:
                target.set_xticks([130,135,140,145]);target.set_yticks([30,35,40,45])
            elif name in ['Ogasawara','Minamitorishima']:
                target.set_xticks([(bounds[0]+bounds[1])/2])
                if name=='Minamitorishima':target.set_yticks([(bounds[2]+bounds[3])/2])
            land,outlines,context=basemaps[name]
            for path in context:target.add_patch(PathPatch(path,facecolor='#F0F1EF',edgecolor='#AAB4BA',linewidth=.2,zorder=.5))
            for path in land:target.add_patch(PathPatch(path,facecolor=LAND,edgecolor='none',zorder=.8))
            grid,extent,rr,cc=native_raster(row[mask],col[mask],v[mask],np.nan,np.float64)
            np.testing.assert_array_equal(grid[rr,cc],v[mask])
            target.imshow(np.ma.masked_invalid(grid),extent=extent,origin='upper',interpolation='nearest',
                          resample=False,cmap=cmap,norm=norm,aspect=target.get_aspect(),zorder=2)
            target.add_collection(LineCollection(outlines,colors='#73818D',linewidth=.25,alpha=1,zorder=3))
            if position is not None:
                target.text(.02,1.04,name,transform=target.transAxes,fontsize=6,va='bottom',color='black')
        ax.text(-.10,1.035,panel,transform=ax.transAxes,fontweight='bold',fontsize=11,va='top',color='black')
        ax.text(.97,.98,note,transform=ax.transAxes,ha='right',va='top',fontsize=7.5,color='black')
        panel_receipts.append({'panel':panel,'note':note,'meshes':len(v),'nonzero_meshes':int(np.count_nonzero(v)),
                               'zero_meshes':int(np.count_nonzero(v==0)),
                               'maximum_mesh_maintained_length_percent':float(v.max()),
                               'display_maximum_length_share_percent':float(v.max()),
                               'meshes_above_colour_maximum':int(np.count_nonzero(v>vmax))})
    ax=axes[1,1]
    overlaps=pd.read_parquet(ROOT/'data/processed/national-pavement-allocation/national_pavement_rule_overlap_preprocessed.parquet')
    rules=['Traffic','Condition','Access']
    matrix=overlaps.pivot(index='First Pavement Priority',columns='Second Pavement Priority',values='Pavement Rule Selection Overlap').loc[rules,rules]*100
    assert matrix.notna().all().all()
    np.testing.assert_array_equal(matrix,matrix.T)
    np.testing.assert_allclose(np.diag(matrix),100)
    overlap_cmap=LinearSegmentedColormap.from_list('readable_overlap',['#FAFCFE','#80B1D3'])
    sns.heatmap(matrix,ax=ax,cmap=overlap_cmap,vmin=0,vmax=100,annot=True,fmt='.2f',cbar=False,
        square=True,linewidths=.6,linecolor='white',annot_kws={'fontsize':8,'color':'black'})
    ax.set_xlabel('');ax.set_ylabel('');ax.set_xticklabels(rules,rotation=0);ax.set_yticklabels(rules,rotation=0)
    ax.tick_params(labelsize=7)
    ax.text(-.10,1.075,'d',transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom',color='black')
    ax.text(.5,1.025,'Selected-road set overlap (%)',transform=ax.transAxes,ha='center',va='bottom',fontsize=7.5,color='black')
    cax=fig.add_axes([.12,.062,.80,.018])
    ticks=[0,1,2,3,4,5]
    bar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=ticks,extend='max')
    bar.ax.tick_params(labelsize=7,pad=2)
    bar.set_label('Maintained share of candidate road length within each mesh (%)',fontsize=7.5)
    fig.canvas.draw();OUT.parent.mkdir(parents=True,exist_ok=True)
    check_publication_layout(fig, EXP)
    fig.savefig(OUT,dpi=600);fig.savefig(EXP/'preview.png',dpi=150);plt.close(fig)
    OUT.with_suffix('.caption.md').write_text("""# National Spatial Allocation of Pavement Maintenance

Panels a-c map traffic, condition and access allocations at a common 1% requested national length-relative budget. Colours show the selected candidate source-road length within each native 1 km mesh divided by all candidate source-road length in that mesh. Original source curves are clipped at mesh boundaries; their metric clipped lengths allocate the original source-record costs proportionally, conserving both candidate and selected lengths. Boundary-only segments are assigned once using half-open cells. All road-bearing meshes are included, even outside the inhabited origin sample. Zero selected length is blue; cells with no candidate road denominator are NoData. The shared map scale is linear 0-5%. Values above 5% use the maximum colour, indicated by the upper colourbar extension. Only colours saturate; stored maintained-length shares are unchanged. Panel d retains its 0-100% overlap scale. Panel d shows count-based Jaccard similarity between unique original source-road selected sets: intersection divided by union, displayed as percentages. It does not weight road length, population or benefit. The maps and matrix use the same 1% sets as the protection and residual-burden comparisons, with explicitly hypothetical missing traffic and condition inputs. Road locations do not identify protected residents, physical repair cost or observed projects. Candidate records include retained source multiplicity and parallel alternatives. The national budget denotes 1% of total relative source length, not 1% of road counts.
""")
    with Image.open(OUT) as im:assert min(im.info['dpi'])>599;dimensions=list(im.size)
    result={'status':'passed','panels':panel_receipts,'layout':'two_by_two','dpi':600,'figure_size_mm':[190,210],
        'dimensions':dimensions,'common_map_colour_limits_percent':[0,5],'zero_colour':'#2166AC',
        'upper_colour_saturation':True,'colourbar_extension':'max','underlying_values_clipped':False,
        'overlap_colour_limits_percent':[0,100],
        'mapped_quantity':'Mesh Maintained Road Length Share','source_length_conservation_passed':True,
        'selected_length_conservation_passed':True,'overlap_matrix_percent':matrix.to_dict(),
        'road_bearing_meshes':len(sample),'native_mesh_raster':True,'interpolation':'none',
        'colourbar_position':'figure_bottom','publication_revision_authorization':'MILI-D-20261008-008'}
    (EXP/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Saved {OUT.relative_to(ROOT)} (600 dpi, 190 x 210 mm)')


if __name__=='__main__':main()
