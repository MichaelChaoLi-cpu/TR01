"""Spatial Distribution of Residual Pavement Access Burden.

AnaSOP: original mesh costs under the all-unmaintained reference and three fixed
1% selections at 25% finite impairment, weighted by original local population.
The map decomposes the confirmed residual burden into its mesh contributions.
Native meshes, no interpolation; Linear burden display from 0 to 30.
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
SLUG='spatial_distribution_of_residual_pavement_access_burden'
EXP=ROOT/'data/exp/figure-table-generation'/SLUG
OUT=ROOT/'data/results/figures'/f'Figure_{SLUG}.png'


def main():
    EXP.mkdir(parents=True,exist_ok=True)
    file=ROOT/'data/processed/pavement-policy/pavement_mesh_scenarios_preprocessed.parquet'
    values=[];frames=[];accounting=[]
    summary=pd.read_parquet(ROOT/'data/processed/pavement-policy/pavement_group_outcomes_preprocessed.parquet')
    for rule in ['Reference','Traffic','Condition','Access']:
        df=pd.read_parquet(file,filters=[('Pavement Impedance Increment','=',.25),
                           ('Pavement Maintenance Priority','=',rule),('Pavement Budget Share','=',0. if rule=='Reference' else .01)])
        assert len(df)==175612 and df['Mesh Code'].is_unique
        burden=df['Local Total Population'].to_numpy()*df['Pavement Medical Access Cost Increase'].to_numpy()/1000
        assert np.isfinite(burden).all() and (burden>=0).all()
        expected=summary[summary['Pavement Impedance Increment'].eq(.25)&
                         summary['Pavement Maintenance Priority'].eq(rule)&
                         summary['Pavement Budget Share'].eq(0. if rule=='Reference' else .01)&
                         summary['Pavement Population Group'].eq('All Residents')]
        assert len(expected)==1
        expected_total=float(expected['Pavement Residual Access Burden'].iloc[0])
        np.testing.assert_allclose(burden.sum(),expected_total,rtol=1e-12,atol=1e-7)
        accounting.append({'rule':rule,'mesh_burden_sum':float(burden.sum()),
                           'group_residual_burden':expected_total,'absolute_difference':abs(float(burden.sum())-expected_total)})
        frames.append(df);values.append(burden)
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
    vmax=30.0
    norm=Normalize(vmin=0,vmax=vmax,clip=True)
    # Keep the adopted linear 0-30 normalization. Colour anchors give greater
    # distinction to low positive burdens; valid zeros remain pale blue.
    palette_anchors=[(0.,'#E1EEF7'),(.015,'#4F8BBE'),(.08,'#198F7A'),(.30,'#F4D35E'),(1.,'#C73D35')]
    cmap=LinearSegmentedColormap.from_list('low_burden_contrast',palette_anchors,N=1024)
    cmap.set_bad((0,0,0,0))
    sns.set_theme(context='paper',style='white',font='DejaVu Sans')
    fig,axes=plt.subplots(2,2,figsize=(190/25.4,210/25.4))
    fig.subplots_adjust(left=.075,right=.975,bottom=.12,top=.96,wspace=.18,hspace=.15)
    panel_names=['No maintenance','Traffic priority','Condition priority','Access priority']
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
                               'maximum_mesh_burden_resident_reference_km':float(v.max()),
                               'display_maximum_burden':float(v.max()),
                               'meshes_above_colour_maximum':int(np.count_nonzero(v>vmax))})
    cax=fig.add_axes([.12,.062,.80,.018])
    ticks=[0,5,10,15,20,25,30]
    bar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=ticks,extend='max')
    bar.ax.set_xticklabels(['0','5','10','15','20','25','30'])
    bar.ax.tick_params(labelsize=7,pad=2)
    bar.set_label('Residual access burden (resident reference-km)',fontsize=7.5)
    fig.canvas.draw();OUT.parent.mkdir(parents=True,exist_ok=True)
    check_publication_layout(fig, EXP)
    fig.savefig(OUT,dpi=600);fig.savefig(EXP/'preview.png',dpi=150);plt.close(fig)
    OUT.with_suffix('.caption.md').write_text('''# Spatial Distribution of Residual Pavement Access Burden

Panel a shows the all-unmaintained eligible reference; panels b-d show traffic, condition and access priorities at a 1 percent requested pavement budget and 25 percent impedance increment. Each original census mesh contributes B = P times Delta / 1000 to the confirmed residual burden, where P is Local Total Population and Delta is excess generalized shortest medical-access cost in reference metres. B is in resident reference-kilometre equivalents. Mesh contributions sum to the corresponding All Residents residual burden used in the policy outcomes. The original mesh footprint has one mesh per raster pixel with no interpolation. All panels use the same extent, island views and linear display scale from 0 to 30 resident reference-kilometre equivalents; zero is pale blue. Unequally spaced colour anchors strengthen low-positive-burden contrast while the numerical normalization and colourbar remain linear. Burdens above 30 use the maximum colour, indicated by the colourbar upper extension. Colour saturation applies only to the display; population weights, raw burden values and policy outcomes are unchanged. Unobserved or excluded meshes are NoData. The maps show remaining population-weighted access burden, not selected roads, observed travel time or distinct beneficiaries. The full routing network is retained, while only the 70,049 conditional intervention roads receive the adopted impairment/restoration assumptions. Original selected population and physical road entries remain fixed.
''')
    with Image.open(OUT) as im:assert min(im.info['dpi'])>599;dimensions=list(im.size)
    result={'status':'passed','panels':panel_receipts,'layout':'two_by_two','dpi':600,'figure_size_mm':[190,210],
            'dimensions':dimensions,'common_colour_limits_resident_reference_km':[0,vmax],'zero_colour':'#E1EEF7',
            'palette_anchors_normalized':palette_anchors,'presentation_revision':'MILI-D-20261008-005',
            'mapped_quantity':'mesh contribution to confirmed Pavement Residual Access Burden',
            'mapped_units':'resident reference-kilometre equivalents','population_weighted':True,
            'colour_normalization':'linear','raw_burden_at_display_ticks':ticks,
            'accounting':accounting,'group_burden_reconciliation_passed':True,
            'upper_colour_saturation':True,'colourbar_extension':'max','underlying_values_clipped':False,
            'native_mesh_raster':True,'mesh_pixel_readback_exact':True,'island_partition_exact':True,
            'interpolation':'none','publication_revision_authorization':'MILI-D-20261008-008'}
    (EXP/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Saved {OUT.relative_to(ROOT)} (600 dpi, 190 x 210 mm)')


if __name__=='__main__':main()
