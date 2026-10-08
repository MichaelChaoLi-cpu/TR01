"""Actual all-road unmaintained reference panel for the planned national map.

Intermediate panel preview only; no maintenance-policy results are fabricated.
"""
from pathlib import Path
import sys
import os
import hashlib
import json

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tr01-matplotlib')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/analyses'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from PIL import Image
from figure_national_candidate_medical_distance_and_analysis_sample_coverage import (
    mesh_raster_indices, native_raster, basemap_views, style_map, LAND)

OUT = ROOT/'data/exp/figure-table-generation/national-pavement-reference-panel'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    score_dir = ROOT/'data/exp/national-pavement-access-scoring'
    receipt = json.loads((score_dir/'validation.json').read_text())
    assert receipt['status'] == 'passed_all_road_access_scores'
    config = json.loads((score_dir/'configuration.json').read_text())
    file = ROOT/'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet'
    with file.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert digest == config['source_hashes'][str(file.relative_to(ROOT))]
    sample = pd.read_parquet(file)
    cost = np.load(score_dir/'all_road_reference_0.25.npy')
    base = sample['Candidate Medical Network Distance'].to_numpy()
    np.testing.assert_allclose(cost, base*1.25, rtol=1e-12, atol=1e-7)
    extra = cost-base
    assert (extra >= -1e-7).all()
    extra[np.abs(extra) <= 1e-7] = 0.
    burden = sample['Local Total Population'].to_numpy()*extra/1000
    assert len(burden) == 175612 and np.isfinite(burden).all()
    row, col = mesh_raster_indices(sample)
    lat, lon = sample['Latitude'].to_numpy(), sample['Longitude'].to_numpy()
    regions = [
        ('Main islands', lat>=30, (128.2,146.4,29.8,46.1), None),
        ('Nansei islands', (lat<30)&(lon<134), (122.5,131.7,23.75,30.15), [.055,.735,.34,.225]),
        ('Ogasawara', (lat<30)&(lon>=134)&(lon<150), (140.95,142.55,24.45,27.45), [.055,.395,.15,.22]),
        ('Minamitorishima', (lat<30)&(lon>=150), (153.945,154.02,24.265,24.325), [.265,.465,.15,.10]),
    ]
    assert (sum(mask.astype(int) for _,mask,_,_ in regions)==1).all()
    views = basemap_views(regions)
    # Provisional preview scale chosen against the actual reference distribution.
    vmax = 1000.
    norm = Normalize(0, vmax, clip=True)
    cmap = LinearSegmentedColormap.from_list('b_g_y_r', ['#2166AC','#1A9850','#FEE08B','#D73027'])
    cmap.set_bad((0,0,0,0))
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'black'})
    fig, ax = plt.subplots(figsize=(110/25.4,130/25.4))
    fig.subplots_adjust(left=.115,right=.96,bottom=.18,top=.96)
    for name,mask,bounds,position in regions:
        target = ax if position is None else ax.inset_axes(position)
        style_map(target,bounds,inset=position is not None)
        target.tick_params(labelsize=4 if position else 6, pad=1.5 if position else 2.5,
                           length=1.5 if position else 2.5)
        if position is None:
            target.set_xticks([130,135,140,145]); target.set_yticks([30,35,40,45])
        elif name in ['Ogasawara','Minamitorishima']:
            target.set_xticks([(bounds[0]+bounds[1])/2])
            if name=='Minamitorishima': target.set_yticks([(bounds[2]+bounds[3])/2])
        land,outlines,context = views[name]
        for path in context:
            target.add_patch(PathPatch(path,facecolor='#F0F1EF',edgecolor='#AAB4BA',linewidth=.2,zorder=.5))
        for path in land:
            target.add_patch(PathPatch(path,facecolor=LAND,edgecolor='none',zorder=.8))
        if mask.any():
            grid,extent,rr,cc = native_raster(row[mask],col[mask],burden[mask],np.nan,np.float64)
            np.testing.assert_array_equal(grid[rr,cc],burden[mask])
            target.imshow(np.ma.masked_invalid(grid),extent=extent,origin='upper',interpolation='nearest',
                          resample=False,cmap=cmap,norm=norm,aspect=target.get_aspect(),zorder=2)
        target.add_collection(LineCollection(outlines,colors='#73818D',linewidth=.25,zorder=3))
        if position is not None:
            target.text(.02,1.04,name,transform=target.transAxes,fontsize=4.5,va='bottom')
    ax.text(-.10,1.035,'a',transform=ax.transAxes,fontweight='bold',fontsize=11,va='top')
    ax.text(.97,.98,'No maintenance',transform=ax.transAxes,ha='right',va='top',fontsize=7.5)
    cax = fig.add_axes([.16,.095,.76,.018])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,
                       orientation='horizontal',ticks=[0,250,500,750,1000],extend='max')
    bar.ax.tick_params(labelsize=6)
    bar.set_label('Residual burden (resident reference-km)',fontsize=7)
    fig.text(.5,.014,'Actual reference panel; complete policy figure pending',ha='center',fontsize=6)
    full = OUT/'reference_panel_600dpi.png'
    preview = OUT/'preview.png'
    fig.savefig(full,dpi=600); fig.savefig(preview,dpi=150); plt.close(fig)
    with Image.open(full) as im:
        assert min(im.info['dpi']) > 599
        dimensions = list(im.size)
    report = {'status':'passed_actual_reference_panel_preview','scope':'all_road_25_percent_unmaintained_reference',
              'meshes':len(burden),'population':int(sample['Local Total Population'].sum()),
              'burden_sum_resident_reference_km':float(burden.sum()),'zero_meshes':int((burden==0).sum()),
              'provisional_linear_colour_limits':[0,vmax],'saturated_meshes':int((burden>vmax).sum()),
              'upper_saturation_display_only':True,'native_mesh_pixel_roundtrip':True,
              'dpi':600,'dimensions':dimensions,'policy_comparison_complete':False,
              'accepted_manuscript_output':False}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
