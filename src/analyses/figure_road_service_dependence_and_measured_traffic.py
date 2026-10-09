#!/usr/bin/env python3
"""Road Service Dependence and Measured Traffic.

AnaSOP Sections 5-7: adopted conditional single-source-road loss, unchanged
baseline population, observed bidirectional daytime traffic and strict matching.
Two panels: three-variable binned scatter and full-source-curve burden map.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tr01-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_figure_checks import check_publication_layout
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm, FuncNorm, LinearSegmentedColormap
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from PIL import Image
from figure_national_candidate_medical_distance_and_analysis_sample_coverage import (
    basemap_views, style_map, LAND, BOUNDARY, BASEMAP_DIR)

ROOT = Path(__file__).resolve().parents[2]
TITLE = 'Road Service Dependence and Measured Traffic'
SLUG = 'road_service_dependence_and_measured_traffic'
EXP = ROOT / 'data/exp/figure-table-generation' / SLUG
OUT = ROOT / f'data/results/figures/Figure_{SLUG}.png'
SOURCE = ROOT / 'data/processed/road-traffic-comparison/national_road_traffic_comparison_preprocessed.parquet'
CROSS = ROOT / 'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet'
GEOMETRY = EXP / 'source_road_display_geometry.parquet'
PREP = ROOT / 'data/exp/data-preprocessing/road-traffic-comparison'
TRAFFIC = 'Observed Traffic Volume'
LOST = 'Lost Medical Path Population'
BURDEN = 'Additional Medical Distance Burden'
EXPORT_DPI = 600
FIGURE_SIZE_MM = (190, 180)
BOUNDARY_LINEWIDTH = .22
PANEL_B_SCALE = .8
PANEL_B_FONT_SCALE = 1.0
PANEL_B_ANCHOR = (.965, .17)
PANEL_B_HORIZONTAL_SHIFT = -.05
MAIN_MAP_BOUNDS = (128.2, 151.5, 29.0, 46.7)
MAP_COLOURBAR_POSITION = (.16, .09, .72, .012)
MAP_COLOURMAP = 'blue_green_yellow_red'
MAP_COLOURS = ['#2166AC', '#1A9850', '#FFFF00', '#D73027']
MAP_DISPLAY_VMAX = 5000.0


def panel_b_position(position):
    """Scale the inset and its legend together around their lower-right anchor."""
    x, y, width, height = position
    return (PANEL_B_ANCHOR[0]+PANEL_B_SCALE*(x-PANEL_B_ANCHOR[0])+PANEL_B_HORIZONTAL_SHIFT,
            PANEL_B_ANCHOR[1]+PANEL_B_SCALE*(y-PANEL_B_ANCHOR[1]),
            width*PANEL_B_SCALE,height*PANEL_B_SCALE)


# Invisible layout envelope, used only for layout/occlusion accounting.
HEATMAP_CARD = panel_b_position((.53, .17, .435, .46))


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def label_bar(bar, ticks, label, *, fontsize=8, ticksize=7.2, scale=1, labelpad=5):
    bar.set_ticks(ticks)
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}' if x >= 1 else f'{x:g}'))
    if scale == 1:
        bar.ax.tick_params(labelsize=ticksize, which='minor', length=0)
    else:
        # Previous major colourbar labels use the theme's default (8.8 pt).
        bar.ax.tick_params(labelsize=plt.rcParams['xtick.labelsize']*scale,
                           pad=plt.rcParams['xtick.major.pad']*scale,
                           length=plt.rcParams['xtick.major.size']*scale,
                           width=plt.rcParams['xtick.major.width']*scale)
        bar.ax.tick_params(labelsize=ticksize,which='minor', length=0)
        for spine in bar.ax.spines.values():spine.set_linewidth(plt.rcParams['axes.linewidth']*scale)
    bar.set_label(label, fontsize=fontsize, labelpad=labelpad*scale, color='black')


def main():
    assert json.loads((PREP / 'validation.json').read_text())['status'] == 'passed'
    geometry_receipt = json.loads((EXP / 'geometry_validation.json').read_text())
    assert geometry_receipt['status'] == 'passed' and geometry_receipt['full_source_curves']
    sources = [SOURCE, CROSS, GEOMETRY, BOUNDARY]
    sources += [BASEMAP_DIR / f'neighbour_context/ne_110m_admin_0_countries.{ext}' for ext in ['shp','shx','dbf','prj']]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    assert hashes[str(GEOMETRY.relative_to(ROOT))] == geometry_receipt['output_sha256']
    fields = ['Graph Source Edge ID', TRAFFIC, LOST, BURDEN, 'Traffic Observation Key',
              'Baseline Source Road Length', 'Closed Origin Entry Population']
    data = pd.read_parquet(SOURCE, columns=fields, filters=[('Traffic Match Status','==','matched')])
    data = data.sort_values('Graph Source Edge ID').reset_index(drop=True)
    geometry = pd.read_parquet(GEOMETRY)
    np.testing.assert_array_equal(data['Graph Source Edge ID'],geometry['Graph Source Edge ID'])
    cross = pd.read_parquet(CROSS,columns=['Graph Source Edge ID',TRAFFIC,'Traffic Observation Key'])
    cross = cross.sort_values('Graph Source Edge ID').reset_index(drop=True)
    np.testing.assert_array_equal(data[TRAFFIC],cross[TRAFFIC])
    np.testing.assert_array_equal(data['Traffic Observation Key'].astype(str),cross['Traffic Observation Key'].astype(str))
    n = len(data); assert n == 1183367 and data['Graph Source Edge ID'].is_unique
    units = int(data['Traffic Observation Key'].nunique()); assert units == 25566
    traffic = data[TRAFFIC].to_numpy(dtype=float)
    lost = data[LOST].to_numpy(dtype=float)
    burden = data[BURDEN].to_numpy(dtype=float)
    assert np.isfinite(np.c_[traffic,lost,burden]).all()
    assert (traffic>0).all() and (lost>=0).all() and (burden>=0).all()
    assert (lost>=data['Closed Origin Entry Population']).all()
    x, y = np.log10(traffic), np.log10(1+burden)
    np.testing.assert_allclose(10**x,traffic,rtol=1e-13)
    np.testing.assert_allclose(10**y-1,burden,rtol=1e-10,atol=1e-10)
    sns.set_theme(context='paper',style='ticks',font='DejaVu Sans')
    plt.rcParams.update({'font.size':8.5,'figure.facecolor':'white','text.color':'black',
                         'axes.labelcolor':'black','path.simplify':False})
    fig = plt.figure(figsize=tuple(v/25.4 for v in FIGURE_SIZE_MM), dpi=EXPORT_DPI)
    # Internal a is the binned display; published panel letters are map a/display b.
    b = fig.add_axes([.07,.14,.91,.82],zorder=1)
    card = Rectangle(HEATMAP_CARD[:2],*HEATMAP_CARD[2:],transform=fig.transFigure,
                     facecolor='none',edgecolor='none',linewidth=0,zorder=8)
    fig.add_artist(card)
    a = fig.add_axes(panel_b_position((.65,.333,.29,.257)),zorder=10)
    extent = (float(x.min()-.04),float(x.max()+.04),0,float(y.max()*1.015))
    count_artist = a.hexbin(x,y,gridsize=(65,40),extent=extent,mincnt=1,linewidths=0)
    counts = np.asarray(count_artist.get_array())
    centers = np.asarray(count_artist.get_offsets())
    assert int(counts.sum()) == n
    count_artist.remove()
    colour = LinearSegmentedColormap.from_list('path_loss',['#EFF2F3','#FFF0A8','#FDAE61','#E34A33','#7F0000'])
    scatter = a.hexbin(x,y,C=lost,reduce_C_function=lambda values: np.mean(values) if len(values) else np.nan,gridsize=(65,40),extent=extent,
                       mincnt=0,cmap=colour,linewidths=0,rasterized=True)
    means = np.asarray(scatter.get_array())
    np.testing.assert_allclose(scatter.get_offsets(),centers,rtol=0,atol=1e-12)
    assert len(means)==len(counts) and np.isfinite(means).all() and (means>=0).all()
    np.testing.assert_allclose(means @ counts,lost.sum(),rtol=1e-11,atol=1e-7)
    norm_a = SymLogNorm(linthresh=.1,linscale=1,vmin=0,vmax=float(means.max()))
    scatter.set_norm(norm_a)
    a.set_xlim(extent[0],extent[1]);a.set_ylim(-.08,extent[3]+.03)
    xticks = np.array([2,10,100,1000,10000,100000],dtype=float)
    yticks = np.array([0,10,100,1000,10000,50000],dtype=float)
    traffic_tick_labels = ['2'] + [rf'$10^{{{int(np.log10(v))}}}$' for v in xticks[1:]]
    a.set_xticks(np.log10(xticks),traffic_tick_labels)
    a.set_yticks(np.log10(1+yticks),[f'{v:,.0f}' for v in yticks])
    a.set_xlabel('Observed traffic volume\n(vehicles / daytime 12 h; log scale)',labelpad=3*PANEL_B_SCALE,fontsize=7.2*PANEL_B_FONT_SCALE)
    a.set_ylabel('Additional medical distance burden\n(person-km; log(1+x) scale)',labelpad=5*PANEL_B_SCALE,fontsize=7.2*PANEL_B_FONT_SCALE)
    a.grid(True,color='#DCE2E6',linewidth=.6*PANEL_B_SCALE,zorder=0)
    a.set_axisbelow(True)
    a.tick_params(labelsize=6.8*PANEL_B_FONT_SCALE,pad=3.5*PANEL_B_SCALE,direction='out',length=2.5*PANEL_B_SCALE,width=.7*PANEL_B_SCALE)
    for spine in a.spines.values():spine.set_linewidth(.7*PANEL_B_SCALE)
    note = f'Roads: {n:,}\nUnits: {units:,}'
    note_artist=a.text(.02,.98,note,transform=a.transAxes,ha='left',va='top',fontsize=6.8*PANEL_B_FONT_SCALE,color='black',
           bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2*PANEL_B_SCALE})
    longitude = geometry['Display Longitude'].to_numpy()
    latitude = geometry['Display Latitude'].to_numpy()
    regions = [
        ('Main islands',latitude>=30,MAIN_MAP_BOUNDS,None),
        ('Nansei islands',(latitude<30)&(longitude<134),(122.5,131.7,23.75,30.15),[.05,.69,.32,.235]),
        ('Ogasawara',(latitude<30)&(longitude>=134)&(longitude<150),(140.95,142.55,24.45,27.45),[.06,.40,.12,.19]),
        ('Minamitorishima',(latitude<30)&(longitude>=150),(153.945,154.02,24.265,24.325),[.35,.51,.165,.125])]
    coverage = sum(mask.astype(np.int8) for _,mask,_,_ in regions)
    np.testing.assert_array_equal(coverage,np.ones(n,dtype=np.int8))
    views = basemap_views(regions)
    curves = shapely.from_wkb(geometry['Geometry WKB'].to_numpy())
    assert (shapely.get_type_id(curves)==1).all()
    _,coordinates,offsets = shapely.to_ragged_array(curves)
    ends = offsets[0]
    norm_b = FuncNorm((lambda value: np.log10(1+value),lambda value: 10**value-1),
                      vmin=0,vmax=MAP_DISPLAY_VMAX,clip=False)
    np.testing.assert_allclose(norm_b.inverse(norm_b(burden)),burden,rtol=1e-10,atol=1e-10)
    assert float(norm_b(0))==0
    cmap_b = LinearSegmentedColormap.from_list(MAP_COLOURMAP,MAP_COLOURS)
    cmap_b.set_over(MAP_COLOURS[-1])
    above_display_max=burden>MAP_DISPLAY_VMAX
    assert np.all(cmap_b(norm_b(burden[above_display_max]))==cmap_b(1.0))
    map_records=[];children=[];inset_names=[]
    for name,mask,bounds,position in regions:
        if name=='Minamitorishima' and not mask.any():continue
        target = b if position is None else b.inset_axes(position)
        if position is not None:children.append(target)
        style_map(target,bounds,inset=position is not None)
        target.tick_params(labelsize=6 if position is not None else 7.5, pad=2.0 if position is not None else 3.5)
        if position is None:
            target.set_xticks([130,134,138,142,146,150])
            target.tick_params(top=True,labeltop=True)
            for tick_label in target.get_xticklabels():tick_label.set_fontweight('bold')
            target.grid(True,color='#97A6B1',linewidth=.5,alpha=.7,linestyle=(0,(4,3)),zorder=4)
        if name=='Ogasawara':
            target.set_xticks([142.0])
        land,outlines,context = views[name]
        for path in context:target.add_patch(PathPatch(path,facecolor='#F0F1EF',edgecolor='#AAB4BA',linewidth=.35,zorder=.5))
        for path in land:target.add_patch(PathPatch(path,facecolor=LAND,edgecolor='none',zorder=.8))
        zero = np.flatnonzero(mask&(burden==0))
        positive = np.flatnonzero(mask&(burden>0))
        positive = positive[np.argsort(burden[positive],kind='stable')]
        for idx,colours,width,zorder in [(zero,cmap_b(norm_b(0)),.22,1.2),
                    (positive,cmap_b(norm_b(burden[positive])),.42,2.5)]:
            segments=[coordinates[ends[i]:ends[i+1]] for i in idx]
            target.add_collection(LineCollection(segments,colors=colours,linewidth=width,rasterized=True,zorder=zorder))
        target.add_collection(LineCollection(outlines,colors='#84929C',linewidth=BOUNDARY_LINEWIDTH,zorder=3.5))
        if position is not None:
            inset_names.append(target.text(.5,1.035,name,transform=target.transAxes,fontsize=6.5,ha='center',va='bottom',color='black'))
        map_records.append({'view':name,'source_roads':int(mask.sum()),'positive_burden_roads':len(positive),
                            'zero_burden_roads':len(zero),'bounds':list(bounds)})
    assert sum(r['source_roads'] for r in map_records)==n
    for label,ax in [('a',b),('b',a)]:
        ax.text(-.045 if ax is b else -.34,1.005 if ax is b else 1.07,label,transform=ax.transAxes,fontsize=10 if ax is b else 10*PANEL_B_SCALE,fontweight='bold',ha='left',va='bottom',color='black')
    fig.canvas.draw()
    for ax,mappable,ticks,label in [
        (a,scatter,[v for v in [0,.1,1,10,100,1000,10000] if v<=means.max()],
         'Mean lost medical path population\n(residents per road scenario within a hexagon)'),
        (b,plt.cm.ScalarMappable(norm=norm_b,cmap=cmap_b),[0,1,10,100,1000,MAP_DISPLAY_VMAX],
         'Additional medical distance burden\n(person-km per road-removal scenario; log10(1+x) colour scale)')]:
        frame=ax.get_position()
        position=panel_b_position((.65,.232,.29,.012)) if ax is a else MAP_COLOURBAR_POSITION
        cax=fig.add_axes(position,zorder=10)
        label_bar(fig.colorbar(mappable,cax=cax,orientation='horizontal',
                               extend='max' if ax is b else 'neither',
                               extendfrac=.025 if ax is b else None),ticks,label,
                  fontsize=6.8*PANEL_B_FONT_SCALE if ax is a else 7.5,
                  ticksize=6.8*PANEL_B_FONT_SCALE if ax is a else 7,
                  scale=PANEL_B_FONT_SCALE if ax is a else 1, labelpad=3 if ax is a else 5)
        if ax is a:
            cax.set_facecolor('none')
            colourbar_tick_font=cax.get_xticklabels()[0].get_fontsize()
            panel_b_colourbar_axis=cax
    pd.DataFrame({'Display Log Traffic':centers[:,0],'Display Log One Plus Burden':centers[:,1],
                  'Road Scenario Count':counts.astype(int),'Mean Lost Medical Path Population':means}).to_csv(EXP/'display_bins.csv',index=False)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    card_frame=card.get_window_extent(renderer)
    note_frame=note_artist.get_bbox_patch().get_window_extent(renderer)
    hex_centres_on_canvas=a.transData.transform(centers)
    note_hidden_cells=int(np.count_nonzero(
        (hex_centres_on_canvas[:,0]>=note_frame.x0)&(hex_centres_on_canvas[:,0]<=note_frame.x1)&
        (hex_centres_on_canvas[:,1]>=note_frame.y0)&(hex_centres_on_canvas[:,1]<=note_frame.y1)))
    vertices=scatter.get_paths()[0].vertices
    polygons_on_canvas=a.transData.transform((centers[:,None,:]+vertices[None,:,:]).reshape(-1,2))
    polygons_on_canvas=polygons_on_canvas.reshape(len(centers),len(vertices),2)
    note_intersecting_cells=int(np.count_nonzero(shapely.intersects(
        shapely.polygons(polygons_on_canvas),shapely.box(note_frame.x0,note_frame.y0,note_frame.x1,note_frame.y1))))
    heat_card_inside_map=bool(b.get_window_extent(renderer).contains(card_frame.x0,card_frame.y0)
                              and b.get_window_extent(renderer).contains(card_frame.x1,card_frame.y1))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    tick_labels=[t.get_window_extent(renderer) for ax in [a,b,*children]
                 for t in [*ax.get_xticklabels(),*ax.get_yticklabels()] if t.get_visible()]
    overlaps=sum(first.overlaps(second) for i,first in enumerate(tick_labels) for second in tick_labels[i+1:])
    inset_name_overlaps=sum(name.get_window_extent(renderer).overlaps(tick)
                            for name in inset_names for tick in tick_labels)
    from matplotlib.text import Text
    clipped_text=[]
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():continue
        box=artist.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            clipped_text.append(artist.get_text())
    inset_hidden=0;inset_hidden_curves=0
    for child in children:
        frame=child.get_window_extent(renderer)
        lower,upper=b.transData.inverted().transform([[frame.x0,frame.y0],[frame.x1,frame.y1]])
        inset_hidden += int(((latitude>=30)&(longitude>=lower[0])&(longitude<=upper[0])&
                             (latitude>=lower[1])&(latitude<=upper[1])).sum())
        inset_hidden_curves+=int(np.count_nonzero(shapely.intersects(
            curves[latitude>=30],shapely.box(lower[0],lower[1],upper[0],upper[1]))))
    lower,upper=b.transData.inverted().transform(fig.transFigure.transform([
        HEATMAP_CARD[:2],(HEATMAP_CARD[0]+HEATMAP_CARD[2],HEATMAP_CARD[1]+HEATMAP_CARD[3])]))
    card_hidden=int(((latitude>=30)&(longitude>=lower[0])&(longitude<=upper[0])&
                     (latitude>=lower[1])&(latitude<=upper[1])).sum())
    card_hidden_curves=int(np.count_nonzero(shapely.intersects(
        curves[latitude>=30],shapely.box(lower[0],lower[1],upper[0],upper[1]))))
    main_curve_bounds=shapely.bounds(curves[latitude>=30])
    xmin,xmax,ymin,ymax=MAIN_MAP_BOUNDS
    clipped_main_curves=int(np.count_nonzero((main_curve_bounds[:,0]<xmin)|(main_curve_bounds[:,2]>xmax)|
                                            (main_curve_bounds[:,1]<ymin)|(main_curve_bounds[:,3]>ymax)))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    check_publication_layout(fig, EXP)
    traffic_tick_boxes = [tick.get_window_extent(fig.canvas.get_renderer())
                          for tick in a.get_xticklabels() if tick.get_visible()]
    traffic_tick_gap_mm = min(second.x0-first.x1
                              for first, second in zip(traffic_tick_boxes, traffic_tick_boxes[1:]))/fig.dpi*25.4
    assert traffic_tick_gap_mm >= .8, traffic_tick_gap_mm
    label_bar_gap_mm=(a.xaxis.label.get_window_extent(fig.canvas.get_renderer()).y0-panel_b_colourbar_axis.get_window_extent(fig.canvas.get_renderer()).y1)/fig.dpi*25.4
    assert label_bar_gap_mm >= 1.0, label_bar_gap_mm
    fig.savefig(OUT,dpi=EXPORT_DPI,facecolor='white')
    fig.savefig(EXP/'preview.png',dpi=150,facecolor='white')
    plt.close(fig)
    for path,expected in hashes.items():assert digest(ROOT/path)==expected
    length=data['Baseline Source Road Length'].to_numpy()
    receipt={'title':TITLE,'placement':'main_manuscript','status':'generated_awaiting_human_review',
       'matching_authorization':'MILI-D-20261007-015','scenario_authorization':'MILI-D-20261007-013',
       'traffic_authorization':'MILI-D-20261007-014','subpanels':2,'comparison_source_roads':n,
       'shared_traffic_observation_units':units,'all_source_road_rows_retained':21415800,
       'comparison_coverage_percent_of_source_road_records':n/21415800*100,
       'positive_path_loss_scenarios':int((lost>0).sum()),'zero_extra_distance_scenarios':int((burden==0).sum()),
       'closed_entry_only_path_loss_scenarios':int(((lost>0)&(lost==data['Closed Origin Entry Population'])).sum()),
       'path_loss_exceeding_closed_entry_scenarios':int((lost>data['Closed Origin Entry Population']).sum()),
       'panel_b_display':'Log10 traffic vs log10(1 + person-km burden); colour is within-hexagon mean path-loss population',
       'panel_b_count_conservation':int(counts.sum()),'panel_b_loss_colour_conservation':'passed',
       'occupied_hexagons':len(counts),'maximum_hexagon_mean_loss':float(means.max()),
       'panel_a_display':'Original matched source-road curves; colour is scenario additional-distance burden',
       'map_view_accounting':map_records,'full_source_curves':True,'tick_label_overlap_count':int(overlaps),
       'inset_name_tick_overlap_count':int(inset_name_overlaps),'text_outside_canvas':clipped_text,
       'matched_main_road_midpoints_hidden_by_insets':inset_hidden,'note_colour':'black',
       'matched_main_road_midpoints_hidden_by_heatmap_card':card_hidden,
       'matched_main_road_curves_intersecting_inset_frames':inset_hidden_curves,
       'matched_main_road_curves_intersecting_heatmap_card':card_hidden_curves,
       'layout':'map-dominant canvas with lower-right binned display',
       'administrative_boundary_linewidth_points':BOUNDARY_LINEWIDTH,'display_note_inside_frame':True,
       'layout_revision_authorization':'MILI-D-20261007-017',
       'visual_refinement':'panel b dimensions retain the adopted 20-percent reduction; publication-size text enlarged independently; transparent outside its plotting frame',
       'panel_b_revision_authorization':'MILI-D-20261007-018',
       'publication_readability_revision_authorization':'MILI-D-20261008-008',
       'position_colourbar_revision_authorization':'MILI-D-20261007-019',
       'map_colourmap':MAP_COLOURMAP,'map_palette_revision_authorization':'MILI-D-20261007-024',
       'map_colour_scale':'log10(1 + burden)','map_colour_limits':[0,MAP_DISPLAY_VMAX],
       'map_colourbar_extend':'max','map_source_roads_above_display_max':int(above_display_max.sum()),
       'map_above_max_colour':matplotlib.colors.to_hex(cmap_b(1.0)),
       'map_above_max_values_retained':True,
       'map_colour_roundtrip':'passed','map_colourbar_ticks_in_original_units':True,
       'map_colour_stops':MAP_COLOURS,'map_zero_colour':matplotlib.colors.to_hex(cmap_b(norm_b(0))),
       'map_zero_uses_colourbar':True,'grey_zero_note_present':False,
       'panel_b_horizontal_shift_mm':PANEL_B_HORIZONTAL_SHIFT*FIGURE_SIZE_MM[0],
       'map_colourbar_width_mm':MAP_COLOURBAR_POSITION[2]*FIGURE_SIZE_MM[0],
       'main_road_curves_clipped_by_view_bounds':clipped_main_curves,
       'panel_b_xlabel_colourbar_gap_mm':label_bar_gap_mm,
       'panel_b_x_tick_format':'Powers of ten in original traffic units; lower reference remains 2',
       'panel_b_minimum_x_tick_label_gap_mm':traffic_tick_gap_mm,
       'panel_b_dimension_scale':PANEL_B_SCALE,'panel_b_font_scale':PANEL_B_FONT_SCALE,
       'panel_b_outer_background':'transparent','panel_b_outer_frame_visible':False,
       'panel_b_plot_position':list(a.get_position().bounds),
       'panel_b_plot_size_mm':[a.get_position().width*FIGURE_SIZE_MM[0],a.get_position().height*FIGURE_SIZE_MM[1]],
       'panel_b_font_sizes_points':{'axis_titles':a.xaxis.label.get_fontsize(),
          'axis_tick_labels':a.get_xticklabels()[0].get_fontsize(),'note':note_artist.get_fontsize(),
          'panel_letter':a.texts[1].get_fontsize(),'colourbar_title':6.8*PANEL_B_FONT_SCALE,
          'colourbar_major_tick_labels':colourbar_tick_font},
       'heatmap_card_inside_map_frame':heat_card_inside_map,
       'occupied_hexagon_centres_covered_by_note':note_hidden_cells,
       'occupied_hexagons_intersecting_note_box':note_intersecting_cells,
       'loss_population_range':[float(lost.min()),float(lost.max())],
       'additional_distance_burden_range':[float(burden.min()),float(burden.max())],
       'source_record_length_range_metres':[float(length.min()),float(length.max())],
       'source_record_length_median_metres':float(np.median(length)),
       'values_imputed_or_trimmed':False,'new_baseline_sample_exclusions':False,'inferential_fit':None,
       'scientific_input_hashes_verified_unchanged':hashes,'output':str(OUT.relative_to(ROOT)),'dpi':EXPORT_DPI,
       'figure_size_mm':list(FIGURE_SIZE_MM),'export_revision_authorization':'MILI-D-20261007-016',
       'image_dimensions':list(Image.open(OUT).size)}
    (EXP/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    OUT.with_suffix('.caption.md').write_text(
       f'{TITLE}. (a) Additional Medical Distance Burden assigned to the full original curve of the road removed in each scenario. A blue-green-yellow-red colour scale uses log10(1 + burden) coordinates with a display maximum of {MAP_DISPLAY_VMAX:,.0f} person-kilometres. Values above this threshold share the upper-end red colour, indicated by the colourbar extension; all original values remain intact. Colourbar ticks retain original person-kilometre units; exact zero values use the same blue zero point as the colourbar, without a separate colour category. This is a display-only transformation and saturation, with no change to stored burdens. The map locates the closed-road scenario, not the homes of affected residents or a physical disruption footprint. '
       'Prefecture boundaries, pale-blue water, labelled graticules and island views provide geographic context. '
       '(b) Observed bidirectional daytime 12-hour traffic versus Additional Medical Distance Burden, with hexagon colours showing mean Lost Medical Path Population per source-road scenario in each display cell. '
       f'All {n:,} uniquely matched source roads are represented once; the count check includes every zero. Traffic uses a logarithmic horizontal axis, and burden uses log10(1 + burden) coordinates, with original-unit tick labels. '
       f'There are {int((burden==0).sum()):,} exact zero additional-distance scenarios; none is omitted or assigned a positive plotting offset. '
       'Display-cell means include zero-loss scenarios and may be fractional; they are neither fractional census residents nor totals of distinct affected residents. '
       f'The road records share {units:,} measured traffic observation units, so repeated source-road fragments are not independent traffic observations. No regression or spatial-independence inference is performed. '
       'The comparison covers all 47 prefectures but only 5.53 percent of graph source-road records, predominantly national and prefectural roads plus expressways; municipal-road coverage is sparse. All unmatched and ambiguous traffic values remain missing in the full national input archive. '
       'Each scenario removes one source record in both directions, disables fixed entries requiring it, retains other source roads including parallel alternatives, and preserves the original 175,612 meshes and 125,874,788 residents. '
       'Lost Medical Path Population includes fixed-origin entry closure as well as any further downstream loss; it must not automatically be interpreted as geographic isolation. Disconnected residents are represented through path-loss population and receive no arbitrary finite distance penalty in the additional-distance burden. '
       f'Within this matched sample, {receipt["positive_path_loss_scenarios"]:,} scenarios have positive path loss: {receipt["closed_entry_only_path_loss_scenarios"]:,} have closed-entry loss only and {receipt["path_loss_exceeding_closed_entry_scenarios"]:,} exceed that entry population. '
       f'Source-record lengths have median {np.median(length):.1f} m and range {length.min():.2f} to {length.max():.1f} m; segmentation is not an equal-cost maintenance unit. '
       'The source population and medical registrations refer to 2020, measured traffic to 2021 and candidate roads to 2024. The strict geometric matching rule is adopted sample construction, not an independent national matching-accuracy estimate. '
       'The figure is a descriptive conditional comparison of service dependence and measured traffic, not observed maintenance effectiveness, failure risk, repair benefit or an allocation-policy ranking. Sources: e-Stat census, MLIT National Land Numerical Information, and the 2021 National Road and Street Traffic Conditions Survey.\n')
    print(json.dumps({k:receipt[k] for k in ['output','comparison_source_roads','shared_traffic_observation_units',
                      'positive_path_loss_scenarios','closed_entry_only_path_loss_scenarios','path_loss_exceeding_closed_entry_scenarios',
                      'tick_label_overlap_count','matched_main_road_midpoints_hidden_by_insets','image_dimensions']},indent=2),flush=True)


if __name__=='__main__':
    main()
