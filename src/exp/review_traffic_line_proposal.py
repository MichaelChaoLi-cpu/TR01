"""Make a bounded curve-overlay review of the proposed matching rule."""
from pathlib import Path
import argparse
import csv
import io
import json
import os
import zipfile
import numpy as np
import pandas as pd
import shapely
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'data/exp/figure-table-generation/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT=Path(__file__).resolve().parents[2]
PROC=ROOT/'data/processed/national-network-rule-validation'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidates',required=True);args=parser.parse_args()
    out=ROOT/args.candidates;cfg=json.loads((out/'configuration.json').read_text());registry=pd.read_parquet(ROOT/cfg['registry_path'])
    frames=[]
    for path in sorted((out/'tiles').glob('*.npz')):
        with np.load(path) as values:frame=pd.DataFrame({k:values[k] for k in values.files})
        frame['Traffic Observation Key']=registry['Traffic Observation Key'].to_numpy()[frame['Registry Group Index'].to_numpy()]
        frames.append(frame)
    candidates=pd.concat(frames,ignore_index=True)
    qualified=candidates[candidates['Technical Proposal Qualified']].drop_duplicates(['Graph Source Edge ID','Traffic Observation Key'])
    multiplicity=qualified.groupby('Graph Source Edge ID')['Traffic Observation Key'].nunique()
    unique=set(multiplicity[multiplicity==1].index);ambiguous=set(multiplicity[multiplicity>1].index)
    cases=[('Tokyo, National Route 1','13100:10020'),('Nikko, mountain national road','9000:14010'),('Nikko, prefectural road','9000:40540')]
    selected=candidates[candidates['Traffic Observation Key'].isin([unit for _,unit in cases])]
    wanted=set(selected['Graph Source Edge ID'])
    offsets=np.load(PROC/'tile_offsets.npy');ordinals=np.load(PROC/'edge_source_ordinals.npy',mmap_mode='r')
    curves={}
    for tile,start,count in offsets:
        edges=[edge for edge in wanted if start<=edge<start+count]
        if not edges:continue
        lookup={int(ordinals[edge]):edge for edge in edges}
        with zipfile.ZipFile(ROOT/f'data/raw/roads/n13_2024_geojson/N13-24_{int(tile):04d}_GEOJSON.zip') as archive:
            member=next(n for n in archive.namelist() if n.endswith('.geojson'));ordinal=0
            with io.TextIOWrapper(archive.open(member),encoding='utf-8-sig') as handle:
                for line in handle:
                    if not line.lstrip().startswith('{ "type": "Feature",'):continue
                    ordinal+=1
                    if ordinal in lookup:curves[lookup[ordinal]]=np.asarray(json.loads(line.strip().rstrip(','))['geometry']['coordinates'])
    assert set(curves)==wanted
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    for ax,(title,unit) in zip(axes,cases):
        group=registry[registry['Traffic Observation Key'].eq(unit)]
        traffic=shapely.union_all(shapely.from_wkb(group['Geometry WKB'].to_numpy()))
        local=selected[selected['Traffic Observation Key'].eq(unit)]
        eligible=set(local[local['Technical Proposal Qualified']]['Graph Source Edge ID'])
        all_ids=set(local['Graph Source Edge ID'])
        ax.add_collection(LineCollection([curves[e] for e in all_ids],colors='#b7b7b7',linewidths=.8))
        for part in shapely.get_parts(traffic):
            xy=shapely.get_coordinates(part);ax.plot(xy[:,0],xy[:,1],color='#7757a5',linewidth=2.5,alpha=.65)
        ax.add_collection(LineCollection([curves[e] for e in eligible if e in unique],colors='#d76524',linewidths=1.4))
        ax.add_collection(LineCollection([curves[e] for e in eligible if e in ambiguous],colors='black',linewidths=2))
        lo_x,lo_y,hi_x,hi_y=shapely.bounds(traffic);width=max(hi_x-lo_x,.0003);height=max(hi_y-lo_y,.0003)
        ax.set_xlim(lo_x-width*.06,hi_x+width*.06);ax.set_ylim(lo_y-height*.06,hi_y+height*.06)
        ax.set_aspect(1/np.cos(np.deg2rad((lo_y+hi_y)/2)),adjustable='datalim')
        ax.set_box_aspect(1);ax.tick_params(labelsize=7)
        from matplotlib.ticker import FormatStrFormatter,MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_title(title,fontsize=10)
        ax.set_xlabel('Longitude',fontsize=8);ax.set_ylabel('Latitude',fontsize=8)
        ax.text(.02,.02,f'Unique proposals: {len(eligible&unique)}\nAmbiguous: {len(eligible&ambiguous)}',
                transform=ax.transAxes,fontsize=8,color='black',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
        ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color='#7757a5',lw=2.5,label='Official traffic line'),
                        Line2D([],[],color='#d76524',lw=1.5,label='Unique proposed road'),
                        Line2D([],[],color='black',lw=2,label='Ambiguous road'),
                        Line2D([],[],color='#b7b7b7',lw=1,label='Other same-class candidate')],
               loc='outside upper center',ncol=4,fontsize=8)
    fig.savefig(out/'bounded_rule_review.png',dpi=150,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    receipt={'scope':'Three purposive observation units; not national accuracy validation',
             'qualified_road_records':len(multiplicity),'unique_proposed_road_records':len(unique),
             'ambiguous_road_records':len(ambiguous),'accepted_correspondences':0,
             'cases':[{'label':label,'observation_unit':unit} for label,unit in cases]}
    (out/'bounded_review_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
