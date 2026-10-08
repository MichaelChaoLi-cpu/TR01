"""Render saved representative source-road routes for visual quality review."""
import os,json
from collections import defaultdict
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tr01-matplotlib')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from pyproj import Transformer
from screen_national_network_tiles import OUT


def main():
    routes=json.loads((OUT/'representative_routes.geojson').read_text())['features'];groups=defaultdict(list)
    for f in routes:groups[f['properties']['path_id']].append(f)
    table=pd.read_csv(OUT/'representative_route_checks.csv')
    fig,axes=plt.subplots(4,4,figsize=(16,14))
    for ax,row in zip(axes.ravel(),table.itertuples()):
        pieces=sorted(groups[row.path_id],key=lambda f:f['properties']['segment_order'])
        origin=pieces[0]['geometry']['coordinates'][0];destination=pieces[-1]['geometry']['coordinates'][-1]
        tr=Transformer.from_crs(6668,f'+proj=aeqd +lat_0={origin[1]} +lon_0={origin[0]} +datum=WGS84 +units=m',always_xy=True)
        segments=[]
        for f in pieces:
            c=np.array(f['geometry']['coordinates']);x,y=tr.transform(c[:,0],c[:,1]);segments.append(np.column_stack((x,y))/1000)
        ax.add_collection(LineCollection(segments,colors='#2166ac',linewidths=1.2));ax.autoscale()
        x,y=tr.transform(*destination);ax.plot([0,x/1000],[0,y/1000],color='grey',linestyle=':',linewidth=.7)
        ax.scatter([0,x/1000],[0,y/1000],c=['#d73027','#1b7837'],s=20,zorder=3)
        ax.set_aspect('equal');ax.set_title(row.path_id.replace('ems_base','EMS').replace('medical','Medical')+'\n'+f'{row.network_distance_m/1000:.1f} km; screen ratio {row.detour_screen_ratio:.1f}',fontsize=9)
        ax.set_xlabel('Local easting (km)',fontsize=8);ax.set_ylabel('Local northing (km)',fontsize=8);ax.tick_params(labelsize=7)
    fig.suptitle('National route quality diagnostics: strict endpoint graph\nBlue: source-road route; red: mesh road anchor; green: selected target road anchor; dotted: anchor-to-anchor chord',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'representative_routes.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
