"""Skill-guided descriptive audit; no ageing variable or processed data is selected."""
import csv,hashlib,importlib.util,json,os,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tr01-matplotlib')
from build_vehicle_network_pilot import ROOT,mesh_center
OUT=ROOT/'data/exp/data-briefing/population-sample-audit'
ENTRY=ROOT/'data/exp/national-population-entry'
spec=importlib.util.spec_from_file_location('briefing_skill',ROOT/'.codex/skills/data-briefing/scripts/create_data_briefing.py');skill=importlib.util.module_from_spec(spec);spec.loader.exec_module(skill)


def load(table):
    frames=[];sources=[]
    manifest=pd.read_csv(ROOT/'data/exp/data-acquisition/population_2020_download_manifest.csv',dtype=str).set_index('filename')
    for p in sorted((ROOT/f'data/raw/population/census_2020_1km_jgd2011/{table}').glob('*.zip')):
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==manifest.loc[p.name,'sha256']
        with zipfile.ZipFile(p) as z:
            name=z.namelist()[0];frame=pd.read_csv(z.open(name),encoding='cp932',skiprows=[1],dtype=str,keep_default_na=False)
            sources.append(dict(table=table,path=str(p.relative_to(ROOT)),member=name,rows=len(frame),sha256=digest));frames.append(frame)
    return pd.concat(frames,ignore_index=True).set_index('KEY_CODE',drop=False),sources


def main():
    OUT.mkdir(parents=True,exist_ok=True);figures=OUT/'figures';figures.mkdir(exist_ok=True)
    a,s1=load('T001140');b,s2=load('T001188');b=b.loc[a.index]
    assert len(a)==176964 and a.index.is_unique and b.index.is_unique
    metadata=['KEY_CODE','HTKSYORI','HTKSAKI','GASSAN']
    pd.testing.assert_frame_equal(a[metadata],b[metadata])
    assert a.T001140001.equals(b.T001188001)
    assert set(a.HTKSYORI)=={'0','1','2'}
    group=a.KEY_CODE.where(a.HTKSYORI.ne('2'),a.HTKSAKI)
    links=[]
    for r in a[a.HTKSYORI.eq('2')].itertuples(index=False):
        assert r.HTKSAKI in a.index and a.at[r.HTKSAKI,'HTKSYORI']=='1'
        listed=a.at[r.HTKSAKI,'GASSAN'].split(';');assert r.KEY_CODE in listed
        links.append(dict(KEY_CODE=r.KEY_CODE,HTKSYORI=r.HTKSYORI,HTKSAKI=r.HTKSAKI,recipient_GASSAN=a.at[r.HTKSAKI,'GASSAN']))
    for r in a[a.HTKSYORI.eq('1')].itertuples(index=False):
        listed=r.GASSAN.split(';');assert len(listed)==len(set(listed))
        for key in listed:assert key in a.index and a.at[key,'HTKSYORI']=='2' and a.at[key,'HTKSAKI']==r.KEY_CODE
    assert a.loc[a.HTKSYORI.eq('0'),['HTKSAKI','GASSAN']].eq('').all().all()
    population=pd.to_numeric(a.T001140001,errors='raise');older=pd.to_numeric(a.T001140019,errors='coerce')
    assert a.loc[a.HTKSYORI.eq('2'),'T001140019'].eq('*').all() and older[a.HTKSYORI.ne('2')].notna().all()
    bands65=[f'T001188{i:03}' for i in range(43,64,3)]
    bands75=[f'T001188{i:03}' for i in range(49,64,3)]
    knowncols=[f'T001188{i:03}' for i in range(4,64,3)]
    numeric=b[knowncols].apply(pd.to_numeric,errors='coerce');available=a.HTKSYORI.ne('2')
    assert np.array_equal(numeric.loc[available,bands65].sum(axis=1),older[available])
    assert np.array_equal(numeric.loc[available,bands75].sum(axis=1),pd.to_numeric(a.loc[available,'T001140022']))
    assert numeric.loc[available].notna().all().all()
    unit_population=population.groupby(group).sum();members=group.groupby(group).size()
    units=a.loc[unit_population.index]
    published65=pd.to_numeric(units.T001140019);known_age=numeric.sum(axis=1,min_count=len(knowncols)).loc[unit_population.index]
    assert (known_age<=unit_population).all() and (published65<=known_age).all()
    local_mismatch=int((older>population).sum());after_mismatch=int((published65>unit_population).sum());assert after_mismatch==0
    points=pd.read_csv(ENTRY/'point_attachment_diagnostics.csv',dtype={'point_id':str,'population_raw':str},keep_default_na=False,low_memory=False,float_precision='round_trip');points=points[points.dataset.eq('population')].copy()
    points['KEY_CODE']=points.point_id.str.split(':').str[1];points=points.set_index('KEY_CODE').loc[a.index]
    assert points.population_raw.equals(a.T001140001.rename('population_raw'))
    pd.DataFrame(links).to_csv(OUT/'suppression_recipient_links.csv',index=False)
    ledger=a[metadata+['T001140001','T001140019','T001140022']].copy();ledger['diagnostic_group_id']=group
    ledger['entry_method']=points.entry_method;ledger['routing_eligible']=points.routing_eligible
    ledger.to_csv(OUT/'mesh_age_entry_audit.csv',index=False)
    # One row per publication support; these are diagnostic counts, not final variables.
    units_out=pd.DataFrame(dict(diagnostic_group_id=unit_population.index,member_meshes=members.values,diagnostic_member_population_sum=unit_population.values,published_recipient_or_singleton_age65=published65.values,published_known_age_sum=known_age.values,diagnostic_unknown_age_residual=(unit_population-known_age).values))
    units_out.to_csv(OUT/'publication_group_checks.csv',index=False)
    coverage=[]
    for keys,idx in ledger.groupby(['HTKSYORI','entry_method']).groups.items():
        coverage.append(dict(HTKSYORI=keys[0],entry_method=keys[1],mesh_count=len(idx),population=int(population.loc[idx].sum())))
    pd.DataFrame(coverage).to_csv(OUT/'age_entry_coverage.csv',index=False)
    sample=[]
    for mode,mask in [('original_500m_attachment',points.attached_500m),('candidate_with_mesh_proxy',points.routing_eligible),('pending_entry',~points.routing_eligible),('candidate_medical_component_reachable',points.medical_component_reachable_exact),('candidate_ems_component_reachable',points.ems_base_component_reachable_exact)]:
        sample.append(dict(scope=mode,meshes=int(mask.sum()),population_in_these_meshes=int(population[mask].sum()),population_share=float(population[mask].sum()/population.sum())))
    pd.DataFrame(sample).to_csv(OUT/'sample_coverage.csv',index=False)
    # Retain skill-standard variable screening for both original, untransformed tables.
    variable_rows=[];screening=[]
    for table,df in [('T001140',a.reset_index(drop=True)),('T001188',b.reset_index(drop=True))]:
        rows,candidates,times=skill.summarize_dataset(table,Path(f'data/raw/population/census_2020_1km_jgd2011/{table}/*.zip'),df)
        assert not times;variable_rows.extend(rows);screening.extend(candidates)
    pd.DataFrame(variable_rows).to_csv(OUT/'variable_summary.csv',index=False);pd.DataFrame(screening).to_csv(OUT/'variable_screening.csv',index=False)
    symbols=[]
    for table,df in [('T001140',a),('T001188',b)]:
        for col in df.columns[4:]:
            for value,n in df.loc[pd.to_numeric(df[col],errors='coerce').isna(),col].value_counts().items():symbols.append(dict(dataset=table,variable=col,symbol=value,count=int(n)))
    pd.DataFrame(symbols).to_csv(OUT/'symbol_audit.csv',index=False)
    pd.DataFrame(s1+s2).to_csv(OUT/'population_source_integrity.csv',index=False)
    fig,ax=skill.plt.subplots(1,2,figsize=(10,4))
    role=a.HTKSYORI.value_counts().reindex(['0','1','2']);ax[0].bar(['Standalone','Recipient','Suppressed'],role,color=['#4C78A8','#F58518','#B279A2']);ax[0].set_ylabel('Mesh records');ax[0].set_title('Published census age support')
    pending=population[~points.routing_eligible];ax[1].hist(pending,bins=25,color='#E45756',edgecolor='white');ax[1].set_xlabel('Published local population');ax[1].set_ylabel('Pending meshes');ax[1].set_title('200 unresolved road-entry meshes')
    fig.tight_layout();fig.savefig(figures/'age_support_and_pending_population.png',dpi=160);skill.plt.close(fig)
    fact=dict(status='passed',tables_read=2,source_archives_verified=len(s1+s2),source_rows_per_table=len(a),variables_summarized=len(variable_rows),plots_generated=1,skipped_files=0,time_series_candidates=0,original_population_total=int(population.sum()),suppressed_meshes=int(a.HTKSYORI.eq('2').sum()),recipient_meshes=int(a.HTKSYORI.eq('1').sum()),standalone_meshes=int(a.HTKSYORI.eq('0').sum()),suppressed_population=int(population[a.HTKSYORI.eq('2')].sum()),recipient_and_suppressed_population=int(population[a.HTKSYORI.ne('0')].sum()),reciprocal_links_checked=len(links),publication_support_units=len(unit_population),elderly_above_local_population_before=local_mismatch,elderly_above_group_population_after=after_mismatch,published_age65_total=int(published65.sum()),diagnostic_unknown_age_total=int((unit_population-known_age).sum()),age_band_and_broad_age_counts_match=True,metadata_match_across_tables=True,raw_data_modified=False,final_age_variable_selected=False,sample_coverage=sample)
    (OUT/'age_aggregation_summary.json').write_text(json.dumps(fact,indent=2)+'\n');print(json.dumps(fact,indent=2),flush=True)


if __name__=='__main__':main()
