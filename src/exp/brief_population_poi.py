"""Read archived sources without extraction; create a skill-guided briefing.

Run from the root: .venv/bin/python src/exp/brief_population_poi.py
No analysis-ready data or modified source records are saved. Numeric coercion is
used only for descriptive statistics; all nonnumeric source symbols are audited.
"""
import csv
import importlib.util
import io
import json
import math
import os
import zipfile
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tr01-matplotlib')
import pandas as pd
import numpy as np
import shapefile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/preprocessing'))
from correct_municipal_office_code import apply_corrections
OUT = ROOT / 'data/exp/data-briefing'
spec = importlib.util.spec_from_file_location('briefing_skill', ROOT / '.codex/skills/data-briefing/scripts/create_data_briefing.py')
skill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skill)


def layers(archive, provenance):
    for name in archive.namelist():
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as child:
                yield from layers(child, provenance + '!' + name)
        elif name.endswith('.shp'):
            yield archive, name, provenance + '!' + name


def main():
    tables = OUT / 'tables'
    plots = OUT / 'figures/distributions'
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    inventory, summaries, screening, generated = [], [], [], []
    frequencies, coverage, symbols, labels, skipped = [], [], [], [], []
    numeric_summaries, anomalies = [], []
    facts = {}

    def review(name, source, df):
        inventory.append(dict(dataset=name, source_file=source, rows=len(df), columns=len(df.columns)))
        variables, candidates, _ = skill.summarize_dataset(name, Path(source), df)
        summaries.extend(variables)
        screening.extend(candidates)

    def histogram(name, field, values, title):
        path = plots / f'{name}_{field}.png'
        values = pd.to_numeric(values, errors='coerce').dropna()
        positive = values[values>0]
        if not positive.empty:
            fig, ax = skill.plt.subplots(figsize=(8,5))
            bins = np.geomspace(max(.5,positive.min()*.9),positive.max()*1.1,40)
            ax.hist(positive,bins=bins,color='#4C78A8',edgecolor='white')
            ax.set_xscale('log')
            ax.set_title(title)
            ax.set_xlabel('Published population count (logarithmic axis)')
            ax.set_ylabel('Mesh count')
            ax.text(.98,.98,f'Numeric zero counts: {(values==0).sum():,}\nNonnumeric symbols excluded',
                    ha='right',va='top',transform=ax.transAxes,fontsize=9)
            fig.tight_layout()
            fig.savefig(path,dpi=160)
            skill.plt.close(fig)
            generated.append(str(path.relative_to(OUT)))

    def categorical(name, field, values):
        path = plots / f'{name}_{skill.safe_name(field)}.png'
        if skill.plot_categorical_distribution(values, path, f'{name}: {field}'):
            generated.append(str(path.relative_to(OUT)))

    for table in ['T001140', 'T001188']:
        folder = ROOT / 'data/raw/population/census_2020_1km_jgd2011' / table
        frames = []
        for path in sorted(folder.glob('*.zip')):
            with zipfile.ZipFile(path) as archive:
                with archive.open(archive.namelist()[0]) as stream:
                    reader = csv.reader(io.TextIOWrapper(stream, encoding='cp932'))
                    columns, titles = next(reader), next(reader)
                    if not frames:
                        labels.extend(dict(dataset=table, variable=c, source_label=t.strip()) for c,t in zip(columns,titles))
                    frames.append(pd.DataFrame(reader, columns=columns))
        df = pd.concat(frames, ignore_index=True)
        review(table, str(folder.relative_to(ROOT)) + '/*.zip', df)
        numeric = df.iloc[:,4:].apply(pd.to_numeric, errors='coerce')
        for col in numeric:
            counts = df.loc[numeric[col].isna(),col].value_counts(dropna=False)
            symbols.extend(dict(dataset=table, variable=col, symbol=str(v), count=int(n)) for v,n in counts.items())
        frequencies.extend(dict(dataset=table, variable='HTKSYORI', value=v, count=int(n))
                           for v,n in df.HTKSYORI.value_counts().items())
        categorical(table, 'HTKSYORI', df.HTKSYORI)
        histogram(table, 'total_population', numeric[table+'001'], f'{table}: published total population per mesh')
        # Describe only numeric published values; do not fill '-' or '*' with zero.
        for col in numeric:
            series = numeric[col]
            numeric_summaries.append(dict(dataset=table, source_file=str(folder.relative_to(ROOT))+'/*.zip',
                variable=col, dtype='numeric_descriptive_view', rows=len(df), non_null=int(series.notna().sum()),
                missing=int(series.isna().sum()), missing_pct=float(series.isna().mean()),
                unique=series.nunique(), role='published_numeric_values_only', mean=series.mean(),
                std=series.std(), min=series.min(), p25=series.quantile(.25), median=series.median(),
                p75=series.quantile(.75), max=series.max()))
        facts[table] = dict(rows=len(df), columns=len(df.columns), duplicate_mesh_codes=int(df.KEY_CODE.duplicated().sum()),
                           suppression_counts=df.HTKSYORI.value_counts().to_dict(),
                           total_numeric_population=int(numeric[table+'001'].sum()),
                           nonnumeric_total=int(numeric[table+'001'].isna().sum()),
                           invalid_mesh_codes=int((~df.KEY_CODE.str.fullmatch(r'\d{8}')).sum()))
        if table == 'T001140':
            for suffix, title in [('019','Published population aged 65+'),('022','Published population aged 75+')]:
                histogram(table, suffix, numeric[table+suffix], title+' per mesh (numeric values only)')
            facts[table]['elderly_exceeds_local_total'] = int((numeric[table+'019']>numeric[table+'001']).sum())
            facts[table]['age_numeric_unavailable'] = int(numeric[table+'019'].isna().sum())
            mesh_set = set(df.KEY_CODE)
        else:
            facts['population_mesh_sets_equal'] = set(df.KEY_CODE) == mesh_set
        print(f'{table}: {len(df):,} rows, {len(df.columns)} columns', flush=True)

    class_fields = dict(fire_stations='P17_003', medical='P04_001', welfare='P14_006',
                        municipal_offices='P34_002', post_offices='P30_004', roadside_stations='P35_011')
    pref_fields = dict(fire_stations='P17_002', welfare='P14_003', municipal_offices='P34_001',
                       post_offices='P30_001', roadside_stations='P35_005')
    for folder in sorted((ROOT / 'data/raw/poi').iterdir()):
        groups = {}
        points = []
        name = folder.name
        for path in sorted(folder.glob('*.zip')):
            with zipfile.ZipFile(path) as archive:
                for z, member, source in layers(archive, str(path.relative_to(ROOT))):
                    stem = member[:-4]
                    encoding = 'utf-8' if name=='welfare' else 'cp932'
                    reader = shapefile.Reader(shp=io.BytesIO(z.read(member)), dbf=io.BytesIO(z.read(stem+'.dbf')), encoding=encoding)
                    fields = [f[0] for f in reader.fields[1:]]
                    df = pd.DataFrame([list(r) for r in reader.iterRecords()], columns=fields)
                    if name == 'municipal_offices':
                        df = pd.DataFrame([apply_corrections(row,path.name)[0] for row in df.to_dict('records')], columns=fields)
                    group = name if reader.shapeType==1 else name+'_jurisdictions'
                    groups.setdefault(group, []).append(df)
                    if reader.shapeType==1:
                        points.extend(tuple(s.points[0]) if s.points else (float('nan'),float('nan')) for s in reader.iterShapes())
        for group, frames in groups.items():
            df = pd.concat(frames, ignore_index=True)
            review(group, str(folder.relative_to(ROOT))+'/*.zip (Shapefile attributes)', df)
            facts[group] = dict(rows=len(df), columns=len(df.columns))
            if group == name:
                field = class_fields[name]
                categorical(name,field,df[field].astype(str))
                for value,count in df[field].astype(str).value_counts(dropna=False).items():
                    frequencies.append(dict(dataset=name, variable=field,value=value,count=int(count)))
                coords = pd.DataFrame(points,columns=['longitude','latitude'])
                valid = coords.longitude.between(120,155) & coords.latitude.between(20,46)
                facts[group].update(point_geometries=len(coords), coordinates_outside_broad_japan_bounds=int((~valid).sum()),
                    duplicate_coordinates=int(coords.duplicated().sum()),
                    extent=[float(coords.longitude.min()),float(coords.latitude.min()),float(coords.longitude.max()),float(coords.latitude.max())])
                if name in pref_fields:
                    bad = ~df[pref_fields[name]].astype(str).str.fullmatch(r'\d{5}')
                    for index in df.index[bad]:
                        anomalies.append(dict(dataset=name, issue='Invalid administrative code',
                                              source_value=str(df.loc[index,pref_fields[name]]),
                                              source_record=json.dumps(df.loc[index].to_dict(),ensure_ascii=False)))
                    pref = df[pref_fields[name]].astype(str).str[:2]
                    for code,count in pref.value_counts().items():
                        coverage.append(dict(dataset=name,prefecture_code=code,records=int(count)))
                    facts[group]['prefecture_codes'] = sorted(pref.unique().tolist())
        print(f'{name}: {facts[name]["rows"]:,} points', flush=True)

    for filename, rows in [('dataset_inventory',inventory),('variable_summary',summaries),
                           ('variable_screening',screening),('category_frequencies',frequencies),
                           ('prefecture_coverage',coverage),('population_symbol_counts',symbols),
                           ('source_variable_labels',labels),('numeric_descriptive_summary',numeric_summaries),
                           ('source_anomalies',anomalies),('skipped_files',skipped)]:
        frame = pd.DataFrame(rows)
        if filename == 'skipped_files' and not rows:
            frame = pd.DataFrame(columns=['source_file','reason'])
        if filename == 'source_anomalies' and not rows:
            frame = pd.DataFrame(columns=['dataset','issue','source_value','source_record'])
        frame.to_csv(tables / (filename+'.csv'),index=False)
    pd.DataFrame(columns=['dataset','source_file','time_columns']).to_csv(tables/'time_series_candidates.csv',index=False)
    (OUT/'inspection_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
    skill.write_markdown_report(OUT/'README.md',inventory,[],generated,skipped,False)
    skill.update_anasop_data_overview(ROOT,OUT,inventory,[],skipped,len(summaries),len(generated))
    overview_names = ['Census population and households', 'Census five-year age groups',
                      'Fire facility points', 'Fire jurisdiction polygons', 'Medical facility points',
                      'Municipal office points', 'Postal facility points', 'Roadside station points',
                      'Welfare facility points']
    anasop = ROOT/'docs/AnaSOP.md'
    content = anasop.read_text()
    for i, name in enumerate(overview_names, 1):
        content = content.replace(f'| Data source {i} |',f'| {name} |')
    anasop.write_text(content)
    limitations = '''
### Source-specific interpretation limits

- Population observations refer to 2020; there is no longitudinal population series in the current acquisition.
- The acquired census tables contain published mesh records; an absent mesh record is not automatically an observed zero.
- Raw values and codes were retained. Temporary numeric views describe numeric published values only; nonnumeric symbols were audited separately and never replaced with zeros.
- Age-specific counts in suppressed meshes are aggregated into recipient meshes while total population remains local. Direct mesh-level elderly shares can therefore be invalid; aggregation handling remains a preprocessing decision.
- Facility reference years range from 2012 to 2023. Historical presence does not establish current service availability. Medical and postal sources include suspended facilities.
- Fire headquarters, stations and branches remain distinct. The user specified ambulance deployment at stations and branches as a modeling assumption; no vehicle availability or dispatch time was observed.
- Duplicate coordinates may represent co-located facilities, not duplicate records. Coordinate bounds are a screening check, not validation against administrative boundaries or the road network.
- Spatial layers use different reference systems. No coordinate transformation, network snapping or destination exclusion was performed.
- Fire jurisdiction polygons were inventoried separately from facility points; alternate GeoJSON representations were not counted again.
- One invalid municipal administrative code was corrected in the derived inspection view using a verified, record-specific rule. Original source archives remain unchanged; the corrected record belongs to Nikko City. No invalid municipal codes remain in the corrected view.
- Facility categories and population variables are candidates; final selection belongs to variable construction and research planning.
- Road connectivity, infrastructure conditions, traffic counts, maintenance costs and intervention effects have not been evaluated in this briefing.
'''
    with (OUT/'README.md').open('a') as stream:
        stream.write(limitations)
        stream.write('\n## Inspection facts\n\n```json\n'+json.dumps(facts,ensure_ascii=False,indent=2)+'\n```\n')
    with (ROOT/'docs/AnaSOP.md').open('a') as stream:
        stream.write(limitations)
    print(json.dumps(dict(datasets=len(inventory),variables=len(summaries),plots=len(generated),facts=facts),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
