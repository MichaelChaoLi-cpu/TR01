"""Scoped source briefing; preserve every field and existing AnaSOP sections.

Run locally from the project root with .venv/bin/python. Source values and IDs
are not recoded; reordered columns only determine four exploratory plot choices.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/exp/feasibility-check/national-pavement-inventory/pavement_basic_index.csv.gz'
OUT = ROOT / 'data/exp/data-briefing/pavement-inventory'
SOP = ROOT / 'docs/AnaSOP.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(text):
    import re
    return {int(m.group(1)): m.group(0) for m in re.finditer(
        r'^## ([1-8])\. .*?(?=^## [1-8]\. |\Z)', text, re.M | re.S)}


def refine_distributions(df):
    """Replace the same four bundled plots with readable exploratory displays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = OUT / 'figures/distributions'
    prefix = 'data_exp_data-briefing_pavement-inventory_pavement_source_briefing_view__'
    titles = ['Pavement inspection grade', 'Reported inspection interval length',
              'Latest inspection year', 'Reported last-repair year']
    fields = ['tenken.kiroku.hantei_kubun', 'syogen.enchou', 'tenken.nendo',
              'syogen.saishuu_syuuzen_nendo']
    for field, title in zip(fields, titles):
        fig, ax = plt.subplots(figsize=(6.5, 3.5), layout='constrained')
        values = pd.to_numeric(df[field], errors='coerce')
        if field == fields[0]:
            codes = ['1', '2', '3', '8']
            ax.bar(['I', 'II', 'III', 'Uninspected (8)'],
                   [int(df[field].eq(c).sum()) for c in codes], color='#4C78A8')
            ax.set_xlabel('Official category')
        elif field == fields[1]:
            counts = df[field].value_counts()
            if len(counts) <= 10:
                keys = sorted(counts.index, key=float)
                ax.bar(keys, [counts[k] for k in keys], color='#4C78A8')
            else:
                ax.hist(values.dropna(), bins=50, color='#4C78A8', edgecolor='white')
            ax.set_xlabel('Reported interval length (m)')
        elif field == fields[2]:
            counts = df[field].value_counts()
            keys = sorted(counts.index, key=int)
            labels = ['0 (source code)' if k == '0' else k for k in keys]
            ax.bar(labels, [counts[k] for k in keys], color='#4C78A8')
            ax.tick_params(axis='x', labelrotation=60, labelsize=8)
            ax.set_xlabel('Latest recorded inspection year; cross-sectional counts')
        else:
            ax.hist(values.dropna(), bins=50, color='#4C78A8', edgecolor='white')
            ax.set_xlabel('Reported year')
            ax.text(.02, .95, f'Blank source entries: {int(df[field].eq("").sum()):,}',
                    transform=ax.transAxes, va='top', fontsize=9, color='black')
        ax.set_ylabel('Inspection records')
        ax.set_title(title, fontsize=12)
        ax.ticklabel_format(axis='y', style='plain')
        fig.savefig(directory / (prefix + field.replace('.', '_') + '.png'), dpi=200)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = sha(SOURCE)
    df = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    assert len(df) == 472720 and df['shisetsu_id'].nunique() == len(df)
    assert df['source_prefecture'].nunique() == 47
    fields = ['tenken.kiroku.hantei_kubun', 'syogen.enchou',
              'tenken.nendo', 'syogen.saishuu_syuuzen_nendo']
    # Lossless CSV view for the bundled reader, which does not discover .csv.gz.
    columns = fields + [c for c in df.columns if c not in fields]
    view = OUT / 'pavement_source_briefing_view.csv'
    df[columns].to_csv(view, index=False)
    reread = pd.read_csv(view, dtype=str, keep_default_na=False)[df.columns]
    assert reread.equals(df)
    length = pd.to_numeric(df['syogen.enchou'], errors='coerce')
    lat = pd.to_numeric(df['syogen.ichi.ido'], errors='coerce')
    lon = pd.to_numeric(df['syogen.ichi.keido'], errors='coerce')
    audit = {
        'records': len(df), 'source_fields': len(df.columns),
        'source_sha256': source_hash,
        'view_preserves_all_fields_and_values': True,
        'empty_field_counts': {c: int(df[c].eq('').sum()) for c in df},
        'grade_counts': df['tenken.kiroku.hantei_kubun'].value_counts().to_dict(),
        'year_counts': df['tenken.nendo'].value_counts().to_dict(),
        'repair_year_counts': df['syogen.saishuu_syuuzen_nendo'].value_counts().to_dict(),
        'administrator_counts': df['syogen.kanrisya.kubun'].value_counts().to_dict(),
        'invalid_or_nonpositive_length': int((length.isna() | length.le(0)).sum()),
        'length_summary_m': length.describe(percentiles=[.25, .5, .75, .95, .99]).to_dict(),
        'invalid_or_out_of_range_coordinates': int((lat.isna() | lon.isna() |
            ~lat.between(-90, 90) | ~lon.between(-180, 180)).sum()),
        'zero_coordinates': int((lat.eq(0) | lon.eq(0)).sum()),
        'source_prefecture_disagreement': int((
            df['source_prefecture'].str.zfill(2) !=
            df['syogen.gyousei_kuiki.todoufuken_code'].str.zfill(2)).sum()),
        'accepted_road_assignments': 0, 'scientific_columns_selected_final': False,
        'missing_values_imputed': False,
    }
    (OUT / 'source_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    before = SOP.read_text()
    (OUT / 'anasop_before_briefing.md').write_text(before)
    spec = importlib.util.spec_from_file_location('briefing', ROOT /
        '.codex/skills/data-briefing/scripts/create_data_briefing.py')
    briefing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(briefing)
    report = briefing.create_briefing(SimpleNamespace(
        root=str(ROOT), output=str(OUT.relative_to(ROOT)), inputs=[str(view.relative_to(ROOT))],
        include_time_series='no', max_distribution_plots=4, max_categorical_levels=50,
        max_time_columns=3, max_time_series_variables=6))
    assert report['datasets_read'] == 1 and report['variables_summarized'] == 24
    assert not report['skipped_files'] and not report['time_series_candidates']
    assert report['distribution_plots'] == 4
    refine_distributions(df)
    old_sections = sections(before)
    subsection = '''### Pavement inspection inventory for the main maintenance scenarios

The complete public basic inventory contains 472,720 distinct inspection records across all 47 prefectures. Acquisition reconciled the national count and every prefecture count; all 117 cached response pages passed recorded-hash checks. The scoped briefing retains all 21 returned basic fields and three acquisition-provenance fields, reviews their missingness and screening status, and generates four exploratory distributions. No files were skipped. Inspection and last-repair years describe heterogeneous latest records, not repeated observations of each road; no longitudinal visualizations were generated.

There are 236,296 category-I, 165,134 category-II and 53,294 category-III records, giving 454,724 records with a usable observed pavement grade. Another 17,996 records are marked uninspected. Uninspected records are retained and do not imply good condition. All records report the same national administrator category; coverage of all prefectures therefore does not establish coverage of municipal, prefectural or all other roads. Grades are pavement-specific and must not inherit the bridge category-IV interpretation.

Inspection records provide a point, a reported interval length, pavement type, inspection year and latest-repair information. Every acquired record reports a 100-metre interval, while the existing network source curves have heterogeneous lengths. Their points are locators, not verified complete interval footprints. Record IDs do not identify one-to-one network maintenance units. Complete interval geometry, condition-calibrated speed or impairment, observed maintenance expenditure and future deterioration are not established by the acquired basic inventory. Final conditional anchors, repeated-record aggregation and the measured-traffic common sample remain to be constructed; the 454,724 graded records are an input count, not the eligible road count.

The human-adopted main scenario uses finite, uniform impedance increases of 25 percent, with 10 and 50 percent sensitivities. These increments are explicit assumptions, not estimates from the grades. Costs and four budget shares will be defined on the eventual common road-segment sample. Missing condition or measured traffic will not be imputed. The full medical/population baseline and independent bridge comparison remain intact.
'''
    old3 = old_sections[3]
    existing_header = '### Pavement inspection inventory for the main maintenance scenarios\n'
    if existing_header in old3:
        import re
        old3 = re.sub('^' + re.escape(existing_header) + r'.*?(?=^### |\Z)',
                      '', old3, flags=re.M | re.S)
    note = ('The summary counts above describe the initial and bridge briefings. '
            'Subsequent projection and pavement briefings are reported separately below.\n\n')
    if note.strip() not in old3:
        old3 = old3.replace('### Time-Series Candidates\n', note + '### Time-Series Candidates\n', 1)
    updated3 = old3.rstrip() + '\n\n' + subsection.rstrip() + '\n\n'
    final = before.replace(old_sections[3], updated3)
    SOP.write_text(final)
    after_sections = sections(final)
    assert all(old_sections[n] == after_sections[n] for n in old_sections if n != 3)
    assert sha(SOURCE) == source_hash
    (OUT / 'briefing_run.json').write_text(json.dumps(report, indent=2) + '\n')
    validation = {'status': 'passed_scoped_source_briefing',
        'source_values_and_all_fields_preserved': True, 'section_updated': 3,
        'other_sections_preserved_exactly': True, 'source_hash_unchanged': True,
        'no_final_variables_or_policy_outcomes_constructed': True,
        'time_series_candidates': 0, 'time_series_plots': 0,
        'datasets_read': 1, 'variables_summarized': 24, 'distribution_plots': 4,
        'skipped_files': 0, 'sop_sha256_at_briefing': sha(SOP)}
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    with (OUT / 'README.md').open('a') as stream:
        stream.write('\n## Source interpretation\n\n'
            'The CSV view preserves all fields, rows and values; column order places grade, '
            'interval length, inspection year and last-repair year first for four exploratory plots. '
            'Identifier and provenance fields are summarized but are not scientific predictors. '
            'Numeric inspection categories are labels, not continuous severity measurements. '
            'Year zero and uninspected category 8 remain original source codes; neither is imputed. '
            'Cross-sectional latest-year frequencies are not a longitudinal panel. '
            'No final variables or accepted road assignments are created. '
            'The same four bundled distribution plots are refined locally for readable titles '
            'and categorical grade/year displays without recoding saved source values.\n')
    print(json.dumps({'audit': audit, 'validation': validation}, indent=2))


if __name__ == '__main__':
    main()
