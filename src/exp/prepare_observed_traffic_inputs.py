"""Build a source-preserving input for the approved measured daytime variable.

Run before the generated national traffic preprocessing script. Source CSVs use
positional fields because their headers contain repeated names.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/exp/national-traffic-acquisition/sources'
OUT = ROOT / 'data/exp/data-preprocessing/observed-traffic'
FIELDS = {0: 'Traffic Basic Section', 3: 'Traffic Road Class', 4: 'Traffic Route Number',
          5: 'Traffic Route Name', 18: 'Traffic Municipality Code', 23: 'Traffic Section Length',
          26: 'One Way Code', 30: 'Traffic Observation Authority', 31: 'Traffic Observation Unit',
          32: 'Up Observation Section', 34: 'Up Observation Code', 35: 'Up Individual Survey Code',
          36: 'Up Observation Duration Code', 37: 'Up Observation Date',
          40: 'Up Daytime Source Total', 44: 'Down Observation Section',
          46: 'Down Observation Code', 47: 'Down Individual Survey Code',
          48: 'Down Observation Duration Code', 49: 'Down Observation Date',
          52: 'Down Daytime Source Total', 58: 'Bidirectional Daytime Source Total',
          61: 'Bidirectional Full Day Source Total'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, hashes = [], {}
    for pref in range(1, 48):
        path = SOURCE / f'kasyo{pref:02d}.csv'
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding='cp932', newline='') as handle:
            reader = csv.reader(handle)
            headers = next(reader)
            assert len(headers) == 159
            rows = []
            for ordinal, values in enumerate(reader, 1):
                assert len(values) == 159
                row = {name: values[index] for index, name in FIELDS.items()}
                row.update({'Traffic Source Prefecture': f'{pref:02d}', 'Traffic Source Row': ordinal})
                rows.append(row)
        frame = pd.DataFrame(rows)
        assert not frame['Traffic Basic Section'].duplicated().any()
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df['Traffic Geometry Section'] = df['Traffic Basic Section'].str.zfill(11)
    # Preserve every literal source value; numeric views are separately derived.
    up = pd.to_numeric(df['Up Daytime Source Total'], errors='coerce')
    down = pd.to_numeric(df['Down Daytime Source Total'], errors='coerce')
    total = pd.to_numeric(df['Bidirectional Daytime Source Total'], errors='coerce')
    observed = df['Up Observation Code'].eq('1') & df['Down Observation Code'].eq('1')
    unconverted = df['Up Individual Survey Code'].eq('0') & df['Down Individual Survey Code'].eq('0')
    durations = df['Up Observation Duration Code'].isin(['1', '2']) & df['Down Observation Duration Code'].isin(['1', '2'])
    counts_valid = up.notna() & down.notna() & total.notna() & up.ge(0) & down.ge(0) & total.ge(0)
    sums_valid = (up + down).eq(total)
    eligible = observed & unconverted & durations & counts_valid & sums_valid
    assert (observed == eligible).all(), 'Unexpected observed code/count inconsistency requires inspection'
    df['Observed Traffic Volume'] = total.where(eligible, np.nan)
    df['Traffic Measurement Eligible'] = eligible
    df['Traffic Measurement Status'] = np.where(eligible, 'Observed bidirectional daytime total',
                                               'Unavailable as an observed bidirectional total')
    df['Traffic Observation Key'] = df['Traffic Observation Authority'] + ':' + df['Traffic Observation Unit']
    df.to_parquet(OUT / 'traffic_construction_input.parquet', index=False)
    selected = df[eligible]
    groups = selected.groupby('Traffic Observation Key')['Observed Traffic Volume'].nunique()
    # Do not collapse repeated units: differences within them are reported.
    coverage = df.groupby('Traffic Source Prefecture', sort=True).agg(
        SourceRows=('Traffic Basic Section', 'size'),
        MeasuredRows=('Traffic Measurement Eligible', 'sum'))
    coverage.to_csv(OUT / 'prefecture_coverage.csv')
    summary = {'status': 'passed', 'authorization': 'MILI-D-20261007-014',
               'source_prefectures': len(frames), 'source_rows': len(df), 'columns': len(df.columns),
               'measured_rows': int(eligible.sum()), 'missing_measured_rows': int((~eligible).sum()),
               'distinct_measured_observation_units': int(len(groups)),
               'units_with_multiple_reported_totals': int((groups > 1).sum()),
               'single_direction_observed_rows_without_bidirectional_measure': int(((df['Up Observation Code'].eq('1') ^ df['Down Observation Code'].eq('1')) & ~eligible).sum()),
               'zero_measured_totals': int(selected['Observed Traffic Volume'].eq(0).sum()),
               'measured_quantiles': selected['Observed Traffic Volume'].quantile([0,.25,.5,.75,.95,1]).to_dict(),
               'all_source_rows_retained': True, 'no_missing_values_filled': True,
               'no_graph_assignments': True, 'source_hashes': hashes,
               'comparison_unit_limit': 'Repeated observation units are not independent measurements; geometry fragments must not inflate the measurement count.',
               'one_way_limit': 'A missing directional observation is not set to zero; single-direction-only records remain outside the bidirectional comparison.',
               'year_limit': '2021 traffic values will be compared with a 2024 candidate network; this is not same-year observed maintenance evidence.'}
    (OUT / 'construction_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    variables = [{'original_name': name, 'readable_name': name, 'full_name': name,
                  'is_final_variable': 'yes' if name == 'Observed Traffic Volume' else 'no',
                  'preprocessing': []} for name in df.columns]
    decisions = {'authorization': 'MILI-D-20261007-014',
                 'construction_script': 'src/exp/prepare_observed_traffic_inputs.py',
                 'source_fields_by_position': FIELDS,
                 'datasets': {'data/exp/data-preprocessing/observed-traffic/traffic_construction_input.parquet': {
                     'output': 'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet',
                     'script': 'src/preprocessing/preprocess_national_observed_traffic.py', 'variables': variables}}}
    (OUT / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')
    pd.DataFrame(variables).to_csv(OUT / 'confirmed_variable_list.csv', index=False)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
