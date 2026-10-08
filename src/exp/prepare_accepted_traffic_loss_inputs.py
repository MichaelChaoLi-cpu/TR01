"""Construct the adopted traffic crosswalk and preserve every graph road row."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / 'data/exp/national-traffic-geometry/line-candidates/2d023859cca7'
OUT = ROOT / 'data/exp/data-preprocessing/road-traffic-comparison'
LOSS = ROOT / 'data/exp/national-road-service-loss'
GRAPH = ROOT / 'data/processed/national-network-rule-validation'
COMPARISON = ROOT / 'data/processed/road-traffic-comparison/national_road_traffic_comparison_preprocessed.parquet'
CROSSWALK = ROOT / 'data/processed/observed-traffic/accepted_traffic_road_crosswalk_preprocessed.parquet'
FIELDS = ['Graph Source Edge ID', 'Observed Traffic Volume', 'Lost Medical Path Population',
          'Additional Medical Distance Burden', 'Traffic Observation Key', 'Traffic Match Status',
          'Baseline Source Road Length', 'Closed Origin Entry Population']
STATES = ['unmatched', 'ambiguous', 'matched']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inputs():
    receipt = json.loads((OUT / 'matching_decision.json').read_text())
    assert receipt['authorization'] == 'MILI-D-20261007-015' and receipt['status'] == 'adopted'
    audit = json.loads((PROPOSAL / 'proposal_coverage_validation.json').read_text())
    assert audit['status'] == 'passed' and audit['all_national_tiles']
    national = json.loads((LOSS / 'validation.json').read_text())
    assert national['status'] == 'passed' and national['national_complete']
    roads = pd.read_parquet(PROPOSAL / 'proposed_unique_crosswalk.parquet').sort_values('Graph Source Edge ID').reset_index(drop=True)
    ambiguous = pd.read_parquet(PROPOSAL / 'ambiguous_crosswalk.parquet')['Graph Source Edge ID'].unique()
    ids = roads['Graph Source Edge ID'].to_numpy(dtype=np.int64)
    assert len(ids) == 1183367 and roads['Graph Source Edge ID'].is_unique
    assert not np.isin(ids, ambiguous).any()
    assert roads['Technical Proposal Qualified'].all() and roads['Complete Observation Unit Geometry'].all()
    assert roads['Road Fraction Within Diagnostic Buffer'].ge(.8).all()
    assert roads['Aligned Sample Fraction'].ge(.8).all()
    counts = pd.read_parquet(ROOT / 'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet')
    counts = counts[counts['Traffic Measurement Eligible']].groupby('Traffic Observation Key')['Observed Traffic Volume']
    assert counts.nunique().eq(1).all()
    np.testing.assert_array_equal(roads['Observed Traffic Volume'], roads['Traffic Observation Key'].map(counts.first()))
    assert roads['Observed Traffic Volume'].gt(0).all()
    return roads, ids, ambiguous, national


def construct():
    roads, ids, ambiguous, national = inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in [PROPOSAL / 'proposed_unique_crosswalk.parquet',
              PROPOSAL / 'ambiguous_crosswalk.parquet', LOSS / 'national_road_service_loss.parquet',
              ROOT / 'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet']}
    cross = roads.copy()
    cross['Result Role'] = 'Adopted strict correspondence; MILI-D-20261007-015'
    cross.to_parquet(OUT / 'accepted_crosswalk_construction_input.parquet', index=False)
    n = national['source_road_rows']
    states = np.zeros(n, dtype=np.int8)
    states[ambiguous] = 1
    states[ids] = 2
    codes, units = pd.factorize(roads['Traffic Observation Key'], sort=True)
    keys = np.full(n, -1, dtype=np.int32)
    keys[ids] = codes
    volume = np.full(n, np.nan)
    volume[ids] = roads['Observed Traffic Volume']
    lost = np.load(LOSS / 'lost_medical_path_population.npy', mmap_mode='r')
    burden = np.load(LOSS / 'additional_medical_distance_burden.npy', mmap_mode='r')
    length = np.load(GRAPH / 'edge_lengths.npy', mmap_mode='r')
    entry = np.load(LOSS / 'closed_origin_entry_population.npy', mmap_mode='r')
    output = OUT / 'national_traffic_loss_construction_input.parquet'
    writer = None
    try:
        for first in range(0, n, 500000):
            last = min(n, first + 500000)
            key_array = pa.DictionaryArray.from_arrays(pa.array(keys[first:last], mask=keys[first:last] < 0), pa.array(units))
            state_array = pa.DictionaryArray.from_arrays(pa.array(states[first:last]), pa.array(STATES))
            table = pa.table(dict(zip(FIELDS, [np.arange(first,last,dtype=np.int32), volume[first:last],
                             np.asarray(lost[first:last]), np.asarray(burden[first:last]), key_array, state_array,
                             np.asarray(length[first:last]), np.asarray(entry[first:last])])) )
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    scientific = {'Observed Traffic Volume', 'Lost Medical Path Population', 'Additional Medical Distance Burden'}
    datasets = {}
    for src, dst, script, cols in [
        (OUT / 'accepted_crosswalk_construction_input.parquet', CROSSWALK,
         'src/preprocessing/preprocess_accepted_traffic_road_crosswalk.py', list(cross.columns)),
        (output, COMPARISON, 'src/preprocessing/preprocess_national_road_traffic_comparison.py', FIELDS)]:
        datasets[str(src.relative_to(ROOT))] = {'output': str(dst.relative_to(ROOT)), 'script': script,
            'variables': [{'original_name': col, 'readable_name': col, 'full_name': col,
                           'is_final_variable': 'yes' if col in scientific else 'no', 'preprocessing': []} for col in cols]}
    decision = {'authorization': 'MILI-D-20261007-015', 'other_authorizations': ['MILI-D-20261007-013', 'MILI-D-20261007-014'],
                'scope': 'Adopted unique road correspondence; all graph roads retained with unavailable traffic missing; no filling or clipping',
                'datasets': datasets}
    (OUT / 'decisions.json').write_text(json.dumps(decision,indent=2)+'\n')
    (OUT / 'construction_validation.json').write_text(json.dumps({'status':'passed','source_hashes':hashes,
          'source_road_rows':n,'matched':len(ids),'ambiguous':len(ambiguous),'unmatched':int((states==0).sum()),
          'assigned_observation_units':len(units),'assigned_values_equal_measured_source':True},indent=2)+'\n')
    print(f'Constructed {n:,} road rows and {len(ids):,} adopted crosswalk rows.',flush=True)


def validate():
    roads, ids, ambiguous, national = inputs()
    actual = pd.read_parquet(CROSSWALK)
    expected = pd.read_parquet(OUT / 'accepted_crosswalk_construction_input.parquet')
    pd.testing.assert_frame_equal(actual, expected)
    assert pq.ParquetFile(COMPARISON).schema_arrow.names == FIELDS
    offset = 0
    matched = ambiguous_count = unmatched = 0
    source = pq.ParquetFile(OUT / 'national_traffic_loss_construction_input.parquet')
    processed = pq.ParquetFile(COMPARISON)
    assert source.metadata.num_rows == processed.metadata.num_rows == national['source_road_rows']
    original_loss = np.load(LOSS / 'lost_medical_path_population.npy', mmap_mode='r')
    original_burden = np.load(LOSS / 'additional_medical_distance_burden.npy', mmap_mode='r')
    original_length = np.load(GRAPH / 'edge_lengths.npy', mmap_mode='r')
    original_entry = np.load(LOSS / 'closed_origin_entry_population.npy', mmap_mode='r')
    ambiguous = np.sort(ambiguous)
    for result in processed.iter_batches(batch_size=500000):
        b = result.to_pandas()
        end = offset + len(b)
        for column, values in [('Lost Medical Path Population', original_loss),
                               ('Additional Medical Distance Burden', original_burden),
                               ('Baseline Source Road Length', original_length),
                               ('Closed Origin Entry Population', original_entry)]:
            np.testing.assert_array_equal(b[column], values[offset:end])
        low, high = np.searchsorted(ids, [offset, end])
        positions = ids[low:high] - offset
        expected_volume = np.full(len(b), np.nan)
        expected_volume[positions] = roads['Observed Traffic Volume'].to_numpy()[low:high]
        np.testing.assert_array_equal(b['Observed Traffic Volume'], expected_volume)
        expected_states = np.full(len(b), 'unmatched', dtype=object)
        alow, ahigh = np.searchsorted(ambiguous, [offset, end])
        expected_states[ambiguous[alow:ahigh]-offset] = 'ambiguous'
        expected_states[positions] = 'matched'
        np.testing.assert_array_equal(b['Traffic Match Status'].astype(str), expected_states)
        np.testing.assert_array_equal(b.iloc[positions]['Traffic Observation Key'].astype(str),
                                      roads['Traffic Observation Key'].to_numpy()[low:high])
        np.testing.assert_array_equal(b['Graph Source Edge ID'], np.arange(offset,offset+len(b)))
        state = b['Traffic Match Status'].astype(str)
        matched += int(state.eq('matched').sum())
        ambiguous_count += int(state.eq('ambiguous').sum())
        unmatched += int(state.eq('unmatched').sum())
        assert b.loc[state.ne('matched'),'Observed Traffic Volume'].isna().all()
        assert b.loc[state.ne('matched'),'Traffic Observation Key'].isna().all()
        assert b.loc[state.eq('matched'),'Observed Traffic Volume'].notna().all()
        assert b['Lost Medical Path Population'].notna().all() and b['Additional Medical Distance Burden'].notna().all()
        offset += len(b)
    assert offset == national['source_road_rows'] and matched == len(ids) and ambiguous_count == len(ambiguous)
    construction = json.loads((OUT / 'construction_validation.json').read_text())
    for path, expected_hash in construction['source_hashes'].items():
        assert digest(ROOT / path) == expected_hash
    receipt = {'status':'passed','authorization':'MILI-D-20261007-015','source_road_rows':offset,
               'comparison_road_rows':matched,'ambiguous':ambiguous_count,'unmatched':unmatched,
               'observed_units':int(roads['Traffic Observation Key'].nunique()),'identity_readback_exact':True,
               'unmatched_traffic_missing':True,'no_new_baseline_exclusions':True,'sources_unchanged':True,
               'output_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [CROSSWALK,COMPARISON]}}
    (OUT / 'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--validate',action='store_true')
    args = parser.parse_args()
    validate() if args.validate else construct()
