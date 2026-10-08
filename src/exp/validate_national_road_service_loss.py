"""Validate the completed national loss estimates and their identity preprocessing."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/exp/national-road-service-loss'
OUT = ROOT / 'data/exp/data-preprocessing/national-road-service-loss'
COLS = ['Graph Source Edge ID', 'Lost Medical Path Population',
        'Additional Medical Distance Burden']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    receipt = json.loads((BASE / 'validation.json').read_text())
    assert receipt['status'] == 'passed' and receipt['national_complete']
    n = receipt['source_road_rows']
    lost = np.load(BASE / 'lost_medical_path_population.npy', mmap_mode='r')
    burden = np.load(BASE / 'additional_medical_distance_burden.npy', mmap_mode='r')
    source = BASE / 'national_road_service_loss.parquet'
    processed = ROOT / 'data/processed/road-service-loss/national_road_service_loss_preprocessed.parquet'
    counts = {}
    for label, path in [('source', source), ('processed', processed)]:
        parquet = pq.ParquetFile(path)
        assert parquet.metadata.num_rows == n
        assert parquet.schema_arrow.names == COLS
        offset = 0
        for batch in parquet.iter_batches(batch_size=500000, columns=COLS):
            values = [batch.column(i).to_numpy() for i in range(3)]
            end = offset + len(values[0])
            np.testing.assert_array_equal(values[0], np.arange(offset, end, dtype=np.int32))
            np.testing.assert_array_equal(values[1], lost[offset:end])
            np.testing.assert_array_equal(values[2], burden[offset:end])
            assert ((values[1] >= 0) & (values[1] <= receipt['original_population'])).all()
            assert np.isfinite(values[2]).all() and (values[2] >= 0).all()
            offset = end
        assert offset == n
        counts[label] = {'rows': offset, 'columns': len(COLS), 'sha256': digest(path)}
    for path, expected in receipt['source_hashes_unchanged'].items():
        assert digest(ROOT / path) == expected, path
    sample = pd.read_parquet(ROOT / 'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet',
                             columns=['Mesh Code', 'Local Total Population'])
    points = pd.read_csv(ROOT / 'data/exp/national-population-entry/point_attachment_diagnostics.csv',
                         usecols=['point_id', 'edge_id'], dtype={'point_id': str}, keep_default_na=False)
    origins = points.set_index('point_id').loc['mesh:' + sample['Mesh Code'].astype(str)]
    edges = origins['edge_id'].to_numpy(dtype=np.int64)
    population = sample['Local Total Population'].to_numpy(dtype=np.int64)
    assert len(sample) == receipt['original_meshes'] and int(population.sum()) == receipt['original_population']
    assert ((edges >= 0) & (edges < n)).all()
    entry = np.zeros(n, dtype=np.int64)
    np.add.at(entry, edges, population)
    assert int(entry.sum()) == receipt['original_population'] and (lost >= entry).all()
    entry_path = BASE / 'closed_origin_entry_population.npy'
    np.save(entry_path, entry)
    config = json.loads((BASE / 'configuration.json').read_text())
    assert digest(ROOT / 'src/analyses/road_service_loss_dynamic.py') == config['dynamic_kernel_sha256']
    assert digest(ROOT / 'src/analyses/road_service_loss.py') == config['reference_kernel_sha256']
    result = {'status': 'passed', 'scenario_authorization': receipt['scenario_authorization'],
              'national_complete': True, 'identity_preprocessing_exact': True,
              'all_source_roads_preserved': True, 'traffic_filter_applied': False,
              'readback': counts, 'original_meshes': receipt['original_meshes'],
              'original_population': receipt['original_population'],
              'positive_lost_path_road_scenarios': int(np.count_nonzero(lost)),
              'positive_additional_distance_road_scenarios': int(np.count_nonzero(burden)),
              'lost_path_scenarios_with_closed_entry_population_only': int(np.count_nonzero((lost > 0) & (lost == entry))),
              'lost_path_scenarios_exceeding_closed_entry_population': int(np.count_nonzero(lost > entry)),
              'closed_entry_diagnostic_sha256': digest(entry_path),
              'source_hashes_verified_unchanged': receipt['source_hashes_unchanged'],
              'interpretation': 'Counts refer to separate road-removal scenarios; they are not counts of distinct affected residents.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
