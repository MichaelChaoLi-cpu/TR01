"""Audit national proposal coverage against original graph source-road classes."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / 'data/processed/national-network-rule-validation'
CACHE = ROOT / 'data/processed/national-network-quality/tiles'
LABELS = {1: 'National road', 2: 'Prefectural road', 3: 'Municipal road',
          4: 'Expressway', 5: 'Other road', 6: 'Unknown road category'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    args = parser.parse_args()
    out = ROOT / args.candidates
    config = json.loads((out / 'configuration.json').read_text())
    complete = json.loads((out / 'validation.json').read_text())
    assert complete['all_national_tiles'] and config['registry_mode'] == 'complete_acquisition_registry'
    registry_path = ROOT / config['registry_path']
    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == config['registry_sha256']
    registry = pd.read_parquet(registry_path)
    proposed = pd.read_parquet(out / 'proposed_unique_crosswalk.parquet')
    ambiguous = pd.read_parquet(out / 'ambiguous_crosswalk.parquet')
    offsets = np.load(GRAPH / 'tile_offsets.npy')
    ordinals = np.load(GRAPH / 'edge_source_ordinals.npy', mmap_mode='r')
    classes = np.empty(len(ordinals), dtype=np.uint8)
    cache_hashes = {}
    for tile, start, count in offsets:
        tile, start, count = int(tile), int(start), int(count)
        path = CACHE / f'{tile}.npz'
        with path.open('rb') as handle:
            cache_hashes[str(path.relative_to(ROOT))] = hashlib.file_digest(handle, 'sha256').hexdigest()
        with np.load(path) as arrays:
            np.testing.assert_array_equal(arrays['fids'], ordinals[start:start + count])
            assert len(arrays['classes']) == count
            classes[start:start + count] = arrays['classes']
    assert np.isin(classes, list(LABELS)).all()
    ids = proposed['Graph Source Edge ID'].to_numpy(dtype=np.int64)
    ai = ambiguous['Graph Source Edge ID'].drop_duplicates().to_numpy(dtype=np.int64)
    assert len(np.unique(ids)) == len(ids) and not np.isin(ids, ai).any()
    target = registry['Target N13 Road Class'].to_numpy()[proposed['Registry Group Index'].to_numpy()]
    np.testing.assert_array_equal(classes[ids], target)
    totals = np.bincount(classes, minlength=7)
    matched = np.bincount(classes[ids], minlength=7)
    ambiguous_counts = np.bincount(classes[ai], minlength=7)
    rows = [{'N13 Road Class': code, 'Road Category': label,
             'All Graph Source Roads': int(totals[code]),
             'Unique Proposed Source Roads': int(matched[code]),
             'Ambiguous Source Roads': int(ambiguous_counts[code]),
             'Source Roads Without Unique Proposal': int(totals[code] - matched[code])}
            for code, label in LABELS.items()]
    table = pd.DataFrame(rows)
    assert table['All Graph Source Roads'].sum() == len(ordinals)
    table.to_csv(out / 'proposal_road_class_coverage.csv', index=False)
    receipt = {'status': 'passed', 'all_national_tiles': True,
               'source_roads': len(ordinals), 'unique_proposals': len(ids),
               'source_road_proposal_coverage_percent': float(len(ids) / len(ordinals) * 100),
               'ambiguous_source_roads': len(ai), 'source_class_compatibility_exact': True,
               'source_ordinal_identity_exact': True, 'metadata_cache_hashes': cache_hashes,
               'accepted_correspondences': 0,
               'interpretation': 'Technical coverage audit of an unaccepted rule proposal; not a national matching-accuracy estimate.'}
    (out / 'proposal_coverage_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
