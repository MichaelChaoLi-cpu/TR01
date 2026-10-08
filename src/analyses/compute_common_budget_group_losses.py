"""Exact affected-origin accounting for the adopted conditional policy sample.

Reuse the verified single-source-road kernel, fixed baseline and witness screen.
Group labels concern publication-support areas, not individual older residents.
"""
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analyses'))
from road_service_loss import RoadServiceLoss
from road_service_loss_dynamic import DynamicRoadServiceLoss

OUT = ROOT / 'data/exp/common-budget-group-loss'
PROC = ROOT / 'data/processed/national-network-rule-validation'
CACHE = ROOT / 'data/exp/road-service-loss-validation/dynamic'
GROUPS = {'Higher Ageing': 'Higher Ageing Area', 'Distance Disadvantaged': 'Baseline Distance Disadvantaged Area'}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    part = path.with_suffix('.part')
    part.write_text(json.dumps(value, indent=2) + '\n')
    part.replace(path)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    units = pd.read_parquet(ROOT / 'data/processed/common-budget/conditional_maintenance_units_preprocessed.parquet')
    sample = pd.read_parquet(ROOT / 'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet')
    original = pd.read_parquet(ROOT / 'data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet')
    assert sample['Mesh Code'].equals(original['Mesh Code'])
    assert sample['Local Total Population'].equals(original['Local Total Population'])
    validation = json.loads((CACHE / 'validation.json').read_text())
    hashes = validation['source_hashes_verified_unchanged'].copy()
    for rel in ['data/processed/common-budget/conditional_maintenance_units_preprocessed.parquet',
                'data/processed/common-budget/conditional_mesh_area_groups_preprocessed.parquet']:
        hashes[rel] = digest(ROOT / rel)
    for rel, expected in hashes.items():
        assert digest(ROOT / rel) == expected, rel
    config = {'authorization': 'MILI-D-20261007-026', 'source_hashes': hashes,
              'kernel_sha256': digest(ROOT / 'src/analyses/road_service_loss_dynamic.py'),
              'script_sha256': digest(Path(__file__)), 'group_columns': GROUPS}
    config_path = OUT / 'configuration.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Group input configuration changed.'
    else:
        atomic_json(config_path, config)
    points = pd.read_csv(ROOT / 'data/exp/national-population-entry/point_attachment_diagnostics.csv',
                        usecols=['point_id', 'dataset', 'category', 'routing_eligible', 'edge_id', 'along_m'],
                        dtype={'point_id': str, 'category': str}, keep_default_na=False,
                        low_memory=False, float_precision='round_trip')
    points['along_m'] = pd.to_numeric(points['along_m'], errors='coerce')
    origins = points.set_index('point_id').loc['mesh:' + sample['Mesh Code'].astype(str)]
    facilities = points[points['dataset'].eq('medical') & points['category'].isin(['1', '2']) & points['routing_eligible']]
    pairs = np.load(PROC / 'edge_nodes_exact.npy', mmap_mode='r')
    lengths = np.load(PROC / 'edge_lengths.npy', mmap_mode='r')
    ref = RoadServiceLoss(pairs, lengths, origins['edge_id'].to_numpy(), origins['along_m'].to_numpy(),
                         facilities['edge_id'].to_numpy(), facilities['along_m'].to_numpy(), graph=load_npz(PROC / 'graph_exact.npz'))
    del points, facilities, origins, original
    gc.collect()
    fast = DynamicRoadServiceLoss(ref, replacements=np.load(CACHE / 'source_removal_replacement_weights.npy', mmap_mode='r'))
    np.testing.assert_allclose(fast.baseline, sample['Candidate Medical Network Distance'], rtol=1e-11, atol=1e-7)
    critical = np.load(CACHE / 'critical_source_road_screen.npy', mmap_mode='r')
    pop = sample['Local Total Population'].to_numpy(dtype=np.int64)
    group_masks = {name: sample[col].to_numpy(dtype=bool) for name, col in GROUPS.items()}
    origin_order = np.argsort(ref.origin_edges, kind='stable')
    origin_edges = ref.origin_edges[origin_order]
    columns = ['Graph Source Edge ID'] + [f'{name} {o}' for name in GROUPS for o in ['Lost Medical Path Population', 'Additional Medical Distance Burden']]
    records = []
    max_lost_error = 0
    max_burden_error = 0.0
    recomputed = 0
    for first in range(0, len(units), 500):
        last = min(first + 500, len(units))
        part_units = units.iloc[first:last]
        path = OUT / f'batch_{first:06d}.parquet'
        if path.exists():
            frame = pd.read_parquet(path)
            np.testing.assert_array_equal(frame['Graph Source Edge ID'], part_units['Graph Source Edge ID'])
            records.append(frame)
            continue
        batch = []
        for _, unit in part_units.iterrows():
            edge = int(unit['Graph Source Edge ID'])
            if critical[edge]:
                ix, distances = fast.distances(edge, return_changes=True)
                recomputed += 1
            else:
                left, right = np.searchsorted(origin_edges, edge, side='left'), np.searchsorted(origin_edges, edge, side='right')
                ix = origin_order[left:right]
                distances = np.full(len(ix), np.inf)
            finite = np.isfinite(distances)
            lost_ix = ix[~finite]
            reach_ix = ix[finite]
            delta = distances[finite] - fast.baseline[reach_ix]
            assert (delta >= -1e-7).all()
            delta[np.abs(delta) <= 1e-7] = 0
            burdens = pop[reach_ix] * delta / 1000.0
            loss = int(pop[lost_ix].sum())
            burden = float(burdens.sum())
            loss_error = abs(loss - int(unit['Lost Medical Path Population']))
            burden_error = abs(burden - float(unit['Additional Medical Distance Burden']))
            max_lost_error = max(max_lost_error, loss_error)
            max_burden_error = max(max_burden_error, burden_error)
            assert loss_error == 0
            np.testing.assert_allclose(burden, unit['Additional Medical Distance Burden'], rtol=1e-10, atol=1e-8)
            row = {'Graph Source Edge ID': edge}
            for name, mask in group_masks.items():
                group_l = int(pop[lost_ix[mask[lost_ix]]].sum())
                complement_l = int(pop[lost_ix[~mask[lost_ix]]].sum())
                group_t = float(burdens[mask[reach_ix]].sum())
                complement_t = float(burdens[~mask[reach_ix]].sum())
                assert group_l + complement_l == loss
                np.testing.assert_allclose(group_t + complement_t, burden, rtol=1e-11, atol=1e-9)
                row[f'{name} Lost Medical Path Population'] = group_l
                row[f'{name} Additional Medical Distance Burden'] = group_t
            batch.append(row)
        frame = pd.DataFrame(batch, columns=columns)
        temp = path.with_suffix('.part')
        frame.to_parquet(temp, index=False)
        temp.replace(path)
        records.append(frame)
        atomic_json(OUT / 'progress.json', {'status': 'running', 'completed_units': last, 'total_units': len(units),
                    'elapsed_seconds': time.perf_counter() - started, 'recomputed_this_run': recomputed})
        print(f'Group accounting {last:,}/{len(units):,}; {time.perf_counter()-started:.1f} s', flush=True)
    frame = pd.concat(records, ignore_index=True)
    assert frame['Graph Source Edge ID'].is_unique and len(frame) == len(units)
    # Cache read-back retains exact IDs, integer losses and bounded group burdens.
    joined = units.merge(frame, on='Graph Source Edge ID', validate='one_to_one', sort=False)
    for name in GROUPS:
        assert joined[f'{name} Lost Medical Path Population'].between(0, joined['Lost Medical Path Population']).all()
        assert (joined[f'{name} Additional Medical Distance Burden'] <= joined['Additional Medical Distance Burden'] + 1e-8).all()
    joined.to_parquet(OUT / 'group_loss_construction.parquet', index=False)
    for rel, expected in hashes.items():
        assert digest(ROOT / rel) == expected
    v = {'status': 'passed', 'complete': True, 'units': len(frame), 'baseline_meshes': len(sample),
         'baseline_population': int(pop.sum()), 'all_group_complement_sums_checked': True,
         'every_recomputed_total_checked_against_accepted_national_loss': True,
         'maximum_integer_loss_error': max_lost_error, 'maximum_burden_error': max_burden_error,
         'source_hashes_unchanged': hashes, 'elapsed_seconds': time.perf_counter() - started,
         'groups_are_area_composition': True, 'age_counts_redistributed': False,
         'unique_beneficiary_claim': False, 'authorization': 'MILI-D-20261007-026'}
    atomic_json(OUT / 'validation.json', v)
    atomic_json(OUT / 'progress.json', {'status': 'complete', 'completed_units': len(frame), 'total_units': len(frame)})
    print(json.dumps(v, indent=2), flush=True)


if __name__ == '__main__':
    main()
