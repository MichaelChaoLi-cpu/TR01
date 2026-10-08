"""Portable read-only asset lookup; optionally print all retained candidate pairs."""
import argparse
import json
import sqlite3

import numpy as np

from build_national_bridge_candidates import OUT, ARRAY_FIELDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--facility-id', required=True)
    parser.add_argument('--full-pairs', action='store_true')
    args = parser.parse_args()
    with sqlite3.connect(f'file:{OUT / "candidate_summary.sqlite"}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute('SELECT * FROM bridge_summary WHERE shisetsu_id=?', (args.facility_id,)).fetchone()
        if row is None:
            raise SystemExit('Facility ID not found in the retained inventory')
        result = dict(row)
    if args.full_pairs:
        pairs = []
        for path in sorted((OUT / 'tiles').glob('*.npz')):
            with np.load(path) as data:
                selected = np.flatnonzero(data['bridge_index'] == int(result['bridge_index']))
                for index in selected:
                    candidate = {key: data[key][index].item() for key in ARRAY_FIELDS}
                    candidate['source_id'] = f'{path.stem}:{candidate["feature_ordinal"]}'
                    pairs.append(candidate)
        assert len(pairs) == int(result['candidates_within_200m'])
        result['full_pairs'] = sorted(pairs, key=lambda r: (r['distance_m'], r['source_id']))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
