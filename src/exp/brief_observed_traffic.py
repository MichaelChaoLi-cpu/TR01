"""Reproduce the scoped briefing with the skill's portable bundled script."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/exp/data-briefing/observed-traffic'


def main():
    df = pd.read_parquet(ROOT/'data/processed/observed-traffic/national_observed_traffic_preprocessed.parquet')
    units = df[df['Traffic Measurement Eligible']].drop_duplicates('Traffic Observation Key')
    with tempfile.TemporaryDirectory(prefix='tr01-traffic-briefing-') as directory:
        staging = Path(directory)
        (staging/'data/processed').mkdir(parents=True)
        units[['Observed Traffic Volume']].to_parquet(staging/'data/processed/traffic_observation_units.parquet', index=False)
        subprocess.run([sys.executable, str(ROOT/'.codex/skills/data-briefing/scripts/create_data_briefing.py'),
                        '--root', str(staging), '--max-distribution-plots', '2', '--pretty'], check=True)
        shutil.copytree(staging/'data/exp/data-briefing', OUT, dirs_exist_ok=True)
    readme = OUT/'README.md'
    readme.write_text(readme.read_text() + '\nScope: one measured daytime total per distinct eligible observation unit. '
                      'The complete source inventory retains 99,417 section rows and 30 selected or derived fields. '
                      'The one-column briefing view contains 26,217 observations, avoiding duplicate counting of shared measurements. '
                      'Temporary source paths in generated tables describe that isolated view, not a new authoritative dataset. '
                      'Reproduce from the processed national traffic table with `uv run python src/exp/brief_observed_traffic.py`. '
                      'The isolated root prevents the bundled script from replacing the existing population/facility Data Overview.\n')


if __name__ == '__main__': main()
