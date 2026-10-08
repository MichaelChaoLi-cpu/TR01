"""Reproduce bundled selector plus the confirmed length-copy derivative."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISIONS = ROOT / 'data/exp/data-preprocessing/pavement-inputs/decisions.json'
GENERATOR = ROOT / '.codex/skills/data-preprocessing/scripts/generate_preprocessing_script.py'


def main():
    subprocess.run([sys.executable,str(GENERATOR),'--root',str(ROOT),
                    '--decisions',str(DECISIONS),'--pretty'],check=True)
    decisions=json.loads(DECISIONS.read_text())
    assert decisions['authorization']=='MILI-D-20261007-044'
    assert len(decisions['datasets'])==1
    source, dataset=next(iter(decisions['datasets'].items()))
    derived=dataset['derived_variables']
    assert len(derived)==1 and derived[0]['operator']=='copy'
    spec=importlib.util.spec_from_file_location('generator',GENERATOR)
    generator=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    extra={'original_name':'derived_pavement_relative_cost',**derived[0]}
    generator.validate_readable_names(source,dataset['variables']+[extra])
    label=json.dumps(derived[0]['readable_name'])
    base=json.dumps(derived[0]['source_readable_name'])
    block='\n'.join([
        '# --- Human-confirmed derivative and domain checks (MILI-D-20261007-044) ---',
        'assert len(df) == 70049 and df["Graph Source Edge ID"].is_unique',
        'assert df["Anchored Pavement Inspection Category"].isin([1, 2, 3]).all()',
        'assert df["Baseline Source Road Length"].gt(0).all()',
        'assert df["Observed Traffic Volume"].gt(0).all()',
        f'df[{label}] = df[{base}].copy()',
        'assert len(df.columns) == 14 and not df.isna().any().any()', '', ''])
    path=ROOT/dataset['script']
    text=path.read_text()
    assert text.count('# --- Save ---')==1
    path.write_text(text.replace('# --- Save ---',block+'# --- Save ---'))
    print('Confirmed derivative included: '+dataset['script'])


if __name__=='__main__':
    main()
