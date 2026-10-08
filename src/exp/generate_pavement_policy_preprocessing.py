"""Generate the four approved readable policy preprocessing scripts locally."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/exp/data-preprocessing/pavement-policy'
SOURCE=ROOT/'data/exp/pavement-policy-computation'


def main():
    assert json.loads((SOURCE/'validation.json').read_text())['full_policy_computation_completed']
    decisions=json.loads((BASE/'decisions.json').read_text())
    assert decisions['authorization']=='MILI-D-20261007-047'
    datasets={}
    for slug in ['pavement_rankings','pavement_selections','pavement_mesh_scenarios','pavement_group_outcomes']:
        source=SOURCE/(slug+'.parquet')
        cols=pq.ParquetFile(source).schema_arrow.names
        datasets[str(source.relative_to(ROOT))]={
            'output':f'data/processed/pavement-policy/{slug}_preprocessed.parquet',
            'script':f'src/preprocessing/preprocess_{slug}.py',
            'variables':[{'original_name':c,'readable_name':c,'full_name':c,'is_final_variable':'yes',
                          'preprocessing':[]} for c in cols]}
    decisions['datasets']=datasets
    decisions['selection_derivatives']=['Pavement Impedance Increment','Pavement Effective Road Cost']
    (BASE/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    generator=ROOT/'.codex/skills/data-preprocessing/scripts/generate_preprocessing_script.py'
    subprocess.run([sys.executable,str(generator),'--root',str(ROOT),'--decisions',
                    str((BASE/'decisions.json').relative_to(ROOT)),'--pretty'],check=True)
    path=ROOT/'src/preprocessing/preprocess_pavement_selections.py'
    content=path.read_text()
    extension='''# Approved baseline-increment cost derivative; complete selections stay fixed.
roads = pd.read_parquet(ROOT / "data/processed/pavement-common-sample/pavement_common_sample_preprocessed.parquet",
                        columns=["Graph Source Edge ID", "Pavement Relative Maintenance Cost"])
costs = df["Graph Source Edge ID"].map(roads.set_index("Graph Source Edge ID")["Pavement Relative Maintenance Cost"])
assert costs.notna().all()
df["Pavement Impedance Increment"] = 0.25
df["Pavement Effective Road Cost"] = costs * (1 + 0.25 * (~df["Pavement Maintenance Selected"]))
assert len(df) == 70049 * 12

'''
    assert content.count('# --- Save ---')==1
    path.write_text(content.replace('# --- Save ---',extension+'# --- Save ---'))
    module_spec=importlib.util.spec_from_file_location('pavement_name_check',generator)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.validate_readable_names('confirmed derived policy fields',[
        {'original_name':v['readable_name'],'readable_name':v['readable_name']}
        for v in decisions['derived_variables']])
    print('All 22 derived names validated; selection cost derivative inserted.')


if __name__=='__main__':main()
