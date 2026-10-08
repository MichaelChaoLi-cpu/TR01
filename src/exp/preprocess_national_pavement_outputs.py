"""Use the skill generator, retaining its selection/rename code in bounded batches."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scope', choices=['inputs', 'policy', 'allocation'])
    args = parser.parse_args()
    scope = args.scope
    exp = ROOT/f'data/exp/data-preprocessing/national-pavement-{scope}'
    exp.mkdir(parents=True, exist_ok=True)
    source_dir = exp if scope == 'inputs' else ROOT/f'data/exp/national-pavement-{scope}-computation'
    stems = {'inputs':['national_pavement_inputs'], 'policy':[
        'national_pavement_rankings','national_pavement_selections',
        'national_pavement_mesh_scenarios','national_pavement_group_outcomes'],
        'allocation':['national_pavement_mesh_allocation','national_pavement_rule_overlap']}[scope]
    decisions = {'authorization':'MILI-D-20261008-001','datasets':{}}
    for stem in stems:
        source = source_dir/f'{stem}.parquet'
        columns = pq.ParquetFile(source).schema_arrow.names
        decisions['datasets'][str(source.relative_to(ROOT))] = {
            'output':f'data/processed/national-pavement-{scope}/{stem}_preprocessed.parquet',
            'script':f'src/preprocessing/preprocess_{stem}.py',
            'variables':[{'original_name':c,'readable_name':c,'full_name':c,
                          'is_final_variable':'yes','preprocessing':[]} for c in columns]}
    scoped = exp/'decisions.json'
    scoped.write_text(json.dumps(decisions,indent=2)+'\n')
    global_path = ROOT/'data/exp/data-preprocessing/decisions.json'
    global_decisions = json.loads(global_path.read_text()) if global_path.exists() else {'datasets':{}}
    global_decisions['datasets'].update(decisions['datasets'])
    global_path.write_text(json.dumps(global_decisions,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'.codex/skills/data-preprocessing/scripts/generate_preprocessing_script.py'),
                    '--root',str(ROOT),'--decisions',str(scoped.relative_to(ROOT)),'--pretty'],check=True)
    rows = {}
    for source,info in decisions['datasets'].items():
        script = ROOT/info['script']
        generated = script.read_text()
        (exp/(script.stem+'.generated.py')).write_text(generated)
        select = generated.split('# --- Select and rename columns ---\n')[1].split('# --- Save ---')[0]
        # No variable-level operations are requested: exactly the generated
        # selection/rename, with streaming as a storage implementation only.
        adapted = generated.split('# --- Load ---')[0]+'import pyarrow as pa\nimport pyarrow.parquet as pq\n'
        adapted += "dst.parent.mkdir(parents=True, exist_ok=True)\npart = dst.with_suffix('.part')\nwriter = None\ncount = 0\ntry:\n"
        adapted += "    for batch in pq.ParquetFile(src).iter_batches(batch_size=250000):\n        df = batch.to_pandas()\n"
        adapted += ''.join('        '+line+'\n' for line in select.strip().splitlines())
        adapted += "        table = pa.Table.from_pandas(df, preserve_index=False)\n        if writer is None:\n            writer = pq.ParquetWriter(part, table.schema, compression='zstd')\n        writer.write_table(table)\n        count += len(df)\nfinally:\n    if writer is not None:\n        writer.close()\npart.replace(dst)\nassert pq.ParquetFile(dst).metadata.num_rows == count\nprint(f'Saved {count:,} readable rows: {dst.relative_to(ROOT)}', flush=True)\n"
        script.write_text(adapted)
        subprocess.run([sys.executable,str(script)],check=True)
        srcfile,dstfile=pq.ParquetFile(ROOT/source),pq.ParquetFile(ROOT/info['output'])
        assert srcfile.metadata.num_rows==dstfile.metadata.num_rows
        assert srcfile.schema_arrow.names==dstfile.schema_arrow.names
        for left,right in zip(srcfile.iter_batches(batch_size=250000),dstfile.iter_batches(batch_size=250000)):
            assert left.equals(right, check_metadata=False), source
        rows[info['output']]=dstfile.metadata.num_rows
    (exp/'readable_output_validation.json').write_text(json.dumps({
        'status':'passed','authorization':decisions['authorization'],'row_counts':rows,
        'skill_generated_scripts_retained':True,'bounded_batch_adaptation':True,
        'all_values_read_back_equal':True,'extra_variable_operations':[]},indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
