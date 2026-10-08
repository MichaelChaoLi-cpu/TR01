"""Prepare a reviewable population-only selection; no confirmed decisions or variables."""
import pandas as pd
from audit_census_age_aggregation import load,OUT as AUDIT
from build_vehicle_network_pilot import ROOT
OUT=ROOT/'data/exp/data-preprocessing'

def main():
    common={'KEY_CODE':'Mesh Code','HTKSYORI':'Suppression Status','HTKSAKI':'Aggregation Recipient Mesh','GASSAN':'Contributing Suppressed Meshes'}
    selected={'T001140':dict(common,T001140001='Local Total Population',T001140019='Published Population Aged 65+',T001140022='Published Population Aged 75+'),'T001188':dict(common,T001188001='Local Total Population')}
    for i,start in enumerate(range(0,100,5)):
        selected['T001188'][f'T001188{4+3*i:03}']=f'Published Population Aged {start}-{start+4}' if start<95 else 'Published Population Aged 95+'
    inventory=[];proposal=[];parts=[]
    labels=pd.read_csv(ROOT/'data/exp/data-briefing/tables/source_variable_labels.csv',keep_default_na=False)
    for table in selected:
        frame,_=load(table);parts.append(f'## {table}\n\n| Original name | Proposed English name |\n|---|---|')
        for col in frame:
            row=dict(dataset=table,original_name=col,dtype=str(frame[col].dtype),null_pct=float(frame[col].eq('').mean()*100),special_symbol_pct=float(frame[col].isin(['*','Y','-']).mean()*100),feasibility_status='unknown',sample_values='; '.join(frame[col].head(3)),source_label=labels[(labels.dataset==table)&(labels.variable==col)].source_label.iloc[0],proposed_selected=col in selected[table])
            inventory.append(row)
            if col in selected[table]:
                proposal.append(dict(dataset=table,original_name=col,proposed_readable_name=selected[table][col],status='awaiting_confirmation',role='support identifier' if col in common else 'source count for candidate construction'))
                parts.append(f'| {col} | {selected[table][col]} |')
    pd.DataFrame(inventory).to_csv(OUT/'population_full_variable_inventory.csv',index=False)
    pd.DataFrame(proposal).to_csv(OUT/'population_variable_proposal.csv',index=False)
    text='''# Population variable proposal — awaiting confirmation

This proposal is not a confirmed selection and does not create parquet files or edit AnaSOP. The complete inventory contains all 123 original fields, grouped by the two source datasets, with dtype, empty-field percentage, special-symbol percentage, feasibility status, labels and examples. The bundled scanner catalogs existing direct tabular engineering artifacts in `variable_list.csv`; it cannot directly inspect the archived CP932 census tables. The population-only inventory supplements that scan through the previously validated archive reader.

Proposed scope: seven fields from the population/household table and 25 fields from the five-year-age table. Retain source population codes and age symbols; numeric views of `*` or `Y` stay missing with source flags. No filling, truncation, deletion, logarithm or imputed elderly location is proposed. Confirmed output names would exactly match the English names below.

Proposed accounting: retain all original meshes and the official suppression linkage. Construct age counts on the 146,540 verified publication-support groups. Recommend `Population Aged 65+ Share Among Known Ages` as the explicitly labeled main ageing measure; retain total support population and age-unknown counts so the denominator is transparent. Alternatively, choose the total-population denominator and label it accordingly. Neither choice imputes unknown ages.

Proposed accessibility linkage: keep distances at each member mesh, with candidate in-mesh entries separated from the original baseline. Preserve group age totals separately. Do not assign all elderly residents to the recipient mesh or distribute elderly counts proportionally as observed facts. The later group-to-accessibility analysis remains a research choice.

The 200 pending entry meshes stay flagged; no sample exclusion is proposed. This phase constructs demographic support and membership, not elderly-specific individual accessibility or maintenance priority.

Formal research objectives are not yet recorded in AnaSOP Sections 1–2, and no question-based feasibility check has been completed. The proposed variables are demographic groundwork; this does not authorize a maintenance model. After scope confirmation, reconcile the already chosen research objective through research-question-planning and feasibility-check before accepting analysis roles in Section 4.

Required parquet dependencies `pyarrow` and `openpyxl` are currently absent. No installation has been performed.

'''+ '\n'.join(parts)+'\n'
    (OUT/'population_preprocessing_proposal.md').write_text(text)
    print(f'Prepared {len(inventory)} source fields and {len(proposal)} proposed selections; no final variable selected.')

if __name__=='__main__':main()
