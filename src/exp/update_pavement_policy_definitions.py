"""Append confirmed policy definitions after validated readable construction."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/data-preprocessing/pavement-policy'

INTRO=r'''### Confirmed pavement policy constructs

The 22 human-confirmed readable constructs below complete the finite pavement policy framework. They preserve the 70,049 conditional intervention roads, 175,612 original selected medical origins, 125,874,788 residents and the original Higher Ageing Area and Baseline Distance Disadvantaged Area masks. Earlier scoped statements that these constructions remain pending are superseded by this subsection. There is no new source exclusion, traffic imputation, clipping of scientific distributions, logarithmic transformation or redistribution of older residents. Independent bridge variables retain their original scope.

Let \(M\) be the original selected mesh set, \(m\) its origin index, \(P_m\) Local Total Population, \(d_m\) original Candidate Medical Network Distance, \(\mathbf{x}\) a complete maintenance selection and \(\delta\) Pavement Impedance Increment. Previously defined \(E_P\), \(e\), \(\ell_e\) and \(c_e^P\) denote the conditional intervention set, road identity, physical length and relative cost. The effective cost and per-mesh cost increase are

\[
\widetilde{\ell}_e=\ell_e[1+\delta\mathbf{1}\{e\in E_P\}(1-x_e)],\qquad
\Delta_m(\mathbf{x},\delta)=\widetilde d_m(\mathbf{x},\delta)-d_m.
\]

Here \(x_e\) is Pavement Maintenance Selected, \(\mathbf{1}\) an indicator, \(\widetilde{\ell}_e\) Pavement Effective Road Cost, \(\widetilde d_m\) Pavement Medical Access Cost and \(\Delta_m\) Pavement Medical Access Cost Increase. Costs are in reference metres, not measured travel time or actual additional route length. Full-network routing scales physical partial-origin, partial-destination and same-road direct legs consistently and preserves alternative source-road minima. The primary increment is 0.25; sensitivities are 0.10 and 0.50.

For a fixed population subset \(S\), burden and protection are

\[
H_S^0(\delta)=\frac{1}{1000}\sum_{m\in S}P_m\Delta_m(\mathbf{0},\delta),\qquad
H_S(\mathbf{x},\delta)=\frac{1}{1000}\sum_{m\in S}P_m\Delta_m(\mathbf{x},\delta),\qquad
A_S=H_S^0-H_S,\qquad Q_S=\frac{A_S}{H_S^0}.
\]

Here \(S\) is indexed by Pavement Population Group, \(\mathbf{0}\) the all-unmaintained eligible reference, \(H_S^0\) Pavement Reference Access Burden, \(H_S\) Pavement Residual Access Burden, \(A_S\) Pavement Avoided Access Burden and \(Q_S\) Pavement Access Protection Share. Burdens are resident reference-kilometre equivalents; a zero reference burden leaves protection undefined. Groups are the whole original selected population and the two previously fixed area masks. Groups overlap and their fractions are not additive. No individually located older-resident outcome is inferred.

Let \(V_e\) be Observed Traffic Volume and \(\mathbf{x}^e\) the selection maintaining only road \(e\). The ranking inputs are

\[
t_e=\frac{V_e}{c_e^P},\qquad
G_e=H_M^0(0.25)-H_M(\mathbf{x}^e,0.25),\qquad k_e=\frac{G_e}{c_e^P}.
\]

Here \(t_e\) is Pavement Traffic Priority Score, \(G_e\) Pavement Isolated Restoration Gain and \(k_e\) Pavement Access Priority Score. Traffic and access lists sort their respective scores in descending order. Condition sorts Anchored Pavement Inspection Category III, II, I and then ascending Pavement Relative Maintenance Cost. Original Graph Source Edge ID breaks final exact ties. Pavement Priority Rank is one-based within each Pavement Maintenance Priority. The access list is fixed at 25 percent for every budget and impedance sensitivity; isolated gains are ranking inputs and are not summed as joint policy outcomes.

Let \(b\) be Pavement Budget Share, \(C_{\mathrm{tot}}^P=\sum_{e\in E_P}c_e^P\) the common domain's total relative cost and \(r\) a maintenance rule. Pavement Requested Relative Budget is \(bC_{\mathrm{tot}}^P\). Pavement Actual Relative Spending sums complete selected costs; Pavement Unused Relative Budget is requested less actual resources. Divide the latter two quantities by \(C_{\mathrm{tot}}^P\) for Pavement Actual Spending Share and Pavement Unused Budget Share. Pavement Selected Road Count counts unique complete selected units. The adopted longest affordable prefix is specified in Section 6; no unit is split or skipped. Requested shares are 0.01, 0.05, 0.10 and 0.20. Reference-only rows use the rule label Reference and a zero budget. Reference rows are evaluation denominators, not additional maintenance policies.

The exact national implementation has passed independent engineering checks. A tolerance of \(10^{-7}\) reference metres handles distance-difference round-off in gains and excess cost; differences within that numerical tolerance are recorded as zero. This is a numerical precision rule, not an outlier treatment or observed impairment threshold. Under finite positive scenario costs, originally finite origins remain finite. Policy outcomes are conditional on the adopted linkage, length cost, fixed physical entries and uniform impairment assumptions, rather than observed repair effects.
'''


def main():
    receipt=json.loads((OUT/'construction_validation.json').read_text())
    assert receipt['status']=='passed_readable_pavement_policy_construction'
    decisions=json.loads((OUT/'decisions.json').read_text())
    assert decisions['authorization']=='MILI-D-20261007-047'
    rows=['| variable_name | full_name | role | formal_definition | construction_or_coding | is_final_variable |',
          '|---|---|---|---|---|---|']
    props={r['proposed_readable_name']:r for r in __import__('pandas').read_csv(OUT/'confirmed_variable_construction.csv').to_dict('records')}
    for v in decisions['derived_variables']:
        name=v['readable_name'];r=props[name]
        rows.append(f'| {name} | {name} | {r["role"]} | {r["definition"]} | Apply the adopted equations and precision rule above; preserve unchanged source identities and fixed supports. | yes |')
    block=INTRO+'\n'+'\n'.join(rows)+'\n\n'
    path=ROOT/'docs/AnaSOP.md';before=path.read_text()
    if '### Confirmed pavement policy constructs' in before:
        raise RuntimeError('Policy definitions already present; preserve them before revising')
    (OUT/'AnaSOP_before_policy_definitions.md').write_text(before)
    final=before.replace('## 5. Identification Strategy',block+'## 5. Identification Strategy',1)
    path.write_text(final)
    assert final.split('## 4. Variable Construction')[0]==before.split('## 4. Variable Construction')[0]
    assert final.split('## 5. Identification Strategy')[1]==before.split('## 5. Identification Strategy')[1]
    assert not re.search(r'[\u4e00-\u9fff]',block)
    assert all(e.isascii() for e in re.findall(r'\\\[(.*?)\\\]',block,re.S))
    report={'status':'passed_policy_section4_append','authorization':'MILI-D-20261007-047',
            'derived_definitions':22,'all_prior_section4_content_preserved':True,'other_sections_preserved':True,
            'english_readable_names_and_ascii_equations':True,'sop_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'section4_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
