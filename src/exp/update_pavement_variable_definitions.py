"""Append approved Section 4 definitions through data-preprocessing workflow."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOP = ROOT / 'docs/AnaSOP.md'
EXP = ROOT / 'data/exp/data-preprocessing/pavement-inputs'

SECTION = r'''### Confirmed pavement common-sample inputs

This human-confirmed construction supersedes the earlier preconstruction status in the scoped source-overview notes. The adopted conditional sample contains 70,049 unique source-road units with 264,099 anchored inspection records across 47 prefectures. The rule requires an observed pavement grade I-III, a parsed national-route identity, a nearest traffic-matched full source curve no more than 50 metres away, exactly one candidate within the nearest distance plus 5 metres, and consistent national-route identity. Missing condition and traffic are not imputed. Expressway and other unresolved route identities remain retained outside this conditional sample. The rule establishes a reproducible locator scenario, not verified physical inspection-interval footprints or nationwide matching accuracy.

Let \(E_P\) denote this pavement common sample, \(e\) a source-road unit in \(E_P\), \(\mathcal{J}_e\) its set of anchored usable inspection records, and \(C_j^P\) the observed category of inspection record \(j\), coded 1, 2 and 3 for I, II and III. The adopted road-level marker is

\[
C_e^P=\max_{j\in\mathcal{J}_e}C_j^P.
\]

Here \(C_e^P\) is Anchored Pavement Inspection Category. It marks the worst observed anchored inspection category and is ordinal, not a cardinal risk or impairment score. It does not locate a defect on every portion of the source-road curve or calibrate the scenario impedance increment.

| variable_name | full_name | role | formal_definition | construction_or_coding | is_final_variable |
|---|---|---|---|---|---|
| Graph Source Edge ID | Graph Source Edge ID | pavement maintenance-unit identity | Unique original candidate-network source-road identity \(e\) in \(E_P\). | Retain one row per source-road unit. Multiple inspection records and prefecture labels do not create additional cost units. | yes |
| Baseline Source Road Length | Baseline Source Road Length | physical length and cost input | \(\ell_e\): original candidate-network length of source road \(e\), in metres. | Preserve the retained full-network value and numerical type. Do not replace it with the uniform 100-metre inspection-interval label. | yes |
| Anchored Pavement Inspection Category | Anchored Pavement Inspection Category | condition-priority input | \(C_e^P\): worst observed category among \(\mathcal{J}_e\). | Use I-III only in the common sample. Preserve source records and observed-grade range; do not infer complete road condition, failure probability or speed loss. | yes |
| Pavement Relative Maintenance Cost | Pavement Relative Maintenance Cost | pavement budget accounting | \(c_e^P=\ell_e\), where \(c_e^P\) is the length-proportional relative maintenance cost. | Copy Baseline Source Road Length. Assume a common relative cost per metre; width, work intensity and monetary expenditure are not established by this proxy. | yes |

Observed Traffic Volume retains its previously confirmed daytime 12-hour measured definition and readable name. The pavement domain shares 3,368 traffic observation units; repeated values on source-road fragments are not independent measurements. No unsurveyed-road traffic is inferred.

The following selected fields are retained for audit and reproducibility, rather than as additional maintenance objectives or explanatory variables.

| variable_name | full_name | role | formal_definition | construction_or_coding | is_final_variable |
|---|---|---|---|---|---|
| Anchored Pavement Inspection Record Count | Anchored Pavement Inspection Record Count | grade provenance | Number of usable anchored records in \(\mathcal{J}_e\). | Count original record identities; do not treat this count as road length or physical maintenance-unit count. | no |
| Minimum Anchored Pavement Inspection Category | Minimum Anchored Pavement Inspection Category | grade-range provenance | Lowest observed anchored category in \(\mathcal{J}_e\). | Preserve for audit; the primary marker remains the worst observed category. | no |
| Anchored Inspection Prefecture Count | Anchored Inspection Prefecture Count | administrative provenance | Number of distinct prefecture labels among anchored inspection records. | Retain the 22 units associated with records from multiple prefectures; count each road cost once. These labels do not establish verified interval boundaries. | no |
| Traffic Observation Key | Traffic Observation Key | measurement identity | Shared identity of the retained measured daytime traffic observation. | Preserve the accepted measurement key and repeated-unit structure. | no |
| Registry Group Index | Registry Group Index | correspondence provenance | Retained traffic-geometry registry identity. | Preserve the established correspondence reference; it is not an additional traffic observation. | no |
| N13 Road Level | N13 Road Level | road-source descriptor | Retained source-road level category. | Preserve original values; do not infer pavement condition from this descriptor. | no |
| N13 Road State | N13 Road State | road-source descriptor | Retained source-road state category. | Preserve original values; do not treat it as a measured maintenance effect. | no |
| Candidate Road Source Record | Candidate Road Source Record | source identity provenance | Retained original source-road record identity. | Preserve the accepted correspondence reference alongside Graph Source Edge ID. | no |
| Mixed Anchored Pavement Inspection Grades | Mixed Anchored Pavement Inspection Grades | grade heterogeneity flag | True when minimum and maximum anchored categories differ. | Preserve the Boolean indicator; 23,143 road units have mixed observed categories. It does not measure whole-road defect variance. | no |

The saved input contains all 70,049 adopted units and 14 readable fields: 13 retained input/provenance fields plus the length-cost derivative. The exactly redundant second road-ID column is omitted only from the processed view and remains in the source audit. No additional exclusion, imputation, clipping, logarithmic transformation or source-type conversion is applied. Grade counts are 25,582 I, 31,129 II and 13,338 III. Total unique length-proportional cost is 14,883,983.799254805 relative units, equivalent to 14,883.984 kilometres of source-road length under the assumed common cost per metre. Requested budgets of 1, 5, 10 and 20 percent use this pavement-specific denominator; they are neither facility-count shares nor bridge-domain budgets.

Independent read-back reconstructs the anchor eligibility, inspection-count aggregation and worst category from retained point diagnostics, reproduces every selected source field and dtype, and verifies costs against original full-network lengths. Raw sources, all excluded inspection records, baseline population and medical constructions, and independent bridge definitions are preserved. Pavement impedance, protection outcomes and priority algorithms remain to be finalized before policy calculation; existing bridge outcome names must not be reused for the finite pavement scenario.
'''


def sections(text):
    return {int(m.group(1)):m.group(0) for m in re.finditer(
        r'^## ([1-8])\. .*?(?=^## [1-8]\. |\Z)',text,re.M|re.S)}


def main():
    v = json.loads((EXP/'construction_validation.json').read_text())
    assert v['status']=='passed_pavement_common_input_construction'
    before = SOP.read_text()
    s = sections(before)
    old4 = s[4]
    header='### Confirmed pavement common-sample inputs\n'
    if header in old4:
        old4=re.sub('^'+re.escape(header)+r'.*?(?=^### |\Z)','',old4,flags=re.M|re.S)
    else:
        (EXP/'anasop_before_variable_update.md').write_text(before)
    new4 = old4.rstrip()+'\n\n'+SECTION.rstrip()+'\n\n'
    final = before.replace(s[4],new4)
    SOP.write_text(final)
    after=sections(final)
    assert all(s[n]==after[n] for n in s if n!=4)
    assert all(line in new4 for line in old4.rstrip().splitlines())
    assert not re.search(r'[\u4e00-\u9fff]',new4)
    assert str(ROOT) not in new4 and 'data/exp/' not in new4 and 'maximum_observed_grade' not in new4
    result={'status':'passed_pavement_variable_definition_update','authorization':'MILI-D-20261007-044',
        'section_updated':4,'other_sections_preserved_exactly':True,
        'all_previous_section4_definitions_preserved':True,'readable_names_and_english_content':True,
        'new_final_definitions':4,'reused_observed_traffic_definition':True,
        'provenance_definitions':9,'no_policy_outcomes_declared_final':True,
        'sop_sha256_at_variable_update':hashlib.sha256(SOP.read_bytes()).hexdigest()}
    (EXP/'anasop_update_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
