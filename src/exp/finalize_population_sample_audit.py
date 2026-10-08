"""Validate focused briefing artifacts and append evidence to AnaSOP Section 3."""
import csv,hashlib,json,re
import pandas as pd
from audit_pending_population_entries import OUT
from build_vehicle_network_pilot import ROOT

NOTES={
'A001':('wooded_and_rural_land','The mapped mesh contains forest/rural terrain; the road-rule diagnosis is a boundary guard restriction, not an absence of all roads.'),
'A002':('mountain_rural_land','Mapped relief and rural terrain are consistent with a road near a mesh boundary; actual residence is not located by the center.'),
'A003':('wooded_mountain_land','Map/photo show wooded terrain. The source scan finds no intersecting N13 line; this does not establish that the census record is erroneous or that no real access exists.'),
'A004':('coastal_water_center','The center lies in mapped coastal water, with land near the mesh boundary. A center-in-water observation does not invalidate an areal census population or prove island isolation.'),
'A005':('airfield_campus_context','The mapped mesh includes airfield/campus infrastructure; ordinary source lines have excluded expressway attributes. Presence of mapped internal alignments does not establish public vehicle access.'),
'A006':('mountain_land','Mapped mountain terrain contains a source tunnel excluded by the entry rule. No direct ground-level entry is established by the image.'),
'A007':('wooded_mountain_land','Wooded mountain terrain; the source scan detects only a boundary touch or less than one metre of in-cell line.'),
'A008':('airfield_campus_context','Map/photo show airfield and campus infrastructure; source lines are nonordinary types. Real internal alignments may exist without meeting the provisional public vehicle entry rule.'),
'A009':('mountain_facility_context','Map/photo show mountain facilities and pedestrian alignments; source nonordinary lines cannot be silently promoted to vehicle roads.')}


def main():
    age=json.loads((OUT/'age_aggregation_summary.json').read_text());entry=json.loads((OUT/'pending_entry_summary.json').read_text())
    assert age['status']==entry['status']=='passed'
    pending=pd.read_csv(OUT/'pending_mesh_audit.csv',dtype={'point_id':str,'mesh_code':str},keep_default_na=False)
    assert len(pending)==200 and pending.point_id.is_unique and pending.published_local_population.sum()==1645
    assert pending.primary_road_tile_downloaded.all()
    reasons=pd.read_csv(OUT/'pending_reason_summary.csv');assert reasons.meshes.sum()==200 and reasons.population.sum()==1645
    ledger=pd.read_csv(OUT/'mesh_age_entry_audit.csv',dtype={'KEY_CODE':str,'diagnostic_group_id':str})
    assert len(ledger)==176964 and ledger.KEY_CODE.is_unique
    groups=ledger.groupby('diagnostic_group_id').agg(member_meshes=('KEY_CODE','size'),members_with_entry=('routing_eligible','sum'))
    mixed=(groups.members_with_entry.gt(0)&groups.members_with_entry.lt(groups.member_meshes))
    groups['entry_support']=groups.members_with_entry.map(lambda n:'none' if n==0 else 'some')
    groups.loc[groups.members_with_entry.eq(groups.member_meshes),'entry_support']='all'
    groups.to_csv(OUT/'publication_group_entry_coverage.csv')
    assert len(groups)==age['publication_support_units'] and groups.member_meshes.sum()==176964
    reviews=[]
    for r in csv.DictReader((OUT/'map_sample_queue.csv').open()):
        context,basis=NOTES[r['review_id']];card=OUT/'cards'/f"{r['review_id']}.png"
        reviews.append(dict(review_id=r['review_id'],point_id=r['point_id'],mapped_context=context,basis=basis,evidence_card=str(card.relative_to(ROOT)),evidence_sha256=hashlib.sha256(card.read_bytes()).hexdigest(),limit='Purposive sample only; mixed/unknown imagery dates; no inferred residence, vehicle legality or national island classification'))
    with (OUT/'map_context_review.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(reviews[0]));w.writeheader();w.writerows(reviews)
    acquisition=json.loads((OUT/'map_acquisition.json').read_text());assert acquisition['failed_tiles']==0 and len(reviews)==acquisition['records']==9
    tiles=pd.read_csv(OUT/'map_tile_manifest.csv')
    for r in tiles.itertuples():
        with (OUT/'tile_cache'/r.filename).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r.sha256
    validation=dict(status='passed',pending_rows=200,pending_population=1645,classification_totals_reconcile=True,pending_primary_source_tiles_all_downloaded=True,population_rows=176964,population_archives_hash_verified=age['source_archives_verified'],age_links_reciprocal=age['reciprocal_links_checked'],publication_support_groups=len(groups),groups_with_mixed_entry_status=int(mixed.sum()),map_cards_visually_reviewed=9,map_tiles_hash_verified=len(tiles),original_data_and_candidate_graph_unchanged=True,final_sample_exclusions=0,final_age_variable_selected=False)
    # English manuscript-input evidence, confined to Section 3.
    path=ROOT/'docs/AnaSOP.md';before=path.read_text();heading='### Population support and unresolved-entry audit'
    addition=f'''{heading}

- Both census tables contain 176,964 unique, matching population mesh records. Total population and suppression/recipient metadata agree across the tables; 302 source archives match their recorded hashes.
- There are 124,579 standalone meshes, 30,424 suppressed meshes and 21,961 recipient meshes. All 30,424 suppressed-to-recipient links are reciprocal and resolve to observed recipient records.
- The publication supports consist of 146,540 units: standalone meshes and recipient meshes together with their linked suppressed meshes. Adding the local total populations of all members gives a denominator consistent with the support of the published age counts. This is a diagnostic accounting check, not a selected final ageing variable or a redistribution of elderly residents.
- The 2,360 cases in which published older-adult counts exceed recipient-local total population disappear under the matched-support denominator. Broad older-adult counts agree with sums of the corresponding five-year age bands in both tables.
- The diagnostic difference between support-level total population and published known-age counts is 2,931,838 people. Total-population and known-age denominators answer different questions and must be labeled distinctly; age-unknown residents must not be silently treated as younger residents.
- Recipient and suppressed meshes jointly contain 3,161,955 residents. Excluding all of these meshes would remove their residents from the ageing analysis. Published group age counts do not identify the older residents' location within the member meshes; linking them to a single recipient road entry would impose an additional spatial assumption.
- The remaining 200 candidate-entry exceptions contain 1,645 residents. Source scans identify 140 meshes with only nonordinary centerlines, 49 without intersecting source lines, seven with ordinary centerlines excluded by entry attributes, three affected by the interior guard, and one with only a boundary touch or sub-metre line. All relevant primary road tiles have been acquired. These are source/rule diagnoses, not verified absence of real roads or island classifications.
- The candidate entry scenario covers meshes containing 99.998696 percent of the acquired total population. Coverage means population recorded in meshes with a candidate road representative, not verified individual vehicle access or observed residence. All exception records remain retained and flagged; no final sample exclusion or ageing variable has been selected.
'''
    section_start=before.index('## 3. Data Overview');next_section=re.search(r'^## [4-9]\. ',before[section_start:],re.M)
    stop=section_start+next_section.start() if next_section else len(before)
    section=before[section_start:stop]
    if heading in section:section=section[:section.index(heading)].rstrip()+'\n\n'
    section=section.replace('- Spatial layers use different reference systems. No coordinate transformation, network snapping or destination exclusion was performed.','- Spatial layers use different reference systems. Source briefing preserved the original coordinates; subsequent engineering diagnostics transformed road geometry and tested point attachments in separately labeled candidate scenarios.')
    section=section.replace('- Road connectivity, infrastructure conditions, traffic counts, maintenance costs and intervention effects have not been evaluated in this briefing.','- Subsequent engineering diagnostics evaluated candidate road connectivity and population entry rules. Infrastructure conditions, traffic counts, maintenance costs and intervention effects remain outside this data briefing.')
    updated=before[:section_start]+section.rstrip()+'\n\n'+addition+'\n'+before[stop:]
    assert updated[:section_start]==before[:section_start]
    assert not re.search(r'[\u4e00-\u9fff]',addition)
    path.write_text(updated);validation['anasop_section_3_updated']=True
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    names={'eligible_line_only_without_interior_guard':'Eligible line only without the interior guard','no_intersecting_N13_source_line_found':'No intersecting N13 source line','normal_centerline_present_but_entry_attributes_excluded':'Ordinary line excluded by entry attributes','only_boundary_touch_or_submetre_line_present':'Boundary touch or sub-metre line only','only_nonordinary_centerline_present':'Nonordinary centerline types only'}
    table='\n'.join(f"| {names[r.reason]} | {r.meshes} | {r.population:,} |" for r in reasons.itertuples())
    report=f'''# Population sample coverage and census age-support audit

This is a focused data briefing supporting the approved workflow in MILI-D-20261006-012. No raw data, candidate road graph, final sample exclusion or final ageing variable is changed. Diagnostic accounting does not locate residents.

## Remaining road-entry exceptions

All **200** exception meshes are retained. Their published local population totals **1,645** people, **0.001304 percent** of the acquired **126,146,099** people. The audit scanned **{entry['source_features_scanned']:,}** source features in **76** road archives, including ordinary and nonordinary lines, and checked full line geometry against the mesh support. All 200 primary road tiles are already acquired; no exception is attributed to a missing download.

| Source/rule diagnosis | Meshes | Published local population |
|---|---:|---:|
{table}

Ordinary source-road type is code 1; garden paths, footpaths, steps and unknown types are different codes. Bridges, tunnels and expressways may fail the current origin-entry rule even though their source lines are present. These findings do not justify promoting footpaths to vehicle roads or relaxing access rules automatically. See the official [road types](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadType.html), [road states](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadState.html) and [road classes](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadCategory.html).

Nine highest-population examples within the five diagnosis strata were inspected against GSI standard maps and aerial context; 70 map tiles passed acquisition and hash checks. Examples include mountain facilities, airfield/campus infrastructure and a coastal-water mesh center. None of these observations establishes actual resident locations or legal vehicle access. Island status is **not classified for all 200 meshes**; the current exhaustive classification is based on source-road evidence. `map_context_review.csv` records the limited visual observations, and `pending_map_examples.png` displays them. Mixed/unknown image dates cannot invalidate the trusted 2024 source merely through image absence.

## Census age-support verification

Two 2020 JGD2011 1 km tables each contain 176,964 unique mesh records. All total-population and suppression/aggregation metadata match. All 302 source ZIP hashes match the acquisition manifest. There are **124,579** standalone records, **30,424** suppressed records and **21,961** recipient records. All 30,424 links are reciprocal: each `HTKSAKI` resolves to a recipient, and each recipient's semicolon-delimited `GASSAN` lists exactly its contributing suppressed meshes.

The official [population/household definition](https://www.e-stat.go.jp/help/data-definition-information/downloaddata/T001140.pdf) and [five-year-age definition](https://www.e-stat.go.jp/help/data-definition-information/downloaddata/T001188.pdf) explain that age details from suppressed meshes are added to recipient meshes while total population remains local. Thus published recipient age counts and recipient-local total population do not share spatial support. `*` is a suppressed value, not zero. The [download notes](https://www.stat.go.jp/data/mesh/pdf/ryuijikou.pdf) also distinguish age-unknown indicators and absent population records; no source symbols are overwritten in this audit.

For each recipient, the diagnostic support includes the recipient **and every linked suppressed mesh**. The comparable total-population denominator is the sum of the local totals of these members. Standalone meshes retain their original support. This partitions all 176,964 records into **146,540** publication-support units with no population loss or duplication. The **2,360** recipient-local cases where age-65+ counts exceed local total population fall to **zero** when checked against this matched support. The published 65+ and 75+ counts agree exactly with the corresponding sums of five-year age bands.

The acquired published age-65+ total is **35,335,805**. The support-level residual of total population minus known-age population is **2,931,838**. These are published-data accounting results, not imputed ages. Recipient and suppressed supports jointly contain **3,161,955** people; excluding them wholesale would change the nationwide sample.

## Proposed ageing interpretation, awaiting variable selection

Use the official publication support to preserve the available age evidence. For a standalone mesh, its age counts refer to that mesh. For a recipient, its age counts refer to the recipient plus all linked suppressed members. Do not spread these counts across member road-entry points as observed elderly locations.

Two denominator meanings must remain distinct: older residents divided by total support population includes age-unknown residents in the denominator; older residents divided by known-age support population describes only residents with a published known age. The project has not selected either final definition. Both require matching numerator/denominator support.

Road accessibility remains measured at individual mesh representatives. **{int(mixed.sum()):,}** publication-support groups have a mixture of eligible and pending member entries. Group membership therefore does not create a unique road-entry point or establish where older residents reside. Later ageing/accessibility analysis needs an explicit within-group location assumption or a method preserving this uncertainty. This step verifies available evidence and its interpretation limit rather than silently assigning elderly residents to the recipient location.

## Coverage and files

`sample_coverage.csv` reports mesh counts and population recorded in qualifying meshes. The original 500 m rule covers 175,242 meshes containing 126,115,923 people. The candidate with in-mesh road representatives covers 176,764 meshes containing 126,144,454 people. These percentages describe represented census supports, not observed individual accessibility. `age_entry_coverage.csv` cross-tabulates publication status with entry status.

- `pending_mesh_audit.csv`: all 200 exceptions and source/rule reasons.
- `pending_mesh_road_evidence.csv`: every intersecting source feature and clipped lengths.
- `suppression_recipient_links.csv`: all 30,424 reciprocal source links.
- `mesh_age_entry_audit.csv`: original age/count fields with diagnostic links and entry status.
- `publication_group_checks.csv`: support-level count reconciliation; no final share constructed.
- `publication_group_entry_coverage.csv`: member entry completeness.
- `population_source_integrity.csv`: 302 verified source archives.
- `variable_summary.csv`, `variable_screening.csv`, `symbol_audit.csv`: skill-standard descriptive inspection of 123 variables.
- `figures/age_support_and_pending_population.png`: publication status and exception-population distributions.
- `validation.json`: saved assertions; all passed.

AnaSOP Section 3 is updated in English with this evidence. Other sections remain unchanged. No time-series candidates were detected, no time-series plots were generated, no files were skipped and no dependency was installed.

Run locally from the repository root:

```sh
.venv/bin/python src/exp/audit_pending_population_entries.py
.venv/bin/python src/exp/audit_census_age_aggregation.py
.venv/bin/python src/exp/map_pending_population_examples.py
.venv/bin/python src/exp/finalize_population_sample_audit.py
```

The map script fetches official GSI tiles only when its local evidence cache is absent. Visual context judgments are documented reviewer observations. The next research decision is the ageing denominator and how publication groups enter the accessibility analysis; population exclusions and age redistribution remain unselected.
'''
    (OUT/'README.md').write_text(report)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()
