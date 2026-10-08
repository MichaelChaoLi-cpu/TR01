"""Validate and report saved visual judgments; no automatic maintenance units."""
import csv
import json
import math
from collections import Counter

from pyproj import Transformer

from prepare_stratified_bridge_review import ROOT, NATIONAL, OUT, digest, write_csv


def main():
    prep=json.loads((OUT/'preparation_validation.json').read_text())
    maps=json.loads((OUT/'map_validation.json').read_text())
    cases=json.loads((OUT/'review_cases.json').read_text())
    reviews=json.loads((OUT/'visual_judgments.json').read_text())
    assert digest(OUT/'review_cases.json') == prep['cases_sha256']
    assert digest(ROOT/'docs/AnaSOP.md') == prep['anasop_sha256']
    assert digest(NATIONAL/'validation.json') == prep['national_validation_sha256']
    assert len(cases)==len(reviews)==12
    by_id={c['review_id']:c for c in cases}
    assert len(by_id)==12 and {r['review_id'] for r in reviews}==set(by_id)
    assert maps['failed_tiles']==maps['missing_card_tiles']==0
    with (OUT/'tile_manifest.csv').open() as handle: tiles=list(csv.DictReader(handle))
    assert len(tiles)==maps['tiles']
    assert all(digest(OUT/'tile_cache'/r['file'])==r['sha256'] for r in tiles)
    audit=[]; outputs=[]; boundary=[]; partners=[]
    for r in reviews:
        c=by_id[r['review_id']]
        assert digest(OUT/'cards'/f'{c["review_id"]}.png')==maps['card_sha256'][c['review_id']]
        candidates={g['source_id']:g for g in c['candidate_geometries']}
        assert set(r['supported_candidate_source_ids'])<=set(candidates)
        assert bool(r['supported_candidate_source_ids'])==(r['association_status']!='withheld')
        outputs.append({'review_id':r['review_id'],'bridge_name':c['bridge_name'],'stratum':c['stratum'],
                        'source_prefecture':c['source_prefecture'],'nearest_source_id':c['nearest_source_id'],
                        'supported_candidate_source_ids':';'.join(r['supported_candidate_source_ids']),
                        'association_status':r['association_status'],'nearest_candidate_judgment':r['nearest_candidate_judgment'],
                        'observation':r['observation'],'extent_observation':r['extent_observation'],
                        'next_evidence':r['next_evidence'],'intervention_extent_status':'unresolved',
                        'accepted_maintenance_assignment':False})
        a=candidates[c['nearest_source_id']]
        tile=c['nearest_source_id'].split(':')[0]
        transform=Transformer.from_crs(6668,json.loads((NATIONAL/'tiles'/f'{tile}.json').read_text())['local_crs'],always_xy=True)
        ae=[transform.transform(*a['coordinates'][i]) for i in [0,-1]]
        for b in c['candidate_geometries'][1:8]:
            be=[transform.transform(*b['coordinates'][i]) for i in [0,-1]]
            gap=min(math.hypot(x[0]-y[0],x[1]-y[1]) for x in ae for y in be)
            audit.append({'review_id':c['review_id'],'nearest_source_id':a['source_id'],'other_source_id':b['source_id'],
                          'endpoint_gap_m':gap,'endpoints_coincident_within_0_01m':gap<=.01,
                          'nearest_level':a['road_level'],'other_level':b['road_level'],
                          'same_level_code':a['road_level']==b['road_level'],
                          'same_road_class':a['road_class']==b['road_class'],
                          'same_road_state':a['road_state']==b['road_state'],
                          'accepted_vehicle_connection':False})
        boundary.append({'review_id':c['review_id'],'bridge_length_m':c['bridge_length_m'],
                         'nearest_feature_length_m':c['nearest_source_length_m'],
                         'source_to_asset_length_ratio':c['source_to_asset_length_ratio'],
                         'point_along_nearest_feature_m':c['nearest_position_m'],
                         'point_distance_to_nearest_feature_endpoint_m':c['nearest_endpoint_distance_m'],
                         'full_affected_link_set_defined':False,'bridge_endpoints_verified':False})
        main={k:c[k] for k in ['bridge_name','bridge_length_m','nearest_position_m','source_longitude','source_latitude','shisetsu_id']}
        for i,p in enumerate([main]+c['partners']):
            if c['stratum']!='shared_feature': continue
            partners.append({'review_id':c['review_id'],'marker':'main' if i==0 else f'P{i}',
                             'shisetsu_id':p['shisetsu_id'],'bridge_name':p['bridge_name'],
                             'bridge_length_m':p['bridge_length_m'],'nearest_source_id':c['nearest_source_id'],
                             'point_along_shared_feature_m':p['nearest_position_m'],
                             'source_longitude':p['source_longitude'],'source_latitude':p['source_latitude'],
                             'accepted_shared_intervention':False})
    write_csv(OUT/'visual_review.csv',outputs)
    write_csv(OUT/'endpoint_context_audit.csv',audit)
    write_csv(OUT/'extent_evidence.csv',boundary)
    write_csv(OUT/'shared_source_positions.csv',partners)
    # Numeric endpoint coincidence is descriptive, not a universal joining tolerance.
    by_pair={(r['review_id'],r['other_source_id']):r for r in audit}
    for id, feature in [('S02','5235:81386'),('S07','5134:58397'),('S08','5540:220339'),('S09','4229:14888')]:
        assert by_pair[id,feature]['endpoints_coincident_within_0_01m']
    assert float(cases[0]['nearest_distance_m'])+5 < next(g['distance_m'] for g in cases[0]['candidate_geometries'] if g['source_id']=='6441:152853')
    statuses=dict(Counter(r['association_status'] for r in reviews))
    protocol={
        'status':'analyst_proposal_for_review_not_accepted_preprocessing_rule',
        'evidence_cases':[r['review_id'] for r in reviews],
        'purpose':'Separate corridor identity, vehicle relevance, physical asset bounds and affected graph links before maintenance allocation.',
        'coordinate_gate':'Retain original positions, source hashes and datum uncertainty. Withhold anomalous positions; no ID-derived replacement.',
        'function_gate':'Establish vehicle relevance of each physical component. Attachment, staircase or side-bridge names trigger manager review; never exclude or accept solely by name or width.',
        'identity_gate':'Inspect full candidate set, route/manager/level context and competing crossings. Nearest distance and nearest-plus-5 m are not identity or acceptance rules. Do not transfer condition to a lower or parallel road.',
        'junction_gate':'Distinguish connected source pieces and branches from separate corridors. Source endpoint coincidence does not establish legal connection, bridge identity or redundancy.',
        'extent_gate':'Obtain manager asset endpoints/component limits. Preserve the relationship of one asset to potentially multiple source features, and multiple assets to a source feature. Source length and inventory length alone are not footprint boundaries.',
        'graph_gate':'Trace verified deck intervals to all intersected graph links and any carriageway/approach coverage. Preserve asset IDs and graph/source provenance; review before splitting or modifying the candidate graph.',
        'shared_feature_gate':'A shared source feature is not a shared physical maintenance unit. Identify distinct deck intervals before costs or interventions can be counted.',
        'intervention_gate':'Define plausible affected functions separately from geometric linkage. No automatic universal closure, paired-carriageway failure, equal cost or maintenance-induced service restoration.',
        'outputs_required':'Reviewed asset/source relation; vehicle eligibility evidence; asset bounds; complete affected-link set; manager/year/carriageway provenance; unresolved flags; human approval before analysis-ready unit definition.',
        'accuracy':'This diagnostic sample has no verified ground truth or national accuracy estimate. A separate validation design is required for such a claim.'}
    (OUT/'intervention_unit_review_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    lines=['# Stratified bridge identity and extent review','',
           'This is an exploratory feasibility check under AnaSOP Sections 1-2. Twelve primary cases cover twelve prefectures and eight macroregions; five partner records add shared-feature context. Four mutually exclusive diagnostic strata contain three cases each. Within each stratum, a fixed SHA-256 rank selects one record per chosen region; region loads are balanced sequentially. Source-quality-flagged records and invalid coordinates are excluded from these sampling pools but remain retained in the national inventory. Region/pool sizes and the exact seed are saved in preparation_validation.json and sampling_pool_counts.csv. This is not a national probability sample or a ground-truth audit.','',
           '## Findings','',
           'Ten cases support only provisional corridor associations or candidate sets; two remain withheld. All twelve physical footprints and complete affected graph-link sets are unresolved. Accepted maintenance assignments remain zero. The three clearer-reference cases all still need physical boundary evidence.','',
           '- **Stacked roads:** S01 is nearest to a ground-level municipal road (2.07 m), while the named motorway corridor has a class-4 level-2 bridge candidate at 13.06 m. That candidate falls outside nearest-plus-5 m. S02 and S03 also require vertical and corridor distinctions. A wider retained candidate set is necessary; neither distance nor road level alone establishes identity.',
           '- **Physical components and vehicle function:** S04 names a staircase/attachment component and is offset from the main carriageway. S05 is a side bridge among adjacent crossing/path context with route-category conflict. Both are withheld for manager identity/function evidence; neither is automatically removed from the inventory.',
           '- **Shared source features:** S04 has seven assets nearest to a 466.1 m feature; S05 has two on a 298.5 m feature; S06 has three on a 2202.7 m feature. Partner markers and along-line positions retain the distinct assets. Whole-feature repairs cannot be duplicated as independent short-bridge interventions.',
           '- **Junction pieces:** In S02, S07 and S08, close source pieces have coincident endpoints; the map context shows continuous or branching geometry. Such candidates are not independently observed alternative routes. S09 has a bridge-coded piece adjoining the nearest ordinary-road approach. The 0.01 m numeric audit band reports endpoint coincidence only; it does not approve a graph-joining rule.',
           '- **Footprints:** S10 is a 69.8 m asset beside a 52.6 m bridge-coded feature; S12 is a 19.7 m asset beside a 64.3 m feature. Clear crossing context does not reconcile asset/source boundary definitions. S11 additionally needs carriageway and tunnel-transition evidence.','',
           '## Case ledger','',
           '| Case | Asset | Diagnostic stratum | Provisional association | Supported candidate IDs |',
           '| --- | --- | --- | --- | --- |']
    for r in reviews:
        c=by_id[r['review_id']]
        lines.append(f'| [{r["review_id"]}](cards/{r["review_id"]}.png) | {c["bridge_name"]} | {c["stratum"]} | {r["association_status"]} | {", ".join(r["supported_candidate_source_ids"]) or "withheld"} |')
    lines+=['','Supported IDs are analyst corridor proposals, not accepted asset or intervention assignments. visual_review.csv records the image observations, extent limits and specific next evidence for every case. endpoint_context_audit.csv distinguishes source endpoint geometry from accepted vehicle connections. extent_evidence.csv retains length and point-placement diagnostics. shared_source_positions.csv lists the distinct partners rather than merging them.','',
            '## Concrete review protocol','',
            '1. Confirm source position and physical component/vehicle relevance using manager evidence.',
            '2. Establish the correct corridor and carriageway from the complete candidate set, including vertical and parallel-road context.',
            '3. Obtain bridge endpoints and component limits; reconcile source-feature boundaries and shared-feature partners.',
            '4. Trace verified deck intervals to the complete affected graph-link set and preserve the asset-to-source-to-graph relation.',
            '5. Confirm intervention functions, costs and effects separately before policy ranking.','',
            'These steps are recorded as an analyst proposal in intervention_unit_review_protocol.json. No numerical acceptance radius, new decision-unit definition, graph split, closure rule, paired-carriageway intervention or equal-cost assumption is adopted here. The immediate next task is to obtain manager boundary/component evidence for the identified cases and test a traceable asset-to-graph mapping, while retaining withheld exceptions.','',
            '## Sources, limits and reproduction','',
            'The [N13 September 2024 catalog](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html) documents road centerlines, their attributes and vertical-order field. These are source geometries, not physical bridge footprints or a delivered node-link network. The [GSI tile list](https://maps.gsi.go.jp/development/ichiran.html) supplies standard cartography and seamless photography. Attribution appears on each card; overlays are added by this project. Map/image dates are mixed or unverified, selected inspections span 2021-2025, and bridge coordinate datum remains unverified. Neither map labels nor aerial visibility establishes current legal vehicle use, repair costs or intervention effects.','',
            'Run .venv/bin/python src/exp/prepare_stratified_bridge_review.py, then .venv/bin/python src/exp/render_stratified_bridge_maps.py, then .venv/bin/python src/exp/report_stratified_bridge_review.py. Preparation extracts full candidate/source geometries and verifies the hashed source archives and ordinals. The renderer fetches only missing bounded GSI tiles. visual_judgments.json was written after visual inspection and must be reinspected if card/source hashes change; report regeneration validates its references but cannot reperform visual judgment. No raw data, national candidate table or AnaSOP edit is performed. No new dependency, Git or DVC operation is included.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    result={'status':'passed','primary_cases':12,'context_partner_records':5,'sample_prefectures':12,'macroregions':8,
            'association_status_counts':statuses,'provisional_corridor_cases':sum(r['association_status']!='withheld' for r in reviews),
            'withheld_cases':sum(r['association_status']=='withheld' for r in reviews),
            'unresolved_intervention_extents':12,'accepted_maintenance_assignments':0,'national_accuracy_estimated':False,
            'source_geometry_candidate_pairs':prep['retained_sample_candidate_pairs'],'source_tiles_checked':prep['source_tiles_extracted'],
            'tile_hashes_verified':len(tiles),'card_hashes_verified':12,'supported_ids_exist_in_retained_candidates':True,
            'visual_judgments_sha256':digest(OUT/'visual_judgments.json'),'cases_sha256':prep['cases_sha256'],
            'raw_preserved':True,'national_candidate_table_unchanged':True,'anasop_unchanged':True,
            'anasop_sha256':prep['anasop_sha256'],'scope_revised':False}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
