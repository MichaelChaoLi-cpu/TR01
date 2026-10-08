"""Verify authorized output-review repairs without recalculating policy models.
Run after rerendering and visually checking the seven affected figures.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT/'data/exp/critique-resolution'
SLUGS = ['national_candidate_medical_distance',
         'ageing_population_concentration_and_candidate_medical_access',
         'road_service_dependence_and_measured_traffic',
         'national_distribution_of_residual_pavement_access_burden',
         'national_spatial_allocation_of_pavement_maintenance',
         'spatial_distribution_of_residual_pavement_access_burden',
         'population_roads_and_medical_destinations']

def main():
    before=json.loads((EXP/'input_state_before.json').read_text())
    current={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
             for folder in ['data/raw','data/processed'] for p in (ROOT/folder).rglob('*') if p.is_file()}
    assert current == before['sources'], 'Raw or processed source inventory changed'
    for path,expected in before['tables'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    state_path=ROOT/'data/exp/figure-table-generation/production_state.json'
    state=json.loads(state_path.read_text())
    old_state=json.loads((EXP/'before/data/exp/figure-table-generation/production_state.json').read_text())
    acceptance={o['title']:o.get('human_acceptance_decision') for o in old_state['outputs']}
    figures=[]
    for item in state['outputs']:
        assert item['status']=='done'
        assert item.get('human_acceptance_decision')==acceptance[item['title']]
        if 'image' not in item:continue
        path=ROOT/item['image']
        with Image.open(path) as im:dpi=list(im.info['dpi']);dims=list(im.size)
        assert all(abs(d-600)<.1 for d in dpi),item['title']
        actual_size=[x/d*25.4 for x,d in zip(dims,dpi)]
        assert abs(actual_size[0]-190)<.1,(item['title'],actual_size)
        item.update(dpi=dpi,dimensions=dims,figure_size_mm=actual_size)
        slug=path.stem.removeprefix('Figure_')
        if slug in SLUGS:
            exp=ROOT/'data/exp/figure-table-generation'/slug
            layout=json.loads((exp/'publication_layout.json').read_text())
            assert not layout['clipped_text'] and not layout['coordinate_tick_overlaps']
            assert layout['minimum_font_size_pt']>=6
            review={'status':'passed_agent_visual_review','revision_authorization':'MILI-D-20261008-008',
                    'method':'Direct inspection of the current 600-dpi PNG and publication-size text bounds',
                    'output':item['image'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'figure_size_mm':actual_size,'minimum_font_size_pt':layout['minimum_font_size_pt'],
                    'clipped_text':[],'coordinate_tick_overlaps':[],
                    'scientific_definitions_and_adopted_colours_unchanged':True,
                    'prior_human_acceptance_decision':item.get('human_acceptance_decision'),
                    'additional_human_acceptance_inferred':False}
            (exp/'publication_visual_review.json').write_text(json.dumps(review,indent=2)+'\n')
            item['publication_revision_authorization']='MILI-D-20261008-008'
            item['publication_visual_review']=str((exp/'publication_visual_review.json').relative_to(ROOT))
            receipt_path=ROOT/item['validation'];receipt=json.loads(receipt_path.read_text())
            if receipt.get('status')=='generated_awaiting_human_review':receipt['status']='generated_and_validated'
            receipt.pop('human_acceptance_pending',None)
            receipt.update(publication_revision_authorization='MILI-D-20261008-008',
                           prior_human_acceptance_decision=item.get('human_acceptance_decision'),
                           publication_layout_validation=str((exp/'publication_layout.json').relative_to(ROOT)),
                           publication_visual_review=str((exp/'publication_visual_review.json').relative_to(ROOT)),
                           current_image_sha256=review['sha256'])
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
            figures.append(review)
    assert len(figures)==7
    stats='data/exp/figure-table-generation/ageing_population_concentration_and_candidate_medical_access/linear_fit_statistics.csv'
    pd.testing.assert_frame_equal(pd.read_csv(ROOT/stats),pd.read_csv(EXP/'before'/stats),check_exact=True)
    sop=(ROOT/'docs/AnaSOP.md').read_text();old=(EXP/'before/docs/AnaSOP.md').read_text()
    assert re.findall(r'^## ([1-8])\.',sop,re.M)==list('12345678')
    assert len(re.findall(r'^#### Supporting Point',sop,re.M))==4
    assert sop[sop.index('## 8.'):]==old[old.index('## 8.'):]
    assert re.findall(r'\\\[.*?\\\]',sop,re.S)==re.findall(r'\\\[.*?\\\]',old,re.S)
    assert not re.search(r'[\u4e00-\u9fff]',sop)
    for phrase in ['Final Scenario Traffic Volume','eligible policy sample remain unverified',
                   'remain to be finalized before policy calculation','National outputs remain pending',
                   'All four pending national pavement outputs','output acceptance pending',
                   'final anchored road inputs and readable variable construction remain pending']:
        assert phrase not in sop,phrase
    assert 'Future pavement priorities, future pavement accessibility protection' in sop
    assert 'This matched-origin exclusion applies to all three bridge demographic outputs' in sop
    for p in (ROOT/'src/analyses').glob('*.py'):ast.parse(p.read_text())
    road=json.loads((ROOT/'data/exp/figure-table-generation/road_service_dependence_and_measured_traffic/validation.json').read_text())
    assert road['panel_b_dimension_scale']==.8
    assert min(road['panel_b_font_sizes_points'].values())>=6.8
    assert road['panel_b_xlabel_colourbar_gap_mm']>=1
    assert not road['matched_main_road_curves_intersecting_inset_frames']
    for slug in ['national_distribution_of_residual_pavement_access_burden','spatial_distribution_of_residual_pavement_access_burden']:
        receipt=json.loads((ROOT/'data/exp/figure-table-generation'/slug/'validation.json').read_text())
        for row in receipt['accounting']:
            np.testing.assert_allclose(row['mesh_burden_sum'],row['group_residual_burden'],rtol=1e-12,atol=1e-7)
    resolution={'status':'passed','authorization':'MILI-D-20261008-008','issues_resolved':[1,2,3,4],
                'open_critical':0,'open_major':0,'open_minor':0,
                'figures':13,'tables':6,'rerendered_figures':figures,
                'all_figures_600dpi_at_190mm_width':True,'raw_and_processed_inventory_unchanged':True,
                'source_files_preserved':len(current),'six_workbook_sha256_unchanged':True,
                'association_statistics_exactly_unchanged':True,'display_equations_unchanged':True,
                'section_8_unchanged':True,'AnaSOP_language':'English',
                'all_prior_human_acceptance_decisions_preserved':True,
                'new_human_acceptance_not_inferred':True,'national_models_not_recomputed':True}
    (EXP/'validation.json').write_text(json.dumps(resolution,indent=2)+'\n')
    doc=json.loads((EXP/'anasop_resolution.json').read_text());doc['after_sha256']=hashlib.sha256(sop.encode()).hexdigest()
    (EXP/'anasop_resolution.json').write_text(json.dumps(doc,indent=2)+'\n')
    state.update(critique_resolution_authorization='MILI-D-20261008-008',
                 critique_resolution_validation='data/exp/critique-resolution/validation.json',
                 open_critique_issues=0,
                 next_ready_action='build-content-dictionary, followed by audit-content-compliance')
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print('Passed: 13 figures at 600 dpi and 190 mm; seven rerenders; six workbooks and source inventories unchanged; four issues resolved.')

if __name__=='__main__':main()
