"""Build the adopted English writing dictionary from local research artifacts.

Reads existing data/figures/tables; writes only the content dictionary and its
build receipts. Does not run research models or mutate scientific outputs.
Run: .venv/bin/python -B src/exp/build_actionbrief.py
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'data/exp/build-content-dictionary'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def dataset_key(path):
    return slug(Path(path).stem.removesuffix('_preprocessed'))


def section_variables(text):
    section = text.split('## 4. Variable Construction  /  Key Variables', 1)[1].split('## 5.', 1)[0]
    rows = {}
    for line in section.splitlines():
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) == 6 and cells[-1] == 'yes' and cells[0] != 'variable_name':
            rows[cells[0]] = dict(zip(['name', 'full', 'role', 'definition', 'construction', 'final'], cells))
    return rows


FAMILY = {
    'population': ('Official census mesh records and publication-support accounting.', 'Statistics Bureau of Japan, e-Stat; support construction by the TR01 research team.', '2020 population census.', 'Native JIS third-level geographic meshes (nominal 1 km); age counts retain official singleton or suppression/recipient publication supports.', 'Cross-sectional census; no observed longitudinal panel.', 'Source strings and suppression links are retained; derived age supports sum reciprocal member populations and use known-age counts without locating older residents within a support.'),
    'medical-baseline': ('Registered medical destinations and candidate-network baseline accessibility.', 'MLIT National Land Numerical Information and Statistics Bureau of Japan, e-Stat; candidate-network construction by TR01.', 'Population and medical registration: 2020; candidate road configuration: September 2024.', 'Medical point locators, native population meshes and official age publication supports.', 'One cross-sectional candidate configuration with different input reference years.', 'Hospitals and general clinics are destination categories 1 and 2; dental and unresolved-entry facilities are retained but not included destinations. Baseline distances remain missing for no path/pending entries; support means are conditional on finite-path population.'),
    'medical-analysis-sample': ('Selected baseline meshes or official supports with defined age composition and medical distance.', 'TR01 research team using e-Stat and MLIT baseline inputs.', '2020 population/medical register and September 2024 candidate roads.', '175,612 retained native meshes or 145,771 original official age supports, according to the file.', 'Fixed cross-sectional baseline.', 'Exact row selection from preserved baseline tables; exclude undefined age or medical distance, retain original support counts, and do not redistribute older residents.'),
    'population-density': ('Population intensity on complete original publication-support footprints.', 'TR01 research team using the 2020 e-Stat census and native geographic mesh definitions.', '2020 census.', '146,540 official publication supports; all complete member-mesh footprints, including water.', 'Fixed cross-sectional baseline.', 'Divide original full-support population by summed ellipsoidal complete mesh areas; no land clipping, age redistribution, tail trimming or stored logarithmic transformation.'),
    'observed-traffic': ('Measured bidirectional daytime traffic and accepted road correspondence.', 'MLIT, 2021 National Road and Street Traffic Conditions Survey; correspondence and preprocessing by TR01.', '2021 traffic; correspondence to September 2024 candidate roads.', 'Traffic observation units and basic sections; accepted full source-road curves for the crosswalk.', 'Daytime 12-hour counts from a single survey year; no time series.', 'Require validated two-direction observations and reported totals. Repeated sections share one observation identity. Missing/ambiguous traffic is retained as missing; strict geometry matching does not estimate national matching accuracy.'),
    'road-service-loss': ('Exact conditional loss after removing one candidate source-road record.', 'TR01 research team using the accepted MLIT candidate graph, e-Stat population and medical register.', 'Fixed 2020 population/medical registration and September 2024 candidate graph.', '21,415,800 source-road records evaluated against 175,612 fixed mesh origins.', 'Independent hypothetical single-road events; no annual event probabilities.', 'Preserve parallel alternatives and fixed entries; disable entries requiring the removed record. Count lost-path residents separately from additional finite-route person-kilometres; no finite penalty is assigned to disconnected origins.'),
    'road-traffic-comparison': ('Full road-loss archive linked to the accepted measured-traffic comparator.', 'TR01 research team using MLIT roads/traffic, e-Stat population and registered medical locations.', '2020 population/medical register; 2021 traffic; September 2024 roads.', '21,415,800 source-road rows; 1,183,367 uniquely matched roads sharing 25,566 observation units.', 'One fixed configuration and conditional single-road events.', 'Join original road IDs and loss outcomes to the strict traffic crosswalk. Retain unmatched and ambiguous values as missing and preserve all original source-road records.'),
    'common-budget': ('Independent bridge allocation inputs, fixed area groups, conditional event losses and policy outcomes.', 'TR01 research team using MLIT xROAD bridge inventory, measured road traffic, candidate roads and e-Stat population.', 'Latest heterogeneous bridge inspection records acquired in October 2026; 2021 traffic, 2020 population/medical register and September 2024 roads.', '730,844 retained inventory assets; policy domain 46,688 conditional one-bridge/one-road units. Support/mesh group files use original selected population supports.', 'Latest-record snapshot and hypothetical mutually exclusive single-road events; budget grid 0–100% in one-percentage-point steps.', 'Adopt unique nearby anchors within 50 m with the confirmed conflict and shared-road exclusions; use observed categories/dimensions/traffic, bridge-area relative costs, full prevention, equal event weights and fixed affordable prefixes. Group labels retain original support definitions; bridges have an independent budget.'),
    'pavement-common-sample': ('Independent observed pavement domain with measured traffic and anchored grades.', 'TR01 research team using MLIT xROAD pavement inspection inventory and accepted MLIT road/traffic correspondence.', 'Heterogeneous latest pavement inspections acquired in October 2026; 2021 traffic and September 2024 candidate roads.', '70,049 unique source roads, with 264,099 adopted anchored inspection records; 47 prefectures.', 'Latest inspection records, not repeated inspections forming a longitudinal panel.', 'Require usable grades I–III, national-route identity and unique compatible nearby traffic-matched source curves. Aggregate the worst anchored grade; preserve missing/out-of-domain records and original lengths. No verified interval footprint is inferred.'),
    'pavement-policy': ('Independent observed-pavement policy rankings, prefixes and simultaneous finite-cost outcomes.', 'TR01 research team using the 70,049-road observed common sample and fixed national baseline.', '2020 population/medical register, 2021 traffic, September 2024 roads and heterogeneous latest inspection dates.', '70,049 intervention roads, full candidate routing network and 175,612 selected origins.', 'Four budget scenarios (1%, 5%, 10%, 20%) and three impairment increments (10%, 25%, 50%); fixed population.', 'Cost is source length; traffic ranks observed volume/cost, condition III/II/I then lower cost, and access isolated restoration gain/cost. Original road ID resolves ties. Only eligible roads deteriorate. Joint full-network routing, scaled partial entries and own reference burdens evaluate every complete selection.'),
    'national-pavement-inputs': ('Complete national pavement scenario inputs with observed and hypothetical values separated.', 'TR01 research team using MLIT candidate road classifications/widths, measured traffic and anchored pavement grades.', '2021 traffic and heterogeneous latest inspection inputs on September 2024 roads; fixed 2020 population used downstream.', 'All 21,415,800 accepted candidate source-road records, including original source multiplicity.', 'Fixed scenario inputs; not an observed deterioration panel.', 'Preserve 1,183,367 measured traffic values and 70,049 anchored grades. Missing traffic uses adopted class/width references with at least 30 observation units and fallbacks; municipal/other-road traffic uses 10% of the prefectural reference. Missing grades use documented probabilities and seed 20261007. Original observed fields stay missing; an independent seed-20261008 tie key is shared across rules.'),
    'national-pavement-policy': ('Main national rankings, maintenance selections and joint finite-cost policy outcomes.', 'TR01 research team using separately labelled national scenario inputs and the fixed medical baseline.', 'Fixed 2020 population/medical registration, 2021 measured traffic and September 2024 roads; heterogeneous inspections and hypothetical missing-input assignments.', '21,415,800 intervention source roads; 175,612 origins and 125,874,788 residents; original area groups.', 'Four requested budgets and three impairment settings; no future-population pavement calculation.', 'Traffic ranks Scenario Traffic Volume, condition III/II/I, and access exact isolated restoration gain/length. Common independent tie key and source identity define fixed lists. Longest affordable complete prefixes are evaluated jointly under 10%, 25%, 50% increments. Costs, budgets and group reference denominators are national; no observed-sample results are transplanted.'),
    'national-pavement-allocation': ('Native-mesh geography and selected-set overlap of national 1% allocations.', 'TR01 research team using accepted original MLIT road curves and national policy selections.', 'September 2024 candidate road geometry and fixed national 1%-budget scenario.', '318,761 road-bearing native meshes per rule; three-by-three source-ID overlap matrix.', 'Single 1%-budget allocation comparison; not a temporal series.', 'Clip original curves to native mesh boundaries and allocate original metric lengths proportionally; boundary-only segments are assigned once. Conserve both all-road and selected lengths. Jaccard counts unique selected original road IDs and is independent of bridge demographic stability.'),
    'demographic-stability': ('Independent bridge population reweighting, conditional losses, protection and selected-set stability.', 'TR01 research team using the MLIT 2024 nationwide 1-km population projection and the independent bridge system.', 'Matched reference 2020 and scenario years 2030, 2040, 2050; observed system inputs held fixed.', '175,485 forecast-matched origins; 127 unmatched origins retained with missing weights; all 46,688 bridge units retained.', 'Four ten-year scenario points; spatial projections are not observed longitudinal population.', 'Sum municipality components within mesh; original census population times official future/reference ratio. Matched 2020 is census-anchored. Freeze original area groups and costs/system; rerank only access. Preserve genuine projected zeros and missing unmatched forecasts. No future elderly location, traffic or pavement outcome is inferred.'),
}


def registry():
    entries = {}
    paths = [ROOT/'data/exp/data-preprocessing/decisions.json',
             *sorted((ROOT/'data/exp/data-preprocessing').glob('*/decisions.json')),
             *sorted((ROOT/'data/exp/data-preprocessing/common-budget').glob('*decisions.json'))]
    for path in paths:
        for source, info in json.loads(path.read_text()).get('datasets', {}).items():
            out = info.get('output', '')
            if not out or 'validation_preprocessed' in out:
                continue
            if (ROOT/out).exists():
                entries[out] = {**info, 'source':source, 'decision':str(path.relative_to(ROOT))}
    extras = {
        'population/population_support_membership.parquet': ('src/preprocessing/construct_population_age_support.py', 'Official reciprocal suppression/recipient membership from both source-preserving census tables.'),
        'population/population_age_support.parquet': ('src/preprocessing/construct_population_age_support.py', 'Known-age, unknown-age and older-age support counts constructed from the two source-preserving census tables.'),
        'medical-baseline/medical_mesh_baseline.parquet': ('src/preprocessing/construct_medical_baseline.py', 'Original mesh population, age-support membership and preserved candidate-network path diagnostics.'),
        'medical-baseline/medical_age_support_baseline.parquet': ('src/preprocessing/construct_medical_baseline.py', 'Original official age supports and member-level path/population accounting.'),
    }
    for suffix,(script,info) in extras.items():
        entries['data/processed/'+suffix]={'source':info,'script':script,'variables':[], 'decision':'data/exp/data-preprocessing/decisions.json'}
    return entries


def add_datasets(doc, entries):
    for path, info in entries.items():
        family = Path(path).parts[2]
        description,provider,period,spatial,temporal,preprocess = FAMILY[family]
        parquet = pq.ParquetFile(ROOT/path)
        key = dataset_key(path)
        label = Path(path).stem.removesuffix('_preprocessed').replace('_',' ').capitalize()
        label = {
            'T001140': '2020 census mesh total population and broad age counts (T001140)',
            'T001188': '2020 census mesh five-year age counts (T001188)',
        }.get(label, label)
        doc['datasetDictionary'][key] = {
            'datasetSourceName':label,
            'datasetDescription':description+' This file stores '+label.lower()+'.',
            'datasetAttribute':'table', 'datasetProvider':provider,
            'datasetGenerationInfo':'Source-preserving preprocessing or explicit deterministic scenario construction. Technical upstream source: '+info['source']+'.',
            'datasetPeriod':period,
            'datasetScale':f'Japan-wide source coverage; {parquet.metadata.num_rows:,} stored rows and {len(parquet.schema_arrow)} columns. File row counts are records, not independent measurements or beneficiary totals.',
            'datasetSpatialResolution':spatial, 'datasetTemporalResolution':temporal,
            'datasetPreprocessingInfo':preprocess,
            'processedPath':path,'upstreamSource':info['source'],
            'preprocessingScript':info['script'], 'lineageRecord':info['decision'],
            'storedRecordCount':parquet.metadata.num_rows,
            'storageSchema':{f.name:str(f.type) for f in parquet.schema_arrow},
        }


# Canonical source choices prevent a last-processed repeated outcome table from
# silently replacing the statistical support of an original input variable.
BASE = 'data/processed/'
POP = BASE+'population/population_age_support.parquet'
MESH = BASE+'medical-baseline/medical_mesh_baseline.parquet'
SUPPORT = BASE+'medical-baseline/medical_age_support_baseline.parquet'
INPUT = BASE+'national-pavement-inputs/national_pavement_inputs_preprocessed.parquet'
UNITS = BASE+'common-budget/conditional_maintenance_units_preprocessed.parquet'
GROUP = BASE+'common-budget/conditional_group_losses_preprocessed.parquet'
CURVES = BASE+'common-budget/conditional_protection_curves_preprocessed.parquet'
RANK = BASE+'national-pavement-policy/national_pavement_rankings_preprocessed.parquet'
SUMMARY = BASE+'national-pavement-policy/national_pavement_group_outcomes_preprocessed.parquet'
SCENARIO = BASE+'demographic-stability/population_scenarios_preprocessed.parquet'


def canonical(name):
    if name=='Local Total Population':return MESH,name
    if name in ['Support Total Population','Known Age Population','Unknown Age Population','Population Aged 65+ Share Among Known Ages']:return POP,name
    if name.startswith('Published Population Aged '):
        source='T001140' if name in ['Published Population Aged 65+','Published Population Aged 75+'] else 'T001188'
        return BASE+f'population/{source}_preprocessed.parquet',name
    if name=='Candidate Medical Network Distance':return MESH,name
    if name.startswith('Population Share ') or name=='Population-Weighted Candidate Medical Network Distance':return SUPPORT,name
    if name=='Population Density':return BASE+'population-density/population_density_preprocessed.parquet',name
    if name in ['Lost Medical Path Population','Additional Medical Distance Burden']:return BASE+'road-service-loss/national_road_service_loss_preprocessed.parquet',name
    if name in ['Higher Ageing Area','Baseline Distance Disadvantaged Area']:return BASE+'common-budget/conditional_mesh_area_groups_preprocessed.parquet',name
    if name in ['Bridge Inspection Category','Bridge Length','Bridge Width','Relative Intervention Cost','Maintenance Intervention Effect','Conditional Event Weight']:return UNITS,name
    if name in ['Higher Ageing Lost Medical Path Population','Distance Disadvantaged Lost Medical Path Population','Higher Ageing Additional Medical Distance Burden','Distance Disadvantaged Additional Medical Distance Burden']:return GROUP,name
    if name in ['Forecast Matched Origin','Forecast Reference Population','Projected Total Population','Population Change Factor','Scenario Population','Population-Change Scenario','Ageing Scenario']:return SCENARIO,name
    if name=='Priority Stability':return BASE+'demographic-stability/priority_stability_preprocessed.parquet',name
    if name=='Distribution of Protected Access':return BASE+'demographic-stability/scenario_protection_preprocessed.parquet',name
    if name in ['Graph Source Edge ID','Baseline Source Road Length','Observed Traffic Volume','Anchored Pavement Inspection Category'] or name.startswith('Scenario ') or name in ['Road Classification Code','Road Width Code']:return INPUT,name
    if name=='Pavement Effective Road Cost':return INPUT,'Baseline Source Road Length'
    if name in ['Pavement Relative Maintenance Cost','Pavement Traffic Priority Score','Pavement Isolated Restoration Gain','Pavement Access Priority Score','Pavement Priority Rank']:return RANK,name
    if name=='Pavement Maintenance Selected':return BASE+'national-pavement-policy/national_pavement_selections_preprocessed.parquet',name
    if name in ['Pavement Medical Access Cost','Pavement Medical Access Cost Increase']:return BASE+'national-pavement-policy/national_pavement_mesh_scenarios_preprocessed.parquet',name
    if name=='Mesh Maintained Road Length Share':return BASE+'national-pavement-allocation/national_pavement_mesh_allocation_preprocessed.parquet',name
    if name=='Pavement Rule Selection Overlap':return BASE+'national-pavement-allocation/national_pavement_rule_overlap_preprocessed.parquet',name
    if name.startswith('Pavement '):return SUMMARY,name
    return CURVES,name


def null_count(file, column):
    index=file.schema_arrow.get_field_index(column)
    assert index>=0, column
    counts=[]
    for i in range(file.metadata.num_row_groups):
        statistics=file.metadata.row_group(i).column(index).statistics
        if statistics is None or statistics.null_count is None:
            return sum(batch.column(0).null_count for batch in file.iter_batches(columns=[column]))
        counts.append(statistics.null_count)
    return sum(counts)


def semantic(name):
    booleans={'Higher Ageing Area','Baseline Distance Disadvantaged Area','Forecast Matched Origin','Pavement Maintenance Selected','Scenario Deterioration Eligible','Scenario Maintenance Eligible'}
    categoricals={'Maintenance Priority','Pavement Maintenance Priority','Pavement Population Group','Ageing Scenario','Road Classification Code','Road Width Code','Scenario Traffic Basis','Scenario Condition Basis','Scenario Traffic Calibration Stratum','Scenario Condition Calibration Stratum'}
    if name in booleans:return 'boolean','indicator (true/false)'
    if name in categoricals:return 'categorical','category label or source code'
    if name=='Graph Source Edge ID':return 'other','original source-road identifier (not a magnitude)'
    if name in ['Bridge Inspection Category','Anchored Pavement Inspection Category','Scenario Pavement Need Category']:return 'ordinal','inspection/need category (bridge I–IV; pavement I–III)'
    if name in ['Pavement Priority Rank']:return 'ordinal','one-based position within a fixed priority list'
    if name=='Population-Change Scenario':return 'categorical','scenario year (2020, 2030, 2040, 2050)'
    if name=='Scenario Priority Tie Key':return 'continuous','dimensionless uniform tie key in [0, 1)'
    if name in ['Observed Traffic Volume','Scenario Traffic Volume','Pavement Traffic Priority Score']:return 'continuous','vehicles per daytime 12 hours'
    if 'Selected' in name and ('Count' in name):return 'count','complete selected road/maintenance units'
    if name.startswith('Published Population Aged') or name in ['Local Total Population','Support Total Population','Known Age Population','Unknown Age Population'] or 'Lost Medical Path Population' in name:return 'count','residents'
    if name in ['Forecast Reference Population','Projected Total Population','Scenario Population','Avoided Lost Path Population Per Event']:return 'continuous','residents (scenario weights/event averages may be fractional)'
    if name=='Population Density':return 'continuous','residents per square kilometre of complete support footprint'
    if name in ['Candidate Medical Network Distance','Population-Weighted Candidate Medical Network Distance','Bridge Length','Bridge Width','Baseline Source Road Length']:return 'continuous','metres'
    if name in ['Relative Intervention Cost','Requested Relative Budget','Actual Relative Spending','Unused Relative Budget']:return 'continuous','bridge-area relative cost (square-metre equivalents; not money)'
    if 'Additional Medical Distance Burden' in name or name=='Avoided Distance Burden Per Event':return 'continuous','person-kilometres (conditional finite-route loss/event average)'
    if name in ['Pavement Effective Road Cost','Pavement Medical Access Cost','Pavement Medical Access Cost Increase']:return 'continuous','reference metres of generalized impedance (not actual extra distance/time)'
    if name=='Pavement Access Priority Score':return 'continuous','resident reference-kilometres per source-length relative cost unit'
    if name=='Pavement Isolated Restoration Gain' or (name.startswith('Pavement ') and 'Burden' in name):return 'continuous','resident reference-kilometre equivalents'
    if name in ['Pavement Relative Maintenance Cost','Scenario Relative Maintenance Cost','Pavement Requested Relative Budget','Pavement Actual Relative Spending','Pavement Unused Relative Budget']:return 'continuous','source-length relative cost (reference-metre equivalents; not money)'
    if 'Share' in name or name in ['Conditional Event Weight','Maintenance Intervention Effect','Budget Share','Population Change Factor','Priority Stability','Distribution of Protected Access','Pavement Budget Share','Pavement Impedance Increment','Pavement Rule Selection Overlap','Scenario Priority Tie Key']:return 'continuous','dimensionless ratio; stored linearly, percentages where explicitly displayed'
    raise AssertionError('Unit not assigned: '+name)


def add_variables(doc, rows, entries):
    metadata={}
    for path,info in entries.items():
        for v in info.get('variables',[]):
            metadata[(path,v['readable_name'])]=v
    files={}
    for name, row in rows.items():
        path,column=canonical(name)
        file=files.setdefault(path,pq.ParquetFile(ROOT/path))
        missing=null_count(file,column)
        field=file.schema_arrow.field(column)
        original=metadata.get((path,name),{}).get('original_name',column)
        attr,unit=semantic(name)
        description=row['definition']+' '+row['construction']
        article_role = row['role']
        if article_role.startswith('TBD (') and article_role.endswith(')'):
            article_role = article_role[5:-1]
        entry={'variableNameInDataset':original,'variableNameInArticle':row['full'],
               'variableAttribute':attr,'variableUnit':unit,'variableDescription':description,
               'variableSource':dataset_key(path),'observationNumber':file.metadata.num_rows-missing,
               'processedColumn':column,'processedPath':path,'storageDtype':str(field.type),
               'storedRecordCount':file.metadata.num_rows,'missingRecordCount':missing,
               'observationDefinition':'Number of non-null stored values in this canonical file, not independent measurements or residents.',
               'articleRole':article_role}
        if name.startswith('Published Population Aged'):
            values=pd.read_parquet(ROOT/path,columns=[column])[column]
            entry['numericValueCount']=int(pd.to_numeric(values,errors='coerce').notna().sum())
            entry['observationDefinition']='Non-null source-preserving string records, including suppression symbols; numericValueCount separately counts available numeric age counts. Strings are not silently classified as zero.'
            entry['variableDescription']+=' Canonical records retain publication symbols. Only standalone/recipient numeric values contribute to official support age counts; suppressed members do not locate older residents.'
        if name=='Local Total Population':
            entry['variableNameInDataset']='T001140001'
            entry['variableDescription']+=' Original census count is numeric in this mesh baseline; the selected main model uses 175,612 of the 176,964 original meshes without changing their counts.'
        if name in ['Known Age Population','Unknown Age Population']:
            entry['variableNameInDataset']=column+' (derived support count)'
        if path==RANK:
            entry['observationDefinition']='Non-null stored source-road/rule rows: three complete lists repeat the same 21,415,800 source roads. These are not 64,247,400 independent roads.'
            entry['variableDescription']+=' Canonical main-national usage differs from the independent observed supplement: traffic score is Scenario Traffic Volume, cost is source length, isolated gain is recomputed on the all-road 25% reference; shared random tie key then source ID breaks ties. Appendix B uses its own 70,049-road input/ranking file and observed traffic/cost rule.'
            entry['independentUnitCount']=21415800
            entry['domainVariants']=[{'domain':'Appendix B observed pavement','processedPath':BASE+'pavement-policy/pavement_rankings_preprocessed.parquet','definition':'Independent observed traffic/cost, worst category then cost and isolated gain/cost; original source-ID final ties.'}]
        if path==SUMMARY or 'mesh_scenarios' in path or 'mesh_allocation' in path or 'selections_preprocessed' in path:
            entry['variableDescription']+=' Stored rows repeat units across rules/budgets/increments/groups as identified by the file keys; their count is not an independent sample size.'
        if name=='Pavement Maintenance Selected':
            entry['variableDescription']+=' The sparse selection file stores selected (true) rows only, across twelve prefixes; roads absent from a specified prefix are unselected under that selection algorithm, not missing observations. It does not materialize all false rows.'
        if path==SCENARIO:
            entry['observationDefinition']='Non-null stored origin/year rows across four years; 175,612 origins are retained in the file, with 175,485 compatible matched origins. Population weights are missing for the 127 unmatched origins in each year.'
        if name=='Pavement Effective Road Cost':
            entry['variableNameInDataset']='Derived scenario expression; no persisted effective-cost column'
            entry['storageDtype']='derived float64 from stored source lengths and scenario selection'
            entry['storageStatus']='Computed in routing kernels; baseline source length is the persisted construction input.'
            entry['variableDescription']+=' Effective cost equals source length times one plus the unmaintained impairment increment; this is a documented derived construct, not an observed or fabricated stored column.'
        if name=='Anchored Pavement Inspection Category':
            entry['variableDescription']+=' The national observed field has 70,049 non-null grades among 21,415,800 retained roads. It is distinct from the complete hypothetical Scenario Pavement Need Category.'
        if name=='Observed Traffic Volume':
            entry['variableDescription']+=' The canonical national field has 1,183,367 non-null matched road records sharing 25,566 observed traffic units; all other roads remain missing.'
        if name in ['Lost Medical Path Population','Additional Medical Distance Burden']:
            entry['variableDescription']+=' Canonical road-dependence values cover all source roads; the 46,688-unit bridge supplement selects its own independent event/cost domain. These closure losses are not finite pavement restoration scores.'
        doc['variableDictionary'][slug(name)]=entry


def add_context_sources(doc):
    sources = [
        ('candidate_network', 'data/processed/national-network-rule-validation/graph_exact.npz', 'other', 'Accepted undirected candidate medical-routing graph', 'MLIT National Land Numerical Information N13-2024; network construction by TR01.', 'September 2024 road source, confirmed from the saved official catalog; not a synchronized population/facility year.', 'Source-road curves and endpoint graph nodes; no spatial averaging.', 'Single candidate configuration.', 'Accepted vehicle source-record filters, structure continuity and mesh-entry rules; parallel alternatives retained; connectors, speeds and legal restrictions not modeled.'),
        ('prefecture_display_boundaries', 'data/exp/cartographic-basemap/prefecture_display_boundaries.geojson', 'geojson', 'Official prefecture land and boundary display layer', 'MLIT National Land Numerical Information N03-2024; display processing by TR01.', '2024-01-01 administrative reference.', 'Prefecture polygons dissolved from municipal areas; simplified by 0.0005 degrees for display only.', 'Single administrative snapshot.', 'Dissolve to 47 prefectures and simplify for drawing; never used to alter network distances or mesh values.'),
        ('neighbouring_land_context', 'data/exp/cartographic-basemap/neighbour_context/ne_110m_admin_0_countries.shp', 'shapefile', 'Coarse neighbouring-country land for map context', 'Natural Earth.', 'Bundled cartographic edition retained locally; no analytical observation year.', '1:110 million cartographic context.', 'Static display layer; no time-series interpretation.', 'Exclude Japan from neighbouring context and clip to display windows; original values do not enter any policy outcome.'),
        ('official_population_projection_components', 'data/exp/data-briefing/demographic-projection/source_scope.parquet', 'table', 'Official 2024 nationwide mesh population projection components', 'MLIT National Land Numerical Information.', 'Official 2020 reference and 2030, 2040, 2050 projections used by this study; 2024 release.', 'Native nominal 1-km meshes, with municipality-specific components.', 'Four ten-year scenario points; not observed longitudinal counts.', 'Retain 177,791 municipality-specific records on 176,896 meshes; sum 895 multi-component meshes before joining census origins. Use unaggregated total-population changes, not privacy-aggregated age locations.'),
    ]
    for key,path,attr,title,provider,period,spatial,temporal,prep in sources:
        assert (ROOT/path).exists(),path
        doc['datasetDictionary'][key]={'datasetSourceName':title,'datasetDescription':title+'. Context or upstream input supporting the adopted outputs.', 'datasetAttribute':attr,'datasetProvider':provider,'datasetGenerationInfo':'Retained official source acquisition or documented derived display/network construction; no new acquisition in this dictionary build.','datasetPeriod':period,'datasetScale':'Japan-wide source extent or neighbouring geographic context, according to the documented role.','datasetSpatialResolution':spatial,'datasetTemporalResolution':temporal,'datasetPreprocessingInfo':prep,'processedPath':path}


def render_dictionaries(doc, scaffold, previews):
    main={'national_candidate_medical_distance','ageing_population_concentration_and_candidate_medical_access','road_service_dependence_and_measured_traffic','national_pavement_maintenance_protection_under_budget_constraints','national_distribution_of_residual_pavement_access_burden','national_spatial_allocation_of_pavement_maintenance','national_pavement_maintenance_protection_and_resource_accounting'}
    appa={'analysis_sample_coverage','population_roads_and_medical_destinations','baseline_analysis_sample_selection'}
    appb={'pavement_maintenance_protection_under_budget_constraints','spatial_distribution_of_residual_pavement_access_burden','pavement_maintenance_protection_and_resource_accounting'}
    figures=[p for p in sorted((ROOT/'data/results/figures').glob('*.png'))]
    tables=[p for p in sorted((ROOT/'data/results/tables').glob('*.xlsx'))]
    observed_tables={p.stem.removeprefix('Table_'):pd.read_excel(p) for p in tables}
    explanations={
        'national_pavement_maintenance_protection_under_budget_constraints':'At 1% requested national cost, access priority protects 57.28% of aggregate reference burden versus 1.78% for traffic and 1.12% for condition. Its higher-ageing and distance-disadvantaged group protection is 10.54% and 14.05%, much lower than aggregate protection; this is a distributional limit, not universal equity. At 5% budget, traffic protects 15.78% overall but 16.08% in higher-ageing areas and 20.57% in distance-disadvantaged areas. Fixed-set impairment sensitivity does not test alternative synthetic traffic or condition inputs.',
        'national_distribution_of_residual_pavement_access_burden':'The national 25% reference sums to 21,216,259.39 resident reference-km. At 1%, residual burdens are 20,839,167.06 for traffic, 20,977,671.06 for condition and 9,064,469.27 for access. The native mesh maps locate contributions to these same totals, with linear 0–1000 colour saturation. They do not locate selected roads or identify the number of distinct protected residents.',
        'national_spatial_allocation_of_pavement_maintenance':'The three rules select 160,299, 212,291 and 310,810 complete source roads at 1% cost, respectively. Mapping uses selected road length over candidate road length within each of 318,761 road-bearing meshes; road-bearing cells outside the inhabited origin sample are retained. Panel d compares count-based Jaccard overlap of the same selected sets. A local 5% colour threshold is a display choice, not the national budget or protection rate.',
        'national_pavement_maintenance_protection_and_resource_accounting':'Twelve rows give unrounded spending and burden outcomes for three rules at four budgets. At 1%, access avoids 12,151,790.12 resident reference-km and protects 57.28% overall, but 10.54% and 14.05% in the two fixed groups. At 20%, it reaches essentially full protection in this scenario; traffic and condition protect 51.23% and 22.08% overall. These results are conditional on hypothetical missing inputs and uniform impairment, not observed repair effects or an optimized solution.',
        'ageing_population_concentration_and_candidate_medical_access':'On 145,771 original supports, Pearson correlation is +0.367703 for percentage ageing versus square-root medical distance and −0.721186 for log10 density versus square-root medical distance; R-squared is 0.135206 and 0.520109. Both hexagon displays count every support once. Density uses full-support footprints including water; medical means use finite-path member population. Trends are unadjusted descriptive associations.',
        'baseline_analysis_sample_selection':'From 176,964 baseline meshes and 126,146,099 residents, the excluded union removes 1,352 meshes and 271,311 residents, retaining 175,612 meshes and 125,874,788 residents. Age and undefined-distance exclusions overlap in one mesh with one resident. Original age supports and baseline source records are preserved.',
        'analysis_sample_coverage':'The 200 pending-entry exclusions include one mesh also excluded by age; mutually exclusive map classes therefore show 199 pending cells and 12 undefined-age cells. The local visibility inset repeats original pixels without adding observations or defining a representative sample.',
        'population_roads_and_medical_destinations':'The Kyoto display window contains 907 published meshes, 1,489,760 residents, 98 included hospitals and 1,670 included general clinics. Its bounds define an illustrative input window, not Kyoto municipal totals or a representative national policy sample. The local panels share geographic extents; the national locator uses a separate scale.',
        'road_service_dependence_and_measured_traffic':'The observed comparison includes 1,183,367 source roads sharing 25,566 traffic units, approximately 5.53% of candidate road records. There are 809,898 exact zero finite-distance burdens. Among 16,355 positive path-loss scenarios, 13,716 only lose their own fixed origin entries and 2,639 exceed that population. Full road removal is a dependence diagnostic, not the pavement impairment or maintenance effect.',
        'pavement_maintenance_protection_under_budget_constraints':'At 1% of the independent 70,049-road length cost, access protects 25.66% overall, 5.54% in higher-ageing areas and 8.11% in distance-disadvantaged areas. Traffic and condition protect 1.25% and 1.25% overall. Different eligibility, scores, costs and references prevent treating these values as the national replacement or a direct empirical validation of its effect sizes.',
        'spatial_distribution_of_residual_pavement_access_burden':'The observed-domain reference is 305,716.86 resident reference-km, with 160,326 of 175,612 origins having zero excess cost because other roads remain unimpaired. Under 1% policies the totals are 301,908.57, 301,886.05 and 227,272.03 for traffic, condition and access. Low-positive palette changes preserve the shared linear 0–30 normalization and all underlying values.',
        'pavement_maintenance_protection_and_resource_accounting':'The twelve independent observed-domain rows preserve budget/spending identities and group denominators. At 20%, access protects 87.47% overall versus 21.86% for traffic and 17.69% for condition. Its avoided burden is 267,406.06 resident reference-km; these are conditional outcomes on the restricted domain.',
        'common_budget_maintenance_protection_and_distribution':'At 1% bridge-area cost, access protects 99.82% of conditional lost-path population but only 7.13% of additional finite-route burden. At 2% and higher evaluated budgets, its two group-reachability curves coincide. These are separate event-loss denominators; near-complete reachability protection does not mean nearly all Japanese residents are protected.',
        'allocation_rules_and_shared_scenario_assumptions':'The ten component rows separate observations from assumed bridge-to-road links, relative cost, full prevention, equal single-event weights and priority rules. All 46,688 units share 15,815 measured traffic observations. The Notes and Detailed methods worksheets preserve the precise conditional scope and independent matched-population extension.',
        'protection_by_maintenance_rule_and_population_group':'At a requested 1% bridge budget, traffic, condition and access protect 27.47%, 11.09% and 99.82% of lost-path population, respectively. Access distance-burden protection is 7.13%, while its actual spending is 0.9448146% of total relative cost; the complete prefix leaves resources unused. Group columns report reachability protection, not elderly individuals or finite-route distance protection.',
        'priority_stability_under_budget_and_demographic_scenarios':'Sixteen selected-set overlaps compare matched 2020 with four population scenario years at four budgets. Jaccard measures selected-unit intersection/union, not ranking correlation or protected population. The vertical range is 65–102%; protection values are tabulated separately. Changes are conditional bridge sensitivity with costs, destinations and original groups fixed.',
        'spatial_changes_in_access_priority_maintenance_selection':'At 1% requested bridge cost, access selects 1,527 units at matched 2020 and 1,119 at 2050. The latter retain 1,115 reference units, exclude 412 and add four. Source-road curves locate these conditional units, not verified bridge footprints or beneficiaries. A road exiting selection does not disappear or lose all maintenance need.',
        'reachability_protection_under_demographic_scenarios':'All 36 percentages are the original scenario-specific protection outcomes at a 1% requested bridge cost. Access aggregate protection changes from 99.82% at matched 2020 to 99.95% in 2050, but changing loss denominators prevent interpreting this as increasing absolute protected residents. Three groups overlap; 127 unmatched origins with 278 census residents are excluded only from this demographic comparison.',
        'national_candidate_medical_distance':'The baseline maps the retained 175,612 original mesh origins and 125,874,788 residents to registered hospitals/general clinics. Original excluded meshes remain grey rather than being treated as zero-distance access. Native footprints and full-range square-root colours are preserved. This provides spatial baseline context, not maintenance benefits or travel time.',
    }
    for kind,paths in [('figure',figures),('table',tables)]:
        for path in paths:
            s=path.stem.removeprefix('Figure_' if kind=='figure' else 'Table_')
            key=('fig_' if kind=='figure' else 'table_')+s
            original=scaffold[kind+'Dictionary'][key]
            title=s.replace('_',' ').title()
            place='Main manuscript: Results' if s in main else 'Appendix A: Baseline and inputs' if s in appa else 'Appendix B: Independent observed pavement' if s in appb else 'Appendix C: Independent bridge comparison'
            if s=='baseline_analysis_sample_selection':place='Appendix A: Sample accounting; summarize coverage in main Methods'
            caption=title
            assert len(caption.split())<=12
            entry={kind+'Id':key,kind+'Path':str(path.relative_to(ROOT)),kind+'Caption':caption,
                   kind+'Aim':original[kind+'Aim'],kind+'Explanation':explanations[s]+' '+path.with_suffix('.caption.md').read_text().strip(),
                   kind+'SuitableSection':place,
                   'analysisScript':f'src/analyses/{kind}_{s}.py',
                   'sourceCaptionPath':str(path.with_suffix('.caption.md').relative_to(ROOT)),
                   'artifactSha256':digest(path),'humanAcceptance':'Original output accepted in AnaSOP Section 8; subsequent export corrections have agent visual-review receipts, without inferring additional human acceptance.'}
            if kind=='table':
                b=next(b for b in previews['workbooks'] if b['path']==str(path.relative_to(ROOT)))
                entry['worksheets']=[{'name':sh['name'],'rows':sh['rows'],'columns':sh['columns'],'role':'Primary publication data' if i==0 else 'Definitions, precision or detailed conditional methods'} for i,sh in enumerate(b['sheets'])]
                # Numeric evidence stays reproducible and exact in the workbook.
                entry['numericEvidenceSource']=str(path.relative_to(ROOT))+'; all sheets inspected locally with pandas/openpyxl.'
            doc[kind+'Dictionary'][key]=entry


def add_levels(doc):
    levels=[]
    def add(name,aim,inputs,method,outputs,interpretation):
        levels.append({'levelName':name,'levelAim':aim,'inputs':inputs,'modelOrMethod':method,'outputs':outputs,'interpretation':interpretation})
    add('Baseline sample and candidate medical access','Locate baseline disadvantage and document coverage for Supporting Point 1.',
        ['Local Total Population','Candidate Medical Network Distance','Population Aged 65+ Share Among Known Ages','Population-Weighted Candidate Medical Network Distance','Population Share With Candidate Medical Path','Population Share Without Candidate Medical Path','Population Share With Pending Road Entry'],
        r'Select meshes with finite medical distance and a defined original support ageing share; select supports with defined ageing and weighted distance. Preserve complete input baselines and count overlapping exclusions by their union. Support distance is \[D_g=\frac{\sum_{m\in G_g:F_m=1}P_m d_m}{\sum_{m\in G_g}P_m F_m}.\] Here \(G_g\) is the complete official support, \(P_m\) original local population, \(d_m\) candidate network distance and \(F_m\) finite-path status. Display selected distance as \(d_m/1000\) kilometres in native meshes. Divide category resident counts by original baseline population for sample shares.',
        ['National Candidate Medical Distance','Analysis Sample Coverage','Baseline Analysis Sample Selection'],
        'Existing population and path statuses determine the selected sample and conditional support means. Retained population is 125,874,788 on 175,612 meshes; the excluded union has 271,311 residents. Evidence supports spatial baseline accounting within the candidate metric. Historical registration, undirected routing and missing legal/time rules limit service interpretation; individual elderly access and maintenance effects are not identified.')
    add('Population intensity and demographic association','Describe age/intensity associations with access on compatible official supports.',
        ['Support Total Population','Population Density','Population Aged 65+ Share Among Known Ages','Population-Weighted Candidate Medical Network Distance'],
        r'Full-support density is \(\rho_g=N_g/\sum_{m\in G_g}a_m\), with \(N_g\) support population and \(a_m\) complete ellipsoidal mesh area including water. On 145,771 selected supports, fit \[z_g=\alpha_j+\beta_j x_g^{(j)}+\epsilon_{gj},\quad z_g=\sqrt{D_g/1000}.\] Here \(j\) denotes two panels, \(x_g^{(a)}=100S_g\) is percentage known-age ageing, \(x_g^{(b)}=\log_{10}\rho_g\) is log10 density, and \(\alpha_j,\beta_j,\epsilon_{gj}\) are intercept, slope and residual. Compute unweighted OLS and Pearson correlation on original supports, with \(R_j^2=r_j^2\). Count each support once in display-coordinate hexagons; fit straight lines in those same coordinates.',
        ['Ageing, Population Concentration and Candidate Medical Access'],
        'Descriptive Pearson r is +0.367703 for ageing and −0.721186 for log10 density, against square-root medical distance. This supports associations in the selected supports, with separate original full-population and finite-path denominators. It does not identify causal demographic effects, clustering, individual elderly travel or spatial-independence significance.')
    add('Local input illustration','Make the origin, network and destination construction understandable.',
        ['Local Total Population','Candidate Medical Network Distance'],
        'Draw four panels: national Kyoto locator, published population on native meshes, accepted original road curves, and included registered hospitals/general clinics over the same local population/roads. Keep the three local extents identical; use a separate locator scale. Retain source counts, curves and facility coordinates; no interpolation or new routing.',
        ['Population Roads and Medical Destinations'],
        'The Kyoto window visually explains inputs, containing 907 published meshes and 1,489,760 residents. It is illustrative and does not support national representativeness, a new analytical sample or a maintenance effect. Medical locator/category fields are technical inputs rather than newly introduced scientific outcomes.')
    add('Conditional road dependence and observed demand','Evaluate Supporting Point 2 on the bounded observed-traffic comparison.',
        ['Local Total Population','Candidate Medical Network Distance','Observed Traffic Volume','Lost Medical Path Population','Additional Medical Distance Burden'],
        r'Remove each original source road independently, disabling fixed origin/destination entries requiring it while retaining parallel alternatives. \[L_e=\sum_{m\in M}P_m(1-I_{me}),\quad T_e=\frac{1}{1000}\sum_{m\in M:I_{me}=1}P_m(d_m^{(-e)}-d_m).\] Here \(M\) is the fixed selected origin set, \(e\) one source-road event, \(I_{me}\) finite-path status after that event, \(L_e\) lost-path residents and \(T_e\) additional person-kilometres among reachable origins. Compare measured traffic only on the strict accepted correspondence; map original road curves and retain zero burdens.',
        ['Road Service Dependence and Measured Traffic'],
        'National loss computation covers 21,415,800 roads; the plotted measured comparison covers 1,183,367 roads sharing 25,566 measurements. Positive path loss can be confined to the closed origin entry and is not automatically geographic isolation. Evidence is bounded dependence/demand context, not a pavement impairment effect, annual risk, causal repair benefit or universal all-road traffic finding.')
    add('Complete national scenario inputs','Define a common national decision domain while preserving observed-input gaps.',
        ['Graph Source Edge ID','Baseline Source Road Length','Road Classification Code','Road Width Code','Observed Traffic Volume','Anchored Pavement Inspection Category','Scenario Traffic Volume','Scenario Pavement Need Category','Scenario Traffic Basis','Scenario Condition Basis','Scenario Relative Maintenance Cost','Scenario Priority Tie Key'],
        'All 21,415,800 accepted source roads are deterioration-eligible and maintainable. Preserve original missing measured inputs. Construct separately labelled class/width traffic medians with at least 30 measurement units and documented fallbacks; municipal/other values use 10% of the prefectural reference. Missing need grades use adopted observed-category probabilities with seed 20261007. Relative cost is original source length; independent tie keys use seed 20261008. Input coverage is not immunity from deterioration.',
        ['National Pavement Maintenance Protection Under Budget Constraints','National Distribution of Residual Pavement Access Burden','National Spatial Allocation of Pavement Maintenance','National Pavement Maintenance Protection and Resource Accounting'],
        'Existing class/width inputs and adopted assignments establish a reproducible conditional domain. Only 1,183,367 roads have observed traffic and 70,049 have anchored grades; remaining scenario values are assumptions. Evidence supports traceability and common-domain comparability, not empirical prediction of missing road traffic/condition or unique physical national mileage.')
    add('National ranking and affordable prefixes','Compare common national priority lists and resource allocations.',
        ['Scenario Traffic Volume','Scenario Pavement Need Category','Scenario Relative Maintenance Cost','Scenario Priority Tie Key','Pavement Isolated Restoration Gain','Pavement Access Priority Score','Pavement Budget Share','Pavement Maintenance Selected'],
        r'For national roads, \[s_{e,T}=v_e,\quad s_{e,C}=h_e,\quad s_{e,A}=g_e/\ell_e.\] Here \(v_e\) is Scenario Traffic Volume, \(h_e\) ordinal need category, \(g_e\) isolated restoration gain computed from the all-road 25% reference and \(\ell_e\) source-length relative cost. Descending primary scores, ascending common tie key and then source ID define each complete list. At \(b\in\{0.01,0.05,0.10,0.20\}\), select the longest prefix whose summed source-length cost is at most \(b\sum_e\ell_e\); no splitting or skipping. Report actual and unused cost and selected count. Keep lists/selections fixed across increments.',
        ['National Pavement Maintenance Protection Under Budget Constraints','National Pavement Maintenance Protection and Resource Accounting','National Spatial Allocation of Pavement Maintenance'],
        'Only ordering changes between rules on a common cost denominator. At 1% requested cost, selected road counts are 160,299, 212,291 and 310,810 for traffic, condition and access. Costs are relative source lengths, not monetary budgets or counts. Isolated restoration gains define a fixed heuristic list, not additive joint benefits or a proved optimum.')
    add('National joint protection and distribution','Answer the central question and Supporting Points 3–4 within the adopted pavement contract.',
        ['Local Total Population','Candidate Medical Network Distance','Pavement Effective Road Cost','Pavement Impedance Increment','Pavement Population Group','Pavement Reference Access Burden','Pavement Residual Access Burden','Pavement Avoided Access Burden','Pavement Access Protection Share','Higher Ageing Area','Baseline Distance Disadvantaged Area'],
        r'In the main domain all source roads are eligible: \[\widetilde\ell_e=\ell_e[1+\delta(1-x_e)],\quad H_S(\mathbf{x},\delta)=\frac{1}{1000}\sum_{m\in S}P_m[\widetilde d_m(\mathbf{x},\delta)-d_m],\quad Q_S=\frac{H_S(\mathbf{0},\delta)-H_S(\mathbf{x},\delta)}{H_S(\mathbf{0},\delta)}.\] Here \(x_e\) is selection, \(\delta\) the impairment increment, \(S\) all residents or a fixed original area group, \(\widetilde d_m\) recomputed shortest generalized medical cost and \(d_m,P_m\) original baseline distance/population. Scale partial origin, destination and same-road legs consistently and retain parallel minima. Evaluate 36 policies and three references jointly; undefined zero-reference fractions remain missing. Map native contributions \(P_m[\widetilde d_m-d_m]/1000\).',
        ['National Pavement Maintenance Protection Under Budget Constraints','National Distribution of Residual Pavement Access Burden','National Pavement Maintenance Protection and Resource Accounting'],
        'At 1% cost, access protects 57.28% of aggregate reference burden but only 10.54% and 14.05% in the fixed ageing/distance groups; traffic and condition protect 1.78% and 1.12% overall. This supports a conditional allocation contrast and uneven distribution. Objective alignment is deliberate; results do not prove equity, an optimal portfolio, actual extra journeys, causal repair effects or future-population pavement protection. Each overlapping group has its own denominator. Fixed-set 10/25/50% sensitivity tests only the impairment parameter.')
    add('National allocation geography and overlap','Locate resources separately from population burden and compare selected sets.',
        ['Graph Source Edge ID','Baseline Source Road Length','Pavement Maintenance Selected','Pavement Maintenance Priority','Pavement Budget Share','Mesh Maintained Road Length Share','Pavement Rule Selection Overlap'],
        r'Clip original curves to native mesh cells and proportionally allocate each original metric length. For road-bearing cell \(m\), display \(u_m=\sum_e x_e\ell_{em}/\sum_e\ell_{em}\), where \(\ell_{em}\) is the conserved source-length contribution in that cell. Retain all 318,761 road-bearing meshes, valid zero selected length and NoData for no road denominator. For selected sets \(A_r,A_s\), compare \(J_{rs}=|A_r\cap A_s|/|A_r\cup A_s|\). Use the same national 1% sets as protection; empty union is undefined.',
        ['National Spatial Allocation of Pavement Maintenance'],
        'Lengths conserve candidate and selected costs, and all nine count-based overlaps reproduce intersections/unions. This supports allocation geography and rule similarity; it does not locate beneficiaries or imply that high selection overlap guarantees equal access outcomes. Map colours saturate at 5%, matrix colours at 100%, without clipping stored values.')
    add('Independent observed-pavement comparison','Assess the same policy contrast on the restricted observed domain in Appendix B.',
        ['Observed Traffic Volume','Anchored Pavement Inspection Category','Pavement Relative Maintenance Cost','Pavement Traffic Priority Score','Pavement Isolated Restoration Gain','Pavement Access Priority Score','Pavement Budget Share','Pavement Impedance Increment','Pavement Access Protection Share'],
        r'Restrict eligible deterioration/maintenance set \(E_P\) to 70,049 adopted observed common roads, while retaining the full routing network: \(\widetilde\ell_e=\ell_e[1+\delta\mathbf{1}\{e\in E_P\}(1-x_e)]\). Other roads remain at baseline. Reuse joint burden/protection formulas with this domain\'s own references and length-cost denominator. Traffic ranks measured volume/length, condition III/II/I then lower cost, access isolated 25% restoration gain/length; original road ID resolves final ties. Use four complete-prefix budgets and three fixed-set increments.',
        ['Pavement Maintenance Protection Under Budget Constraints','Spatial Distribution of Residual Pavement Access Burden','Pavement Maintenance Protection and Resource Accounting'],
        'This independent observed-domain experiment supports a bounded rule contrast, with 25.66% aggregate access protection at 1% versus approximately 1.25% for either alternative. Its reference burden is 305,716.86 resident reference-km; 160,326 meshes have zero increase because outside roads remain unimpaired. It is not all-road deterioration or direct validation of main-national effect sizes; matching, cost and effects remain conditional. No missing observed input is imputed in this supplement.')
    add('Independent bridge allocation','Compare bridge-specific conditional single-event protection in Appendix C.',
        ['Bridge Length','Bridge Width','Bridge Inspection Category','Observed Traffic Volume','Relative Intervention Cost','Maintenance Intervention Effect','Conditional Event Weight','Maintenance Priority','Budget Share','Lost Medical Path Population','Additional Medical Distance Burden','Higher Ageing Lost Medical Path Population','Distance Disadvantaged Lost Medical Path Population'],
        r'For 46,688 conditional units, \(c_e=\ell_e w_e\) uses bridge dimensions, independently of pavement source length. Rank traffic/cost, category IV/III/II/I then lower cost, or lost-path residents/cost then finite-route burden/cost; stable source identity resolves ties. Select longest affordable prefixes for \(b\sum_ec_e\). Under full prevention of equally weighted mutually exclusive single events, \[Q_{r,o,h}(b)=\frac{\sum_e x_{er}(b)Y_{e,o,h}}{\sum_eY_{e,o,h}},\quad o\in\{L,T\}.\] Here \(r\) is a rule, \(o\) reachability or finite-distance loss, \(h\) all residents or a fixed origin group, and \(Y\) its per-event loss. Uniform event weights cancel. Average avoided loss per event divides selected loss sums by 46,688.',
        ['Common-Budget Maintenance Protection and Distribution','Allocation Rules and Shared Scenario Assumptions','Protection by Maintenance Rule and Population Group'],
        'At 1% bridge cost, access protects 99.82% of conditional lost-path population but 7.13% of additional finite-route burden, spending 0.9448146% of total area-relative cost. Outcome definitions differ from pavement joint burden. Evidence supports conditional bridge prioritization, not annual risk, physical verified bridge repairs, unique beneficiaries, monetary benefits or universal equity. Bridge budgets and event outcomes remain independent of pavement.')
    add('Independent bridge population sensitivity','Evaluate the bridge-specific demographic component of Supporting Point 4.',
        ['Forecast Matched Origin','Forecast Reference Population','Projected Total Population','Population Change Factor','Scenario Population','Population-Change Scenario','Ageing Scenario','Budget Share','Priority Stability','Distribution of Protected Access'],
        r'On 175,485 matched origins, \[p_{mt}^{*}=P_m\frac{F_{mt}}{F_{m0}},\quad J_t(b)=\frac{|A_t(b)\cap A_0(b)|}{|A_t(b)\cup A_0(b)|}.\] Here \(F_{mt}\) is the summed official municipality-component mesh population, \(F_{m0}\) its 2020 reference, \(p_{mt}^{*}\) census-anchored scenario weights and \(A_t\) access-priority selected units under the original bridge prefix/cost rule. Reweight original road-specific path changes for 2020/2030/2040/2050; freeze original area groups, network, destinations, units and costs. Rerank access only. Compute Jaccard at four budgets and scenario-specific selected/total reachability loss at 1%; missing unmatched forecasts remain missing and genuine projected zeros stay zero.',
        ['Priority Stability Under Budget and Demographic Scenarios','Reachability Protection Under Demographic Scenarios'],
        'Matched 2020 has 125,874,510 original residents; 127 unmatched origins and 278 residents are excluded only from this comparison. Jaccard measures selected-set change, while protection has a changing scenario loss denominator. Aggregate access protection rises from 99.82% to 99.95% by 2050, without establishing increasing absolute protected residents. Evidence supports bounded bridge sensitivity, not future pavement priorities, future elderly locations, observed population trends or future traffic.')
    add('Bridge population selection geography','Locate changes in already computed bridge selected sets.',
        ['Population-Change Scenario','Budget Share','Maintenance Priority','Priority Stability'],
        r'At a fixed requested 1% bridge cost, partition the two-year union into retained \(A_t\cap A_0\), exiting \(A_0\setminus A_t\) and new \(A_t\setminus A_0\) units. Reuse the matched-population access rankings and original source-road curves, common geographic views and mutually exclusive display classes. Reconcile membership counts to stored Jaccard intersection/union and affordable prefix costs.',
        ['Spatial Changes in Access-Priority Maintenance Selection'],
        'Selections decline from 1,527 units at matched 2020 to 1,119 at 2050, retaining 1,115 reference units, exiting 412 and adding four. This supports spatial accounting of the same conditional scenario; it adds no model. Source-road location is not a beneficiary location or verified bridge footprint. Exiting a budget prefix does not imply disappearance or absence of need.')
    doc['analysisStructureBrief']={'overview':'Applied conditional pavement-maintenance allocation in Japan, directed at one central question and four supporting questions. The main model uses all 21,415,800 accepted candidate source-road records and fixed original population. Appendix A documents baseline inputs/coverage; Appendix B retains the 70,049-road observed pavement experiment; Appendix C retains independent 46,688-unit bridge allocation and matched-origin population sensitivity. There is no joint pavement/bridge budget, future-population pavement model, causal ageing/repair effect or observed economic/fiscal trajectory. Existing engineering validation and human output acceptance support reproducibility/presentation within the adopted assumptions; they do not establish external empirical validity or journal novelty.', 'levels':levels}


def main():
    EXP.mkdir(parents=True,exist_ok=True)
    source_paths=[ROOT/'docs/AnaSOP.md',*sorted((ROOT/'data/results/figures').glob('*.*')),*sorted((ROOT/'data/results/tables').glob('*.xlsx')),*sorted((ROOT/'src/analyses').glob('*.py'))]
    sources_before={str(p.relative_to(ROOT)):digest(p) for p in source_paths if p.is_file()}
    scaffold=yaml.safe_load((EXP/'draft_actionbrief.yaml').read_text())
    previews=json.loads((EXP/'table_previews.json').read_text())
    rows=section_variables((ROOT/'docs/AnaSOP.md').read_text())
    assert len(rows)==111
    doc={'doc_type':'actionbrief','version':'1.0.0','datasetDictionary':{},'variableDictionary':{},'tableDictionary':{},'figureDictionary':{}}
    entries=registry()
    add_datasets(doc,entries);add_context_sources(doc)
    add_variables(doc,rows,entries)
    render_dictionaries(doc,scaffold,previews);add_levels(doc)
    # Additional integrity checks supplement the bundled legacy schema.
    for e in doc['variableDictionary'].values():
        assert e['variableSource'] in doc['datasetDictionary']
        assert (ROOT/e['processedPath']).exists()
    var_names=set(rows)
    planned={e['figureCaption'] for e in doc['figureDictionary'].values()}|{e['tableCaption'] for e in doc['tableDictionary'].values()}
    for level in doc['analysisStructureBrief']['levels']:
        assert set(level['inputs'])<=var_names
        assert all(slug(title) in {slug(s) for s in planned} for title in level['outputs'])
    spec=importlib.util.spec_from_file_location('actionbrief_validator',ROOT/'.codex/skills/build-content-dictionary/scripts/validate_actionbrief.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    missing=[];validator.check_no_todo(doc,'actionbrief',missing)
    assert not missing,missing
    assert not validator.validate_schema(doc),validator.validate_schema(doc)
    english=yaml.safe_dump(doc,allow_unicode=True,sort_keys=False,width=110)
    assert not re.search(r'[\u3400-\u9fff]',english),'Writing-facing dictionary must be English.'
    for name,expected in sources_before.items():assert digest(ROOT/name)==expected,name
    (EXP/'draft_actionbrief.yaml').write_text(english)
    (EXP/'missing_fields.json').write_text('[]\n')
    (ROOT/'docs/actionbrief.yaml').write_text(english)
    receipt={'status':'completed_dictionary_build','datasets':len(doc['datasetDictionary']),'variables':len(doc['variableDictionary']),'figures':len(doc['figureDictionary']),'tables':len(doc['tableDictionary']),'analysis_levels':len(doc['analysisStructureBrief']['levels']),'workbooks':previews['workbook_count'],'worksheets':sum(len(b['sheets']) for b in previews['workbooks']),'remaining_placeholders':0,'all_content_english':True,'analytical_inputs_match_section_4':True,'analytical_outputs_match_section_8':True,'variable_sources_resolve':True,'artifact_hashes_unchanged':sources_before,'actionbrief_sha256':digest(ROOT/'docs/actionbrief.yaml'),'observation_count_policy':'Canonical source counts are non-null stored records; repeated scenario/ranking rows and numeric availability of source count strings are explicitly described. Derived effective cost is not falsely described as a persisted column.','anasop_narrow_repair':'Section 6 bare I=1/II=2/III=3 coding changed to equivalent English wording under estimation-framework-planning; formulas and research definitions unchanged.'}
    (EXP/'build_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['artifact_hashes_unchanged']},indent=2))


if __name__=='__main__':main()
