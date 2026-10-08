"""Render the nationwide audit findings from saved machine-readable outputs."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'data/exp/road-quality-check'
s=json.loads((OUT/'summary.json').read_text()); seam=json.loads((OUT/'tile_seam_screen.json').read_text())
attrs=list(csv.DictReader((OUT/'attribute_frequencies.csv').open()))
allowed={'N13_002':set('12345'),'N13_003':set('123456'),'N13_004':set('1234567'),'N13_006':set('123456'),'N13_007':set('12')}
invalid=[r for r in attrs if r['field'] in allowed and r['value'] not in allowed[r['field']]]
(OUT/'code_validation.json').write_text(json.dumps({'invalid_categorical_values':invalid,'count':sum(int(r['count']) for r in invalid)},indent=2)+'\n')
n=lambda key: f'{s.get(key,0):,}'
text=f'''# Nationwide road data quality check

## Assessment

The N13 centerlines provide a nationwide geometric base. They are not yet a validated routable motor-vehicle network. This audit checks source integrity, line geometry, exact endpoints, level differences and tile-boundary candidates; it does not repair or filter roads.

## Scope and results

| Check | Result |
| --- | ---: |
| Archives read | {n('archives')} |
| Road features read | {n('features')} |
| Vertices read | {n('vertices')} |
| Non-LineString features | {n('non_linestring')} |
| Invalid coordinate sequences | {n('invalid_geometry')} |
| Features outside broad Japan coordinate bounds | {n('outside_broad_japan_bounds')} |
| Zero-length features | {n('zero_length_features')} |
| Repeated consecutive vertices | {n('repeated_consecutive_vertices')} |
| Closed lines | {n('closed_lines')} |
| Missing source attribute values | {n('missing_attribute_values')} |
| Unexpected attribute schema | {n('unexpected_schema_features')} |
| Non-integer level values | {n('non_integer_levels')} |
| Invalid categorical code values | {sum(int(r['count']) for r in invalid):,} |
| Exact duplicate geometry groups | {n('duplicate_geometry_groups')} |
| Excess occurrences in those groups | {n('duplicate_geometry_excess')} |
| Cross-tile duplicate geometry groups | {n('cross_tile_duplicate_geometry_groups')} |
| Unique exact endpoint coordinates | {n('unique_exact_endpoints')} |
| Endpoint coordinates with one line-end occurrence | {n('degree_one_endpoints')} |
| Exact endpoint coordinates shared across tiles | {n('exact_cross_tile_endpoints')} |
| Endpoint coordinates with multiple level values | {n('mixed_level_endpoints')} |
| Exact tile-boundary coordinates with one occurrence | {n('boundary_degree_one_endpoints')} |
| Nearby cross-tile boundary pairs within 1.5 m | {seam.get('pairs_within_1_5m',0):,} |
| Nearby cross-tile boundary pairs within 5 m | {seam.get('pairs_within_5m',0):,} |

- All source SHA-256 hashes match the acquisition manifest: {s['raw_hashes_all_match']}.
- Internal feature/geometry/endpoint accounting: {s['internal_count_validation']}.
- Sum of line lengths: {s['length_km_haversine']:,.1f} km, using a spherical haversine approximation. This is a geometry inventory, not official road mileage; it includes all line types and duplicate occurrences.
- Source CRS declarations: `{json.dumps(s['crs'])}`. No coordinate transformation was performed.

## How to interpret the checks

1. Endpoint incidence counts use exact longitude/latitude coordinates, without rounding or snapping. A coordinate with one occurrence can be a legitimate dead end; closed lines contribute two end occurrences at one coordinate. Duplicate geometries contribute additional occurrences. The counts are not a definitive count of disconnected roads.
2. Multiple levels at one coordinate are review candidates. Level differences at a ramp or bridge transition may still be connected. Joining every coincident coordinate could introduce false connections; requiring identical levels everywhere could remove valid transitions.
3. The seam screen considers only coordinates on primary mesh tile boundaries within 1e-8 degrees, compares nearby points belonging to different first-observed tiles, and requires at least one single-occurrence endpoint. It uses haversine distances and reports unique coordinate pairs, not missing links. It does not inventory general interior gaps. Thresholds of 1.5 m and 5 m are diagnostic choices, not adopted repair rules.
4. Duplicate geometry comparison retains exact coordinate sequences, treating reversal as equivalent. It does not equate differently sampled lines or cyclic rotations. Attribute agreement is not established; examples require source inspection before deletion.
5. Broad coordinate bounds are only a coarse screening test. The audit does not establish administrative coverage, identify all interior intersections, resolve grade-separated crossings, compute graph components, or verify actual vehicle access, one-way travel and turn restrictions.

## Consequences for the study

- The centerline type includes ordinary roads, garden paths, footpaths and stairs. A vehicle accessibility model needs an explicit eligibility rule; no category has been excluded here.
- Road state identifies ordinary sections, bridge/elevated sections, tunnels, snow shelters, construction and other/unknown types. It is not pavement condition, deterioration or maintenance need.
- Data registration dates are not opening dates. Road width categories are not observed traffic speeds.
- These data do not supply observed traffic volumes, maintenance expenditure, pavement condition, intervention effects, or municipality fiscal capacity. Those remain separate evidence requirements for the proposed maintenance-priority comparison.
- The next technical step is a documented vehicle-network construction pilot, including level transitions, seam candidates and population/POI attachment. National accessibility estimates should follow pilot validation.

## Reproduction and files

Run from the repository root:

```sh
.venv/bin/python src/exp/check_national_roads.py
.venv/bin/python src/exp/check_road_tile_seams.py
.venv/bin/python src/exp/report_road_quality.py
.venv/bin/python src/exp/validate_road_quality_outputs.py
```

`summary.json`, `archive_summary.csv`, `attribute_frequencies.csv`, `endpoint_degree_distribution.csv`, `tile_seam_screen.json`, `code_validation.json`, and `source_integrity_check.json` hold complete aggregate results. Example CSVs are capped at 200 rows and are not exhaustive issue lists. `diagnostic_index.sqlite` is a regenerated derived index, not a road network.

Raw data and AnaSOP were not modified. No packages were installed. No Git or DVC operation was performed.

## Official definitions

- [N13 2024 source and specifications](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html): national centerlines without node-link structure; source JGD2011; positional standard deviation 25 m; producer visually checked some candidate endpoint gaps within 1.5 m.
- [Centerline type](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadType.html)
- [Road class](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadCategory.html)
- [Road state](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadState.html)
- [Width category](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadWidth.html)
- [Toll category](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_ToilRoadSection.html)
'''
(OUT/'README.md').write_text(text)
print(text[:2800])
