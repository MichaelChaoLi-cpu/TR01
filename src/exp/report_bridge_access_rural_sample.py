"""Render the bounded source check, keeping source discovery separate from inference."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/feasibility-check/bridge-access-rural-sample'
def main():
    s=json.loads((OUT/'summary.json').read_text());v=json.loads((OUT/'validation.json').read_text())
    assert v['status']=='passed'
    text=f'''# Coordinate-bearing bridge access and mountain traffic sample

Review date: 2026-10-06. Research contract: docs/AnaSOP.md, five applied questions. Nationwide scope and the question-specific feasibility labels are retained. This check supplies acquisition and candidate linkage evidence, not an accepted maintenance-allocation input.

## Result and implications

The documented public basic bridge API actually returned coordinates, facility identifiers, inspection years and health codes without login, an API key or payment in these bounded GET requests. This resolves the previous source-location gap at sample level. It does not establish nationally complete acquisition, one synchronized inspection year, maintenance costs, an intervention effect, or accepted bridge-to-network correspondence.

All {s['bridge_rows']} latest-record bridge samples are unique by facility identifier and lie within 50 m of an eligible ordinary N13 centerline. Only {s['bridge_points_with_bridge_elevated_within_50m']} have a bridge/elevated source-state candidate within 50 m. Requiring that label would miss six sampled records. Do not silently omit them or label them absent from the actual vehicle network. Among these six are short structures (2.0, 2.3, 4.0, 3.4 and 2.9 m) and an old bridge of 25.0 m; source state, geometry and asset identity require case review.

The first ten records of the documented latest endpoint for Nikko city are a convenience sample, not a national coverage sample. Inspection years in the returned records are {s['bridge_inspection_year_counts']}. A latest record per asset does not mean the same inspection year across assets.

The additional six traffic sections comprise two National Route 120 sections, two major prefectural-road sections and two general prefectural-road sections in Nikko city. These form a mountain-area illustration; the municipality label alone does not classify every location as rural, and the selection is not representative of all rural roads. The traffic CSV contains {s['traffic_csv_rows']:,} rows and {s['traffic_csv_columns']} source columns.

At approximately 25 m spacing, {s['traffic_sample_points']} points have median nearest-road distance {s['traffic_distance_p50_m']:.2f} m and 95th percentile {s['traffic_distance_p95_m']:.2f} m. All points are within 15 m; the full surveyed line length also falls within a 15 m buffer of the candidate-road union for all six sections. {100*s['traffic_ambiguous_point_share']:.1f}% of points have multiple candidate segments within nearest distance plus 5 m. Adjacent fragments at endpoints count in this measure; it is not a wrong-match rate. Union-buffer coverage cannot determine the correct branch, carriageway or crossing level.

No traffic values or bridge conditions were assigned to graph edges. Point and buffer thresholds are diagnostic tolerances, not accepted crosswalk rules. The API's explicit coordinate fields were used for location; coordinate-looking facility IDs are retained as opaque identities because some returned IDs differ from their current coordinate fields.

## Actual access and code-format checks

Requests used only the published free basic endpoint and bounded limits:

| Request | Returned rows | Response query count | Interpretation |
| --- | ---: | ---: | --- |
| city=09206, limit=5 | 0 | 0 | Five-digit census/traffic code does not retrieve the intended municipal records here |
| pref=09, limit=5 | 5 | 13018 | Historical record endpoint; count is not a unique-asset count |
| small coordinate window, limit=5 | 5 | 325 | Coordinate-bearing basic response |
| bridges/lastest?city=092061&limit=10 | 10 | 1147 | Latest endpoint with the six-digit code present in actual responses |

The spelling `lastest` follows the published API specification. The API city field is six digits in these actual records; census and traffic codes are five digits after restoring leading zeros. The numeric traffic section identifiers omit a leading zero for Tochigi; zero-padding to eleven digits agrees with all six official GeoJSON R03cen values. This formatting is confined to the exploratory join, while the downloaded source values remain unchanged. An empty code query is not evidence of absent bridges.

The counts above describe these saved responses, not a claim of nationwide completeness or future API availability. No account or application was submitted, and no credentials or fees were needed for the bounded public reads. Larger-scale access must follow the documented public interface, moderate request rates and source terms.

## Source documentation

- [Basic API specification](https://rirs.or.jp/tenken-db/pdf/view/api_specification.pdf), pages 5, 11, 13 and 18: public basic bridge endpoint, filters, bounded pagination and the latest endpoint are documented; authenticated paid/private interfaces are separate. The general login/API application page must not be treated as proof that every public basic request needs credentials.
- [Basic bridge parameter dictionary](https://rirs.or.jp/tenken-db/pdf/view/00_Parameters_base.pdf), page 1: facility identity, location, municipality, inspection year, health and action-status fields. These fields were observed in actual returned JSON; the schema alone is not national coverage evidence.
- [Published access terms](https://rirs.or.jp/tenken-db/pdf/view/etsuran_kiyaku.pdf), Appendix 1: basic coordinates and health information are in the free category; detailed records are a separate service. Article 11 permits secondary use of free data with the source attribution “全国道路施設点検データベース”. This reading supports the basic-data route; no paid service has been selected.
- [N13 catalog](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html) and [road-state dictionary](https://nlftp.mlit.go.jp/ksj/gml/codelist/N13_RoadState.html): state 2 denotes bridge/elevated geometry, not a unique asset identity or physical health. N13 source coordinates are JGD2011. The catalog reports a horizontal-position standard deviation of 25 m, so our distance thresholds must not be interpreted as exact asset identity tests.
- [Traffic tables](https://www.mlit.go.jp/road/census/r3/) and [official geometry viewer](https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/webmap.html). Source counts and observation flags were retained. Main-road traffic coverage remains narrower than all-road N13 coverage. Bulk reuse terms for the map geometry remain a separate check.

The three inspection documentation PDFs are saved and hashed alongside the sample downloads. URLs and actual response hashes are in sources/manifest.json.

## Candidate matching and next action

The road scan read primary tiles 5439 and 5539, {s['n13_scanned_features']:,} source features in total, retaining {s['n13_local_candidate_features']} positive-length eligible ordinary centerlines in 100 m traffic-line and 250 m bridge-point windows. Local bridge/elevated candidate distances beyond those windows are not exhaustive global nearest distances; the within-50 m coverage flags are the intended bridge diagnostics. No topological road edits were made.

Source bridge and map coordinates were interpreted as WGS84 geographic coordinates and projected to EPSG:6677; N13 was transformed from JGD2011. Explicit datum metadata for the bridge/map responses was not independently established, so all metric matches remain exploratory. N13 bridge/elevated labeling alone is insufficient. The next matching rule should combine candidate road class, direction, level, alignment, bridge dimensions and visual review, preserving unresolved cases and many-edge assets.

The next practical research step is to specify and validate a conservative asset-to-network crosswalk, using both the ten located bridges and the existing Tokyo/mountain traffic samples. Once this passes, plan a resumable nationwide acquisition of free bridge basics with pagination, latest-record identity checks, coverage summaries and timestamps. This is an acquisition recommendation, not approval to restrict the full maintenance decision set to bridges. Road maintenance cost and maintenance-to-usable-function assumptions still require evidence before the three prioritization rules can be compared.

## Reproduction and validation

1. `.venv/bin/python src/exp/fetch_bridge_access_rural_sample.py` (network access): bounded source samples and official documentation only.
2. `.venv/bin/python src/exp/check_bridge_access_rural_sample.py`: local CSV/JSON reads, source hash validation and candidate geometry checks.
3. `.venv/bin/python src/exp/report_bridge_access_rural_sample.py`: this report from the saved summaries.

Validation passed: {v['download_hashes_verified']} manifest files, six map/traffic identifiers, ten unique latest bridge IDs, two unchanged N13 archive hashes, and unchanged AnaSOP hash. Only exploratory outputs and directly related scripts were written. No package installation, Git or DVC operation occurred. Files include api_probe_results.csv, bridge_coordinate_candidates.csv, mountain_traffic_candidates.csv, mountain_traffic_point_candidates.csv, summary.json and validation.json.
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print('Report written:',OUT/'README.md')
if __name__=='__main__':main()
