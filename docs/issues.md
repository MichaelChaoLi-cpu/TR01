# Research Output Issues

Review date: 2026-10-08. This is a fresh full review following the four corrections authorized under MILI-D-20261008-008. The review concerns the adopted applied study: national pavement allocation at fixed original population, with independent baseline, observed-pavement and bridge appendices. It does not evaluate an unadopted future-population pavement model or introduce additional analyses.

## Issues

| # | item | type | severity | description |
|---|---|---|---|---|
| — | Current adopted output set | — | — | No unresolved issue identified within the reviewed research contract. Disclosed empirical and scenario limitations remain applicable. |

## Severity Summary

| severity | count |
|---|---:|
| critical | 0 |
| major | 0 |
| minor | 0 |

## Review Coverage

- Directly inspected all 13 current PNG originals visually, including the previously revised figures. All have approximately 600-dpi metadata and a physical width of 190 mm. Checked panel labels, axes, legends, colourbars, coordinate labels, inset placement, visible overall titles, and readability against Section 8. Scenario/group labels are panel descriptions rather than embedded overall figure titles.
- Reviewed all six local workbook previews from the required collector, the complete primary-sheet values, notes/captions and relevant source definitions. No primary-sheet missing values, formula cells or external workbook links were found. All six primary sheets agree with their saved table-value files within numerical precision.
- Read the full source of all 33 analysis Python files, including computation kernels, renderers, table exporters, two entry-point wrappers and the publication-layout helper. All 33 pass independent syntax parsing. Helper/computation files support the evidence chain and are not additional planned publication outputs.
- Read all eight AnaSOP sections and checked current questions, variables, domain-specific formulas and output placement against the deliverables. The inventory has no missing planned figure or table. All 19 current publication items correspond to planned items.
- Confirmed placement: six main figures and one main table; Appendix A has two figures and one table; Appendix B has two figures and one table; Appendix C has three figures and three tables. Appendices use sequential A, B and C labels.
- Ran targeted independent, read-only checks of table accounting, processed policy summaries, map burdens, association statistics, allocation lengths and Jaccard ratios. No figure/table generation, preprocessing or expensive nationwide scoring/routing computation was rerun. Only this report and the skill-required output inventory were written during this review.

Inventory: `data/exp/critique/output_inventory.json`.

## Per-Output Findings

### Figures

| item | placement | review finding |
|---|---|---|
| National Candidate Medical Distance | Main | Describes the selected baseline rather than maintenance benefits. Native meshes, grey exclusions, prefecture boundaries, water and island views conform to the plan; full-range square-root colour normalization is disclosed. Current original is 600 dpi. |
| Ageing, Population Concentration and Candidate Medical Access | Main | Both panels retain 145,771 official supports and count each once. Straight trends, Pearson correlations and R-squared consistently use the stated percentage/log10-density predictors and square-root-distance response. Black notes and publication-size text are readable; no causal interpretation is claimed. |
| Road Service Dependence and Measured Traffic | Main | Map a and inset display b match the adopted layout. The map locates a removed-road scenario, while the binned display uses measured traffic and mean lost-path population as colour. Source-record dependence, repeated traffic observations, closed-entry losses and display saturation are disclosed. Revised inset labels and colourbar are legible. |
| National Pavement Maintenance Protection Under Budget Constraints | Main | Four panels compare aggregate and fixed-group protection, then fixed-selection impairment sensitivity. Four budgets, three rules and three increments match the framework; the shared legend is below the panels. No demographic sensitivity is inferred. |
| National Distribution of Residual Pavement Access Burden | Main | Four native-mesh maps use the same national reference, 1% policy sets and 25% increment. Population-weighted contributions reconcile to aggregate residual burdens. Linear 0–1000 display saturation is disclosed, with zero and NoData distinguished. |
| National Spatial Allocation of Pavement Maintenance | Main | Three allocation maps use maintained road length divided by candidate road length within each mesh; the fourth panel shows count-based selected-set overlap. Shared 0–5% map saturation and the separate 0–100% overlap scale match the plan. Allocation locations are distinguished from beneficiary locations. |
| Analysis Sample Coverage | Appendix A | Maps mutually exclusive inclusion/exclusion classes, with age exclusion taking precedence for the overlapping record. The visibility inset repeats original pixels without adding observations. Its 199 pending-class meshes are consistent with the table's 200 pending-entry meshes and one overlapping exclusion. |
| Population Roads and Medical Destinations | Appendix A | Four panels provide a national locator and three matched Kyoto input views. Common local extents and legible legends are retained. Published population coverage is distinguished from the analytical sample, and the example is explicitly illustrative rather than nationally representative. |
| Pavement Maintenance Protection Under Budget Constraints | Appendix B | Uses the independent 70,049-road observed domain, its own cost denominator and simultaneous finite-cost evaluation. Four panels and fixed-set sensitivity conform to the plan. Outcomes are not substituted for the national replacement. |
| Spatial Distribution of Residual Pavement Access Burden | Appendix B | Four panels reconcile to the observed-domain residual burdens. Pale-blue zeros and stronger low-value colour transitions improve distinction; normalization remains linear 0–30, with upper saturation disclosed. Numerous genuine zeros follow the restricted deterioration domain. |
| Common-Budget Maintenance Protection and Distribution | Appendix C | Three analytical panels occupy a two-by-two layout with a legend cell. Distance and reachability protection remain distinct, group shares use their own denominators, and coincident curves are explained. Independent single-event bridge assumptions are explicit. |
| Priority Stability Under Budget and Demographic Scenarios | Appendix C | One selected-set Jaccard plot conforms to the adopted revision; the former protection heatmap is now a separate table. Four budgets and four years use the matched 2020 reference. The narrowed vertical range is disclosed and labels remain readable. |
| Spatial Changes in Access-Priority Maintenance Selection | Appendix C | Four year panels locate retained, exiting and new bridge-associated selections relative to matched 2020. Original source-road curves, common geography and category accounting are retained. Small new-selection counts are acknowledged rather than exaggerated. |

### Tables

Dimensions refer to primary data sheets, excluding headings and supplementary notes.

| item | placement | dimensions | review finding |
|---|---|---|---|
| National Pavement Maintenance Protection and Resource Accounting | Main | 12 × 9 | All three rules at four budgets reproduce national processed outcomes under the 25% increment. Selected counts, avoided burden, three protection columns and requested/actual/unused resources are consistent. |
| Baseline Analysis Sample Selection | Appendix A | 7 × 5 | Overlapping exclusions, their union and retained sample satisfy mesh/population conservation. Shares use the original baseline population denominator. |
| Pavement Maintenance Protection and Resource Accounting | Appendix B | 12 × 9 | All 12 rows reproduce observed-domain outcomes and accounting. Its independent denominator and conditional sample are stated. |
| Allocation Rules and Shared Scenario Assumptions | Appendix C | 10 × 5 | Observed inputs, assumed linkage/cost/effect/events and ranking rules are distinguished. Detailed methods remain available, and the demographic extension is bridge-specific. |
| Protection by Maintenance Rule and Population Group | Appendix C | 12 × 8 | All protection values reproduce the bridge curves at 1%, 5%, 10% and 20% budgets. Actual plus unused spending equals requested resources; group reachability and aggregate distance protection are correctly distinguished. |
| Reachability Protection Under Demographic Scenarios | Appendix C | 9 × 6 | All 36 percentages reproduce processed scenario-specific protection. Notes specify matched origins, changing loss denominators and frozen area groups. It retains the former heatmap values without a new model. |

## Independent Numerical Checks

1. Both pavement domains contain 117 group summaries: 36 complete policies and three references, each evaluated for three population groups. All 24 combined resource-table rows reproduce their respective source outcomes. All 27 rule/increment/group protection series per domain are nondecreasing across the four requested budgets within numerical tolerance.
2. Each of the eight pavement burden-map panels contains 175,612 unique origins. Mesh contribution is original population multiplied by excess generalized access cost, divided by 1000. Each panel sum reproduces its own All Residents residual burden within floating-point tolerance.

| domain | reference burden (resident reference-km) | zero-reference meshes |
|---|---:|---:|
| National pavement | 21,216,259.392440 | 35 |
| Observed pavement supplement | 305,716.856110 | 160,326 |

The different reference magnitudes and zero counts follow different deterioration domains. They are not interchangeable national estimates. A zero burden is not a missing baseline distance.

3. All 12 bridge result-table rows reproduce four processed protection columns and resource accounting. All 36 demographic-table values reproduce scenario-specific reachability shares. All 16 bridge stability values equal recorded intersection counts divided by union counts; matched 2020 has Jaccard equal to one.
4. The seven-row baseline table satisfies inclusion–exclusion and population conservation. Retained population is 125,874,788 across 175,612 meshes. Original baseline population is 126,146,099 across 176,964 meshes. The excluded union contains 1,352 meshes and 271,311 residents; overlapping exclusions are subtracted once.
5. Independent least-squares and covariance calculations reproduce the association figure's statistics on all 145,771 supports: ageing panel Pearson r = 0.367703 and R-squared = 0.135206; log10-density panel Pearson r = −0.721186 and R-squared = 0.520109. These use square-root medical distance. Both stored hexagon-count sums equal 145,771.
6. Each national allocation rule covers 318,761 road-bearing meshes. Maintained-length shares reproduce numerator/denominator ratios. Candidate lengths sum to 1,941,533,728.738884 reference metres; selected lengths reproduce each rule's actual 1%-budget spending. All nine pavement overlap ratios reproduce their recorded intersection/union counts; the matrix is symmetric with unit diagonal.

## Previous Corrections Rechecked

| previous issue | original severity | current finding |
|---|---|---|
| Low-resolution retained main figures | major | Closed: both revised originals are 600 dpi at publication dimensions; all 13 current originals pass resolution/width checks. |
| Historical/current national scenario inconsistencies | major | Closed: Sections 1–7 distinguish the completed national domain from observed-pavement and bridge supplements, with separate inputs, rankings and denominators. |
| Future-population pavement/bridge boundary | minor | Closed: Supporting Point 4 and the relevant framework/captions separate fixed-population pavement sensitivity from independent bridge population scenarios. |
| Small or crowded publication text | minor | Closed: directly inspected revised labels, island coordinates, road-dependence inset and Kyoto layout; no unresolved clipping or overlap identified. |

Historical resolution evidence remains in `data/exp/critique-resolution/validation.json`, `data/exp/critique-resolution/anasop_resolution.json`, the retained before-review documents and item-specific publication-layout/visual-review receipts. These earlier records support traceability; this review does not claim to have rerun their mutation-bearing validators. Historical human acceptance is distinguished from agent review of the revised exports.

## Interpretation Limits Retained

- The national experiment relies extensively on disclosed hypothetical traffic and condition assignments. Its fixed-selection 10/25/50% impedance comparison tests the impairment parameter, not every alternative missing-input construction. This restricts empirical generalization; adopting documented assumptions does not make them observations.
- Length-relative budgets are not monetary expenditure, facility-count shares or official unique physical mileage. Original source multiplicity and segmentation are retained. Generalized access burdens are resident reference-kilometre equivalents, not measured extra journeys or travel time.
- Aggregate accessibility targeting is aligned with the evaluated outcome. Its aggregate advantage cannot alone prove equity, optimal joint allocation or superiority on unmeasured maintenance-risk objectives. Group protection uses separate overlapping denominators and does not locate individual older residents.
- The selected candidate network omits legal-routing/time restrictions and does not verify present medical operating status, entry accuracy or physical inspection/bridge footprints. Numerical validation establishes fidelity to the adopted scenario rather than empirical repair effects or national matching accuracy.
- Official spatial population scenarios apply to the bridge supplement only, with 175,485 matched origins. The 127 unmatched origins and 278 census residents remain outside all three demographic outputs, without changing fixed-population pavement outcomes. Future pavement demographic analysis and observed economic/fiscal trajectories are outside the current contract.
- This review supports proceeding to the writing-input dictionary for the adopted evidence. It is not a claim that publication novelty, every alternative assumption or journal acceptance has been established. AnaSOP already states the need for a focused literature review to substantiate originality.

## Recommended Next Steps

问题未发现。可以继续进行 build-content-dictionary。
