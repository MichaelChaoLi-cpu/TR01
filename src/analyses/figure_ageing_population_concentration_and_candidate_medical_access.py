#!/usr/bin/env python3
"""Ageing, Population Concentration and Candidate Medical Access.

Plan: two descriptive support-level count-binned displays in the main manuscript.
Framework: AnaSOP Sections 5-7, confirmed official-support ageing, density and
conditional medical distance; descriptive OLS fits and Pearson correlations,
without a causal model or a new sample rule.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tr01-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from publication_figure_checks import check_publication_layout

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Ageing, Population Concentration and Candidate Medical Access"
SLUG = "ageing_population_concentration_and_candidate_medical_access"
SAMPLE = ROOT / "data/processed/medical-analysis-sample/medical_age_support_analysis_preprocessed.parquet"
DENSITY = ROOT / "data/processed/population-density/population_density_preprocessed.parquet"
EXP = ROOT / "data/exp/figure-table-generation" / SLUG
OUT = ROOT / f"data/results/figures/Figure_{SLUG}.png"
AGE = "Population Aged 65+ Share Among Known Ages"
DISTANCE = "Population-Weighted Candidate Medical Network Distance"
INTENSITY = "Population Density"
KEY = "Publication Support Code"
FIT_DECISIONS = EXP / "linear_fit_decisions.json"


def fit_relationship(x, y, predictor, panel):
    """Unweighted OLS with an intercept, on original support observations."""
    assert np.isfinite(x).all() and np.isfinite(y).all() and len(x) == len(y)
    fit = linregress(x, y)
    design = np.column_stack([np.ones(len(x)), x])
    independent_coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose([fit.intercept, fit.slope], independent_coefficients,
                               rtol=1e-11, atol=1e-11)
    independent_r = float(np.corrcoef(x, y)[0, 1])
    np.testing.assert_allclose(fit.rvalue, independent_r, rtol=1e-12, atol=1e-12)
    fitted = fit.intercept+fit.slope*x
    residual = y-fitted
    r_squared = 1-float(np.sum(residual**2)/np.sum((y-y.mean())**2))
    np.testing.assert_allclose(r_squared, independent_r**2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(residual.mean(), 0, atol=1e-10, rtol=0)
    return {
        "Panel": panel, "Predictor": predictor, "Response": "Square root of medical distance in kilometres",
        "N": len(x), "OLS Intercept": float(fit.intercept), "OLS Slope": float(fit.slope),
        "Pearson Correlation": independent_r, "R Squared": r_squared,
        "Predictor Minimum": float(x.min()), "Predictor Maximum": float(x.max()),
        "Fitted Square Root Distance Minimum": float(fitted.min()), "Fitted Square Root Distance Maximum": float(fitted.max()),
        "Negative Fitted Count": int(np.count_nonzero(fitted < 0)),
    }


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    exp, output = EXP, OUT
    decisions = json.loads(FIT_DECISIONS.read_text())
    exp.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert decisions["density_predictor_scale"] == "log10"
    assert decisions["response_scale"] == "square root of distance in kilometres"
    sources = [SAMPLE, DENSITY]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    sample = pd.read_parquet(SAMPLE)
    density = pd.read_parquet(DENSITY)
    assert len(sample) == 145771 and sample[KEY].is_unique
    assert len(density) == 146540 and density[KEY].is_unique
    density = density.set_index(KEY).loc[sample[KEY]].reset_index()
    pd.testing.assert_series_equal(density[KEY], sample[KEY])
    for column in ["Member Mesh Count", "Support Total Population"]:
        np.testing.assert_array_equal(density[column], sample[column])
    age = sample[AGE].to_numpy(dtype=float)*100
    distance = sample[DISTANCE].to_numpy(dtype=float)/1000
    intensity = density[INTENSITY].to_numpy(dtype=float)
    assert np.isfinite(age).all() and np.all((age >= 0) & (age <= 100))
    assert np.isfinite(distance).all() and np.all(distance >= 0)
    assert np.isfinite(intensity).all() and np.all(intensity > 0)
    predictors = [age, np.log10(intensity)]
    predictor_names = ["Ageing share in percentage points", "log10 population density"]
    fit_results = [fit_relationship(x, np.sqrt(distance), name, label)
                   for x, name, label in zip(predictors, predictor_names, "ab")]
    sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
    plt.rcParams.update({"font.size": 8, "text.color": "#263746", "axes.labelcolor": "#263746",
                         "axes.edgecolor": "#727C85", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(190/25.4, 110/25.4), sharey=True)
    fig.subplots_adjust(left=.15, right=.855, top=.91, bottom=.23, wspace=.18)
    artists = []
    bin_records = []
    bin_summaries = []
    upper_distance = float(distance.max()*1.04)
    display_distance = np.sqrt(distance)
    density_upper = max(50000, float(intensity.max()*1.05))
    display_x = [age, np.log10(intensity)]
    x_limits = [(-1, 101), (0, np.log10(density_upper))]
    y_ticks = np.array([0, 1, 2, 5, 10, 20, 40, 50], dtype=float)
    for label, ax, x, x_range in zip("ab", axes, display_x, x_limits):
        # Bin in display coordinates so nonlinear axes do not stretch the cells.
        artist = ax.hexbin(x, display_distance, gridsize=(58, 36),
                          extent=(*x_range, 0, np.sqrt(upper_distance)),
                          mincnt=1, cmap="viridis", linewidths=0,
                          rasterized=True, zorder=2)
        artists.append(artist)
        counts = np.asarray(artist.get_array())
        assert np.isfinite(counts).all() and np.all(counts >= 1)
        np.testing.assert_array_equal(counts, np.rint(counts))
        assert int(counts.sum()) == len(sample)
        bin_summaries.append({"panel": label, "occupied_hexagons": len(counts),
                              "support_count_sum": int(counts.sum()),
                              "minimum_count": int(counts.min()), "maximum_count": int(counts.max())})
        centers = np.asarray(artist.get_offsets())
        bin_records.extend({"Panel": label, "Display X": float(center[0]),
                            "Display Square Root Distance": float(center[1]),
                            "Statistical Unit Count": int(count)}
                           for center, count in zip(centers, counts))
        assert np.all((x >= x_range[0]) & (x <= x_range[1]))
        assert np.all((display_distance >= 0) & (display_distance <= np.sqrt(upper_distance)))
        ax.set_xlim(*x_range)
        ax.set_ylim(0, np.sqrt(upper_distance))
        ax.set_yticks(np.sqrt(y_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value*value:g}"))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DCE2E6", linewidth=.6)
        ax.tick_params(labelsize=7.5, direction="out", length=3, width=.7, colors="#374151")
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
        ax.text(-.10, 1.055, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="top")
    axes[0].set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_xlabel("Population aged 65+\namong known ages (%)", fontsize=8, labelpad=8)
    axes[0].set_ylabel("Population-weighted medical distance\n(km; square-root axis scale)",
                       fontsize=8, labelpad=10)
    axes[1].set_xticks(np.log10([1, 10, 100, 1000, 10000]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{10**value:,.0f}"))
    axes[1].set_xlabel("Population density (residents/km²)\n(logarithmic axis)", fontsize=8, labelpad=8)
    fit_coordinates = []
    for panel, ax, result, predictor in zip("ab", axes, fit_results, predictors):
        grid = np.linspace(predictor.min(), predictor.max(), 1001)
        fitted = result["OLS Intercept"]+result["OLS Slope"]*grid
        line_x, line_y = grid, fitted
        # Fit directly in displayed coordinates so every trend is a straight line.
        np.testing.assert_allclose(np.diff(line_y), result["OLS Slope"]*np.diff(line_x),
                                   rtol=1e-10, atol=1e-13)
        positive = fitted >= 0
        ax.plot(line_x, line_y, color="#D55E00", linewidth=1.8, zorder=4,
                path_effects=[path_effects.Stroke(linewidth=3.2, foreground="white"), path_effects.Normal()])
        fit_coordinates.extend({"Panel": panel, "Predictor": float(x), "Fitted Square Root Distance": float(y),
                                "Displayed Within Nonnegative Domain": bool(valid)}
                               for x, y, valid in zip(grid, fitted, positive))
        sign = "+" if result["OLS Slope"] >= 0 else "-"
        x_name = "age (%)" if panel == "a" else "log10 density"
        slope_format = f"{abs(result['OLS Slope']):.3f}"
        annotation = (f"Trend: √d = {result['OLS Intercept']:.3f} {sign} {slope_format} x\n"
                      f"x = {x_name}; d in km\n"
                      f"Pearson r(x, √d) = {result['Pearson Correlation']:+.3f}\nR² = {result['R Squared']:.3f}")
        ax.text(.985, .965, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#000000", linespacing=1.35,
                bbox={"facecolor": "white", "edgecolor": "#D9DEE2", "linewidth": .5, "alpha": .94, "pad": 3}, zorder=6)
    maximum_count = max(item["maximum_count"] for item in bin_summaries)
    norm = LogNorm(vmin=1, vmax=maximum_count)
    for artist in artists:
        artist.set_norm(norm)
    color_ax = fig.add_axes([.89, .25, .014, .62])
    count_ticks = [value for value in [1, 10, 100, 1000, 10000] if value <= maximum_count]
    if maximum_count > count_ticks[-1]:
        if maximum_count/count_ticks[-1] >= 2:
            count_ticks.append(maximum_count)
        else:
            count_ticks[-1] = maximum_count
    colorbar = fig.colorbar(artists[0], cax=color_ax, ticks=count_ticks)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    colorbar.ax.tick_params(labelsize=7.5, which="minor", length=0)
    colorbar.set_label("Statistical units per hexagon\n(log colour scale)", fontsize=7.5, labelpad=6)
    pd.DataFrame(bin_records).to_csv(exp / "display_bin_counts.csv", index=False)
    fit_frame = pd.DataFrame(fit_results)
    fit_frame.to_csv(exp / "linear_fit_statistics.csv", index=False)
    saved_fit = pd.read_csv(exp / "linear_fit_statistics.csv")
    np.testing.assert_allclose(saved_fit.select_dtypes("number"), fit_frame.select_dtypes("number"), rtol=1e-12, atol=1e-12)
    pd.DataFrame(fit_coordinates).to_csv(exp / "linear_fit_display_predictions.csv", index=False)
    np.testing.assert_allclose(np.square(np.sqrt(distance)), distance, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(10**display_x[1], intensity, rtol=1e-14, atol=1e-14)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frames = [ax.get_window_extent(renderer) for ax in axes]
    np.testing.assert_allclose([frames[0].width, frames[0].height], [frames[1].width, frames[1].height], atol=1e-8)
    labels = [text.get_window_extent(renderer) for ax in axes
              for text in [*ax.get_xticklabels(), *ax.get_yticklabels()] if text.get_visible()]
    assert not any(a.overlaps(b) for i, a in enumerate(labels) for b in labels[i+1:])
    check_publication_layout(fig, exp)
    fig.savefig(output, dpi=600, facecolor="white")
    fig.savefig(exp / "preview.png", dpi=150, facecolor="white")
    plt.close(fig)
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected
    receipt = {
        "title": TITLE, "placement": "main_manuscript", "subpanels": 2,
        "status": "generated_and_validated", "framework": "AnaSOP Sections 5-7 descriptive official-support associations",
        "variable_authorization": "MILI-D-20261007-007",
        "display_revision_request": "MILI-D-20261007-008",
        "linear_fit_authorization": decisions["fit_authorization"],
        "predictor_scale_authorization": decisions["predictor_scale_authorization"],
        "display_type": "Hexagon-binned counts of unchanged support observations",
        "selected_official_supports": len(sample), "observation_count_per_panel": [len(sample), len(sample)],
        "observation_unit": "One selected official age publication support", "observation_weight": "one count per support, without additional population weighting",
        "population_density_unit": "residents/km2",
        "density_denominator": "Complete original member mesh footprint area, including water, not occupied-land area",
        "medical_distance_unit": "km", "medical_distance_basis": "Full-support conditional finite-path member-population-weighted baseline distance",
        "age_percentage_range": [float(age.min()), float(age.max())],
        "population_density_range": [float(intensity.min()), float(intensity.max())],
        "medical_distance_range_km": [float(distance.min()), float(distance.max())],
        "zero_distance_supports": int(np.count_nonzero(distance == 0)),
        "x_axis_scales": ["linear percentage", "logarithmic density"],
        "common_y_axis_scale": "square root of nonnegative kilometre coordinates, original kilometre ticks",
        "display_binning": "Equal hexagon geometry in transformed percentage/log10-density and square-root kilometre coordinates",
        "gridsize": [58, 36], "hexagon_counts": bin_summaries,
        "count_colour_map": "viridis", "common_count_colour_range": [1, maximum_count],
        "count_colour_scale": "logarithmic, shared by both panels", "empty_hexagons": "not drawn",
        "all_supports_counted_once_per_panel": True, "bin_count_conservation_check": "passed",
        "original_value_conversion_roundtrip": "passed",
        "values_trimmed_or_imputed": False, "new_regression": "Two unweighted descriptive OLS models, each with an intercept",
        "new_smoothing_or_sampling": False,
        "fit_response_scale": "Square root of distance in kilometres: the plotted vertical coordinate",
        "fit_predictor_density_scale": decisions["density_predictor_scale"],
        "fit_basis": "All unchanged original selected supports; not hexagon centres or counts",
        "linear_fit_statistics": fit_results,
        "independent_OLS_and_correlation_checks": "passed, least squares and covariance correlation",
        "fit_line_display": "Straight OLS predictions directly in the existing display coordinates; axis limits clip lines without changing fits",
        "fit_line_straightness_check": "passed", "annotation_text_colour": "#000000",
        "inferential_p_values_or_confidence_intervals": "not reported; descriptive spatial associations only",
        "frame_dimensions_match": True, "tick_label_overlap_check": "passed",
        "source_hashes_verified_after_render": hashes, "output": str(output.relative_to(ROOT)), "dpi": 600, "figure_size_mm": [190, 110], "publication_revision_authorization": "MILI-D-20261008-008",
    }
    (exp / "validation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    output.with_suffix(".caption.md").write_text(
        f"{TITLE}. (a) Official-support share of residents aged 65 or older among residents with known ages versus population-weighted candidate medical-network distance. "
        "(b) Official-support population density versus the same medical-distance measure. Population concentration is operationalized here as resident intensity, not spatial clustering. "
        f"Each panel counts all {len(sample):,} selected official age publication supports once. Hexagons summarize overlapping observations in transformed display coordinates; brighter colours represent more statistical units, using a shared logarithmic count colour scale. "
        "The colour reports supports per display hexagon, rather than resident counts or a probability density per unit of the original axes. The two horizontal inputs have different units; counts belong to the respective bins, whose geometry is matched across panels. No additional population weighting is applied to the counts. "
        "Density is full-support population divided by the summed ellipsoidal areas of all complete original member meshes, including water within those footprints, in residents per square kilometre. "
        "Its horizontal axis is logarithmic. Both vertical axes use a square-root scale with original kilometre labels; this retains the full distance range and exact zero distances without trimming. "
        "Orange lines show separate unweighted ordinary least-squares fits with intercepts, estimated on all original support observations rather than bin centres. "
        "The predictor in panel a is ageing share in percentage points; in panel b it is log10 population density. The response in both fits and Pearson correlations is the square root of medical distance in kilometres, matching the plotted vertical coordinate. "
        "The figure reports Pearson r and simple-regression R-squared using these same coordinates. Predictions are plotted directly in display coordinates as straight trend lines, with black statistical notes. The axis tick labels retain original kilometres and residents per square kilometre; stored values are unchanged. "
        f"There are {int(np.count_nonzero(distance == 0)):,} zero-distance supports; zero denotes the accepted candidate-graph metric, not verified co-location of residents and services. "
        "Medical distance is weighted by total member population among finite-path meshes within each support. It remains conditional on the population with a candidate path, whereas density and ageing retain original full-support denominators. "
        "Support-level age counts are not redistributed to member meshes, and the observations do not measure individual older-resident travel. The separate fits are unadjusted descriptive relationships; no smoothing, extra sampling, spatial-independence inference or causal interpretation is applied. "
        "Population and medical registrations refer to 2020; the candidate road configuration refers to 2024. Registered hospitals and general clinics do not establish current opening, public availability or capacity. "
        "The figure describes selected-support associations under the accepted undirected candidate-network scenario, not observed journey times or maintenance-policy effects. "
        "Sources: e-Stat census population and age publication groups; MLIT National Land Numerical Information medical facilities and road centre lines.\n")
    print(f"Saved: {output.relative_to(ROOT)}")
    print(f"Validated: hexagon counts sum to {len(sample):,} unchanged supports in each panel; common count colour scale; zeros and full tails retained.")
    for result in fit_results:
        print(f"Panel {result['Panel']}: Pearson r={result['Pearson Correlation']:.6f}, R2={result['R Squared']:.6f}, predictor={result['Predictor']}")


if __name__ == "__main__":
    main()
