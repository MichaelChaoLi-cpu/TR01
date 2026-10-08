#!/usr/bin/env python3
"""Appendix input illustration; AnaSOP Sections 5-7 cartographic framework.

Published population meshes, accepted candidate-road source curves, and included
medical locators share one illustrative window. No routing/model is estimated.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tr01-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from publication_figure_checks import check_publication_layout
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import LineString, box

from figure_national_candidate_medical_distance_and_analysis_sample_coverage import (
    ROOT, BASE, BOUNDARY, BASEMAP_DIR, SEA, LAND, mesh_raster_indices, native_raster,
    polygon_path, polygons, shape, basemap_views,
)

SLUG = "population_roads_and_medical_destinations"
TITLE = "Population Roads and Medical Destinations"
EXP = ROOT / "data/exp/figure-table-generation" / SLUG
OUT = ROOT / f"data/results/figures/Figure_{SLUG}.png"
FACILITIES = ROOT / "data/processed/medical-baseline/medical_facilities_preprocessed.parquet"
BOUNDS = (135.4, 135.875, 34.95, 35.35)  # Kyoto city and surroundings; display bounds
TILES = ("5235", "5335")
REGION = box(BOUNDS[0], BOUNDS[2], BOUNDS[1], BOUNDS[3])
LOCAL_ASPECT = 1 / np.cos(np.deg2rad(np.mean(BOUNDS[2:])))
LOCAL_FRAME_RATIO = (BOUNDS[3]-BOUNDS[2]) * LOCAL_ASPECT / (BOUNDS[1]-BOUNDS[0])
# Equal frame proportions, with a separate geographic scale for the locator.
NATIONAL_MID_LATITUDE = 34.0
NATIONAL_HEIGHT = LOCAL_FRAME_RATIO * 33 * np.cos(np.deg2rad(NATIONAL_MID_LATITUDE))
NATIONAL_BOUNDS = (122, 155, NATIONAL_MID_LATITUDE-NATIONAL_HEIGHT/2,
                   NATIONAL_MID_LATITUDE+NATIONAL_HEIGHT/2)
SELECTION_SOURCE = "https://www.city.kyoto.lg.jp/sakyo/page/0000243219.html"
ROAD = "#506579"
HOSPITAL = "#D55E00"
CLINIC = "#6A3D9A"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def road_geometries():
    """Read source curves using the accepted graph's exact source-feature IDs."""
    pieces, records, inputs, receipts = [], [], {}, []
    for tile in TILES:
        cache = ROOT / f"data/processed/national-network-rule-validation/tiles/{tile}.npz"
        info = cache.with_suffix(".json")
        archive = ROOT / f"data/raw/roads/n13_2024_geojson/N13-24_{tile}_GEOJSON.zip"
        inputs.update({str(p.relative_to(ROOT)): digest(p) for p in [cache, info, archive]})
        with np.load(cache) as data:
            accepted = set(data["fids"].tolist())
        total, selected = 0, 0
        with zipfile.ZipFile(archive) as bundle:
            member = next(name for name in bundle.namelist() if name.endswith(".geojson"))
            with io.TextIOWrapper(bundle.open(member), encoding="utf-8-sig") as stream:
                for line in stream:
                    if not line.lstrip().startswith('{ "type": "Feature",'):
                        continue
                    total += 1
                    if total not in accepted:
                        continue
                    feature = json.loads(line.strip().rstrip(","))
                    coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=float)
                    if (coordinates[:, 0].max() < BOUNDS[0] or coordinates[:, 0].min() > BOUNDS[1]
                            or coordinates[:, 1].max() < BOUNDS[2] or coordinates[:, 1].min() > BOUNDS[3]):
                        continue
                    properties = feature["properties"]
                    assert str(properties["N13_002"]) == "1"
                    assert str(properties["N13_004"]) not in ("5", "6", "7")
                    original = LineString(coordinates)
                    clipped = original.intersection(REGION)
                    if clipped.is_empty:
                        continue
                    lines = ([clipped] if clipped.geom_type == "LineString"
                             else list(clipped.geoms) if clipped.geom_type in ("MultiLineString", "GeometryCollection") else [])
                    valid = [part for part in lines if part.geom_type == "LineString" and part.length > 0]
                    if not valid:
                        continue
                    source_id = f"{tile}:{total}"
                    selected += 1
                    for part, geometry in enumerate(valid):
                        xy = np.asarray(geometry.coords)
                        assert np.isfinite(xy).all()
                        pieces.append(xy)
                        records.append({"type": "Feature", "properties": {"source_id": source_id, "part": part},
                                        "geometry": {"type": "LineString", "coordinates": xy.tolist()}})
        assert total == json.loads(info.read_text())["source_features"]
        receipts.append({"tile": tile, "source_features_scanned": total,
                         "selected_candidate_source_features": selected})
        print(f"Source curves {tile}: {selected:,} candidate records in window", flush=True)
    assert pieces
    (EXP / "display_roads.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:6668"}},
         "features": records}, separators=(",", ":"))+"\n")
    return pieces, inputs, receipts


def local_basemap():
    features = json.loads(BOUNDARY.read_text())["features"]
    land, outlines, names = [], [], []
    for item in features:
        geometry = shape(item["geometry"])
        if not geometry.intersects(REGION):
            continue
        names.append(item["properties"]["Prefecture Name"])
        for polygon in polygons(geometry.intersection(REGION)):
            land.append(polygon_path(polygon))
            outlines.extend(np.asarray(ring.coords) for ring in [polygon.exterior, *polygon.interiors])
    assert land
    return PlotPath.make_compound_path(*land), outlines, names


def map_style(ax, national=False):
    bounds = NATIONAL_BOUNDS if national else BOUNDS
    ax.set_xlim(*bounds[:2]); ax.set_ylim(*bounds[2:])
    ax.set_aspect(1/np.cos(np.deg2rad(np.mean(bounds[2:]))))
    ax.set_facecolor(SEA)
    ax.set_xticks([125, 135, 145, 155] if national else [135.45, 135.65, 135.85])
    ax.set_yticks([25, 35, 45] if national else [35.00, 35.15, 35.30])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}°E" if national else f"{x:.2f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}°N" if national else f"{y:.2f}°N"))
    ax.tick_params(labelsize=7.5, length=3, pad=4, width=.7, direction="out",
                   bottom=True, left=True, top=False, right=False, colors="#374151")
    for text in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        text.set_fontweight("bold")
    ax.grid(True, color="#97A6B1", linewidth=.6, alpha=.8, linestyle=(0, (4, 3)), zorder=5)
    for spine in ax.spines.values():
        spine.set_color("#374151"); spine.set_linewidth(.85)


def locator(ax, basemap):
    land, outlines, context = basemap
    if context:
        ax.add_patch(PathPatch(PlotPath.make_compound_path(*context), facecolor="#E4E8E9",
                               edgecolor="#B4BDC1", linewidth=.3, zorder=.4))
    for path in land:
        ax.add_patch(PathPatch(path, facecolor=LAND, edgecolor="none", zorder=.5))
    ax.add_collection(LineCollection(outlines, colors="#84929C", linewidth=.45, alpha=.85, zorder=3))
    ax.add_patch(Rectangle((BOUNDS[0], BOUNDS[2]), BOUNDS[1]-BOUNDS[0], BOUNDS[3]-BOUNDS[2],
                           fill=False, edgecolor=HOSPITAL, linewidth=1.2, zorder=7))
    ax.annotate("Kyoto example", xy=(np.mean(BOUNDS[:2]), np.mean(BOUNDS[2:])),
                xytext=(124, 32), fontsize=8, fontweight="bold", color=HOSPITAL,
                arrowprops={"arrowstyle": "->", "color": HOSPITAL, "lw": 1},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2},
                zorder=8)


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    population = pd.read_parquet(BASE)
    facilities = pd.read_parquet(FACILITIES)
    sources = [BASE, FACILITIES, BOUNDARY, *sorted((BASEMAP_DIR/"neighbour_context").glob("ne_110m_admin_0_countries.*"))]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    mask = population.Longitude.between(*BOUNDS[:2], inclusive="left") & population.Latitude.between(*BOUNDS[2:], inclusive="left")
    local = population.loc[mask].copy()
    assert local["Mesh Code"].is_unique and local["Local Total Population"].gt(0).all()
    eligible = facilities.loc[facilities["Included Medical Network Destination"]].copy()
    assert len(eligible) == 112401 and eligible["Medical Facility Category"].isin(["1", "2"]).all()
    points = eligible.loc[eligible.Longitude.between(*BOUNDS[:2], inclusive="left") & eligible.Latitude.between(*BOUNDS[2:], inclusive="left")].copy()
    assert points["Medical Facility ID"].is_unique
    row, col = mesh_raster_indices(local)
    grid, extent, rr, cc = native_raster(row, col, local["Local Total Population"].to_numpy(), np.nan, float)
    np.testing.assert_array_equal(grid[rr, cc], local["Local Total Population"])
    roads, road_hashes, road_receipts = road_geometries()
    hashes.update(road_hashes)
    land, outlines, prefectures = local_basemap()
    national_basemap = basemap_views([("national", None, NATIONAL_BOUNDS, None)])["national"]
    norm = LogNorm(vmin=float(local["Local Total Population"].min()), vmax=float(local["Local Total Population"].max()))
    cmap = plt.get_cmap("YlOrBr")
    sns.set_theme(context="paper", style="white", font="DejaVu Sans")
    plt.rcParams.update({"font.size": 9, "text.color": "#263746", "figure.facecolor": "white"})
    fig, layout = plt.subplots(2, 2, figsize=(190/25.4, 210/25.4))
    axes = layout.ravel()
    fig.subplots_adjust(left=.075, right=.97, top=.96, bottom=.12, wspace=.28, hspace=.38)
    for panel, ax in enumerate(axes):
        map_style(ax, national=panel == 0)
        ax.text(-.10, 1.06, "abcd"[panel], transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="left", va="top")
        if panel == 0:
            locator(ax, national_basemap)
            continue
        ax.add_patch(PathPatch(land, facecolor=LAND, edgecolor="none", zorder=.5))
        if panel in (1, 3):
            ax.imshow(np.ma.masked_invalid(grid), extent=extent, origin="upper",
                      interpolation="nearest", resample=False, cmap=cmap, norm=norm,
                      alpha=1 if panel == 1 else .28, aspect=ax.get_aspect(), zorder=1)
        if panel in (2, 3):
            ax.add_collection(LineCollection(roads, colors=ROAD if panel == 2 else "#7F8C97",
                                             linewidth=.25 if panel == 2 else .20,
                                             alpha=.85 if panel == 2 else .55, rasterized=True, zorder=2))
        ax.add_collection(LineCollection(outlines, colors="#84929C", linewidth=.45, alpha=.85, zorder=3))
        if panel == 3:
            for category, marker, colour, size in [("2", "o", CLINIC, 7), ("1", "^", HOSPITAL, 23)]:
                subset = points.loc[points["Medical Facility Category"].eq(category)]
                ax.scatter(subset.Longitude, subset.Latitude, s=size, marker=marker, color=colour,
                           edgecolors="white", linewidths=.25, alpha=.85, zorder=6)
    fig.canvas.draw()
    population_position = axes[1].get_position()
    cax = fig.add_axes([population_position.x0, population_position.y0-.06,
                        population_position.width, .015])
    ticks = [1, 10, 100, 1000, 10000]
    ticks = [value for value in ticks if norm.vmin <= value <= norm.vmax]
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", ticks=ticks)
    bar.ax.set_xticklabels([f"{value:,}" for value in ticks]); bar.ax.tick_params(labelsize=8)
    bar.set_label("Residents per 1-km mesh (log colour scale)", fontsize=8)
    axes[2].legend(handles=[Line2D([0], [0], color=ROAD, linewidth=1, label="Candidate vehicle roads")],
                   loc="upper left", bbox_to_anchor=(.02, -.11), frameon=False, fontsize=8)
    categories = points["Medical Facility Category"].value_counts()
    axes[3].legend(handles=[Line2D([0], [0], marker=marker, color="none", markerfacecolor=colour,
                                  markeredgecolor="white", markersize=size,
                                  label=f"{label} ({int(categories.get(category, 0)):,})")
                           for category, label, marker, colour, size in
                           [("1", "Hospitals", "^", HOSPITAL, 6), ("2", "General clinics", "o", CLINIC, 4)]],
                   loc="upper left", bbox_to_anchor=(.02, -.11), ncol=1, frameon=False, fontsize=8)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    bounds = [ax.get_window_extent(renderer) for ax in axes]
    np.testing.assert_allclose([[b.width, b.height] for b in bounds], [[bounds[0].width, bounds[0].height]]*4, atol=1e-8)
    labels = [text.get_window_extent(renderer) for ax in axes for text in [*ax.get_xticklabels(), *ax.get_yticklabels()]]
    assert not any(first.overlaps(second) for i, first in enumerate(labels) for second in labels[i+1:])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    check_publication_layout(fig, EXP)
    fig.savefig(OUT, dpi=600, facecolor="white")
    fig.savefig(EXP / "preview.png", dpi=150, facecolor="white")
    plt.close(fig)
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected
    receipt = {"title": TITLE, "placement": "appendix", "status": "generated_and_validated",
               "framework": "AnaSOP Sections 5-7 local data-and-method illustration",
               "bounds_west_east_south_north": list(BOUNDS), "crs": "EPSG:6668",
               "layout_rows_columns": [2, 2], "panel_order": ["national_locator", "population", "candidate_roads", "medical_overlay"],
               "national_locator_bounds": list(NATIONAL_BOUNDS), "example_region": "Kyoto city and surroundings",
               "window_selection": "Recognizable Kyoto setting showing an urban-to-mountain/peripheral settlement contrast. Display boundaries are not municipal boundaries. Selection is illustrative, not based on outcome extremity or statistical national representativeness.",
               "selection_geographic_context_source": SELECTION_SOURCE,
               "population_domain": "All published baseline mesh records in the window, distinct from the selected analysis sample",
               "population_meshes": len(local), "population_residents": int(local["Local Total Population"].sum()),
               "population_range": [norm.vmin, norm.vmax], "population_colour_scale": "logarithmic, positive counts untrimmed",
               "overlay_population_opacity": .28, "population_pixels_unchanged": True,
               "medical_points": len(points), "medical_category_counts": {"hospital": int(categories.get("1", 0)), "general_clinic": int(categories.get("2", 0))},
               "medical_scope": "Included Medical Network Destination only; no dental or pending-entry destinations",
               "road_sources": road_receipts, "road_line_parts": len(roads),
               "road_geometry": "Accepted graph source-feature IDs; full source curves clipped to the display window, no endpoint-chord replacement",
               "prefectures": prefectures, "identical_local_panel_extent_and_scale": True,
               "locator_scale": "Separate national scale; red rectangle denotes the exact local display window",
               "equal_frame_dimensions_check": "passed",
               "coordinate_label_overlap_check": "passed", "source_hashes_verified_after_render": hashes,
               "output": str(OUT.relative_to(ROOT)), "dpi": 600, "figure_size_mm": [190, 210], "publication_revision_authorization": "MILI-D-20261008-008", "preview": str((EXP / "preview.png").relative_to(ROOT))}
    (EXP/"validation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    OUT.with_suffix(".caption.md").write_text(
        f"{TITLE}. Appendix data-and-method illustration in a two-by-two layout. (a) National location context, with the Kyoto example window marked by a red rectangle. "
        "(b) Published residents per native 1-km mesh, using an unclipped logarithmic colour scale. (c) Candidate vehicle-road source curves. "
        "(d) Included registered hospitals (orange triangles) and general clinics (purple circles) over the same population raster at lower opacity and the same roads. "
        f"Panels b-d share bounds {BOUNDS[0]}-{BOUNDS[1]} degrees east and {BOUNDS[2]}-{BOUNDS[3]} degrees north and the same map scale; panel a uses a separate national scale. The display window contains {len(local):,} published meshes, "
        f"{int(local['Local Total Population'].sum()):,} residents, {int(categories.get('1', 0)):,} included hospitals and {int(categories.get('2', 0)):,} included general clinics. "
        "Kyoto city and surroundings provide a recognizable example of urban and peripheral settlement patterns. Kyoto City's official description of Sakyo Ward distinguishes southern urban areas, a central urban-rural mixture and northern mountain villages "
        f"([Kyoto City]({SELECTION_SOURCE})). The rectangular display window is not a municipal boundary, and its counts are not Kyoto municipal totals. It is an illustrative window, not a statistically representative national comparison or a new analysis sample. "
        "Population coverage is the published baseline input rather than the selected finite-distance analysis sample. Unpublished cells are transparent; no population or age interpolation is performed. "
        "Road geometry uses source records present in the accepted candidate graph, with curves clipped only for display. Dental and unresolved-entry medical records are excluded from plotted destinations in accordance with the accepted baseline. "
        "Population and medical registrations refer to 2020 and roads to 2024. Registered locations do not verify current opening, public availability or capacity. The illustration does not estimate maintenance effects, legal vehicle routes, speeds or travel time. "
        "Sources: e-Stat census mesh tables; MLIT National Land Numerical Information medical facilities, road centre lines and administrative areas. "
        "Administrative land and prefecture outlines reuse the existing N03 2024 display layer; pale blue indicates water. Coarse Natural Earth neighbouring-land context is used only in the national locator.\n")
    print(f"Saved: {OUT.relative_to(ROOT)}")
    print(f"Validated: {len(local):,} population cells, {len(points):,} included medical points; 2x2 layout with three equal-scale local views.")


if __name__ == "__main__":
    main()
