#!/usr/bin/env python3
"""National Candidate Medical Distance and Analysis Sample Coverage.

Framework: AnaSOP Sections 5-7, selected-sample descriptive baseline.
Candidate distance is displayed in kilometres; sample coverage is accounting.
Official N03 prefecture polygons provide a display-only land and boundary layer.
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
from publication_figure_checks import check_publication_layout
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import shapefile
from PIL import Image, TiffImagePlugin
from shapely.geometry import shape, box
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parents[2]
SLUG = "national_candidate_medical_distance_and_analysis_sample_coverage"
BASE = ROOT / "data/processed/medical-baseline/medical_mesh_baseline.parquet"
SAMPLE = ROOT / "data/processed/medical-analysis-sample/medical_mesh_analysis_preprocessed.parquet"
AGE = ROOT / "data/processed/medical-baseline/medical_age_support_baseline.parquet"
OUT = ROOT / f"data/results/figures/Figure_{SLUG}.png"
EXP = ROOT / "data/exp/figure-table-generation" / SLUG
BASEMAP_DIR = ROOT / "data/exp/cartographic-basemap"
BOUNDARY = BASEMAP_DIR / "prefecture_display_boundaries.geojson"
SEA = "#EAF4F8"
LAND = "#FAFAF6"
RETAINED = "#D3D6D8"
NO_PATH = "#D55E00"
AGE_MISSING = "#CC79A7"
PENDING = "#374151"
RASTERS = OUT.parent / "rasters" / SLUG
CLASS_CODES = {"retained": 1, "undefined_age": 2, "no_path": 3, "pending": 4}


def mesh_vertices(frame):
    """Reconstruct standard 1-km geographic mesh rectangles from their code."""
    code = frame["Mesh Code"].astype(str)
    assert code.str.fullmatch(r"\d{8}").all()
    lat = (code.str[:2].astype(int).to_numpy() * 2 / 3
           + code.str[4].astype(int).to_numpy() / 12
           + code.str[6].astype(int).to_numpy() / 120 + 1 / 240)
    lon = (code.str[2:4].astype(int).to_numpy() + 100
           + code.str[5].astype(int).to_numpy() / 8
           + code.str[7].astype(int).to_numpy() / 80 + 1 / 160)
    assert np.allclose(lon, frame["Longitude"], atol=1e-8, rtol=0)
    assert np.allclose(lat, frame["Latitude"], atol=1e-8, rtol=0)
    corners = np.array([[-1/160, -1/240], [1/160, -1/240],
                        [1/160, 1/240], [-1/160, 1/240]])
    return np.column_stack([lon, lat])[:, None, :] + corners[None, :, :]


def mesh_raster_indices(frame):
    """Integer JIS third-level mesh indices, with no interpolation or snapping."""
    code = frame["Mesh Code"].astype(str)
    row = (code.str[:2].astype(int).to_numpy() * 80
           + code.str[4].astype(int).to_numpy() * 10
           + code.str[6].astype(int).to_numpy())
    col = ((code.str[2:4].astype(int).to_numpy() + 100) * 80
           + code.str[5].astype(int).to_numpy() * 10
           + code.str[7].astype(int).to_numpy())
    assert np.allclose((col+.5)/80, frame["Longitude"], atol=1e-8, rtol=0)
    assert np.allclose((row+.5)/120, frame["Latitude"], atol=1e-8, rtol=0)
    return row, col


def native_raster(row, col, values, fill, dtype):
    """One observed mesh per north-up pixel; unobserved locations stay NoData."""
    ymin, ymax, xmin, xmax = int(row.min()), int(row.max()), int(col.min()), int(col.max())
    rows, cols = ymax-row, col-xmin
    grid = np.full((ymax-ymin+1, xmax-xmin+1), fill, dtype=dtype)
    assert len(np.unique(rows*grid.shape[1]+cols)) == len(row)
    grid[rows, cols] = values
    extent = (xmin/80, (xmax+1)/80, ymin/120, (ymax+1)/120)
    return grid, extent, rows, cols


def save_geotiff(path, grid, extent, nodata, description):
    """Use installed Pillow to write GeoTIFF 1.1 tags (JGD2011, PixelIsArea)."""
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    entries = {
        33550: (12, (1/80, 1/120, 0.0)),
        33922: (12, (0.0, 0.0, 0.0, extent[0], extent[3], 0.0)),
        34735: (3, (1, 1, 1, 4, 1024, 0, 1, 2, 1025, 0, 1, 1,
                    2048, 0, 1, 6668, 2054, 0, 1, 9102)),
        42113: (2, str(nodata)),
        270: (2, description),
    }
    for key, (tag_type, value) in entries.items():
        tags.tagtype[key] = tag_type
        tags[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path, compression="tiff_deflate", tiffinfo=tags)
    with Image.open(path) as saved:
        np.testing.assert_array_equal(np.asarray(saved), grid)
        for key in entries:
            assert saved.tag_v2[key] == tags[key], f"GeoTIFF tag {key} changed."


def build_rasters(baseline, classes, distance, retained):
    row, col = mesh_raster_indices(baseline)
    codes = np.array([CLASS_CODES[name] for name in classes], dtype=np.uint8)
    categorical, extent, rr, cc = native_raster(row, col, codes, 0, np.uint8)
    continuous, same_extent, _, _ = native_raster(
        row, col, np.where(retained, distance, np.nan), np.nan, np.float64)
    assert extent == same_extent
    np.testing.assert_array_equal(categorical[rr, cc], codes)
    np.testing.assert_array_equal(continuous[rr[retained], cc[retained]], distance[retained])
    assert np.count_nonzero(categorical) == len(baseline)
    assert np.isfinite(continuous).sum() == retained.sum()
    category_path = RASTERS / "analysis_sample_status.tif"
    distance_path = RASTERS / "candidate_medical_distance_km.tif"
    save_geotiff(category_path, categorical, extent, 0,
                 "Sample status: 1 retained; 2 undefined age share; 3 no medical path; 4 pending road entry; 0 NoData.")
    exported_distance = np.where(np.isfinite(continuous), continuous, -9999).astype(np.float32)
    max_error = float(np.max(np.abs(exported_distance[rr[retained], cc[retained]].astype(float)-distance[retained])))
    assert max_error < 3e-6, "Float32 export error exceeds 3 millimetres."
    save_geotiff(distance_path, exported_distance, extent, -9999,
                 "Selected-sample candidate medical network distance, kilometres; -9999 NoData.")
    metadata = {
        "crs": "EPSG:6668", "datum": "JGD2011", "raster_type": "PixelIsArea",
        "shape_rows_columns": list(categorical.shape),
        "extent_west_east_south_north": list(extent),
        "pixel_size_longitude_latitude_degrees": [1/80, 1/120],
        "grid": "Native JIS third-level geographic mesh (nominal 1 km), not uniform 1000-m squares",
        "interpolation": "none", "aggregation": "none", "one_mesh_per_pixel": True,
        "mesh_pixel_roundtrip": "passed", "geotiff_readback": "passed",
        "categorical_nodata": 0, "category_codes": CLASS_CODES,
        "category_counts": {name: int(np.sum(categorical == value)) for name, value in CLASS_CODES.items()},
        "distance_nodata": -9999, "distance_unit": "km", "distance_export_dtype": "float32",
        "finite_distance_pixels": int(np.isfinite(continuous).sum()),
        "distance_max_export_rounding_error_km": max_error,
        "figure_distance_values": "original float64 values, without rounding",
        "files": [str(path.relative_to(ROOT)) for path in [category_path, distance_path]],
        "encoding_reference": "https://docs.ogc.org/is/19-008r4/19-008r4.html",
    }
    (RASTERS / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return row, col, codes, metadata


def exclusion_detail(longitude, latitude, classes):
    """Display the 1-degree mainland cell with the most excluded mesh records."""
    excluded = (classes != "retained") & (latitude >= 30)
    tiles = pd.DataFrame({"west": np.floor(longitude[excluded]).astype(int),
                          "south": np.floor(latitude[excluded]).astype(int)})
    ranking = (tiles.groupby(["west", "south"]).size().rename("excluded_meshes")
               .reset_index().sort_values(["excluded_meshes", "west", "south"],
                                           ascending=[False, True, True]))
    peak = ranking.iloc[0]
    west, south = int(peak["west"]), int(peak["south"])
    bounds = (west, west+1, south, south+1)
    mask = ((longitude >= west) & (longitude < west+1)
            & (latitude >= south) & (latitude < south+1))
    counts = {name: int(np.sum(mask & (classes == name))) for name in CLASS_CODES}
    assert sum(value for key, value in counts.items() if key != "retained") == int(peak["excluded_meshes"])
    position = [.69, .065, .285, .225]
    region = ("Exclusion detail", mask, bounds, position)
    receipt = {"panel": "b", "bounds_west_east_south_north": list(bounds),
               "position": position, "display_classes": counts,
               "excluded_meshes": int(peak["excluded_meshes"]),
               "selection": "Most excluded meshes in a 1-degree longitude/latitude cell among main-island views; ties ordered west then south",
               "purpose": "Visibility detail, not a representative sample or inference subset",
               "repeats_main_view_pixels": True, "added_to_national_counts": False}
    return region, receipt


def style_map(ax, bounds, *, inset=False):
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(1 / np.cos(np.deg2rad((ymin+ymax)/2)))
    ax.set_facecolor(SEA)
    if inset:
        ax.set_xticks([xmin + .2*(xmax-xmin), xmax - .2*(xmax-xmin)])
        ax.set_yticks([ymin + .2*(ymax-ymin), ymax - .2*(ymax-ymin)])
        ax.tick_params(labelsize=6.5, length=2.5, width=.65, pad=3,
                       direction="out", colors="#374151",
                       bottom=True, left=True, top=False, right=False,
                       labelbottom=True, labelleft=True)
        decimals = 2 if xmax-xmin < .2 else 1
    else:
        ax.set_xticks([130, 134, 138, 142, 146])
        ax.set_yticks([30, 34, 38, 42, 46])
        ax.tick_params(labelsize=9.5, length=3.5, width=.75, pad=5,
                       direction="out", colors="#374151",
                       bottom=True, left=True, top=False, right=False)
        decimals = 0
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.{decimals}f}°N"))
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    ax.grid(True, color="#97A6B1", linewidth=.6 if inset else .75,
            alpha=.8, linestyle=(0,(4,3)), zorder=4)
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(.65 if inset else .85)


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [part for item in geometry.geoms for part in polygons(item)]
    return []


def polygon_path(polygon):
    """Compound path with oriented holes, so inland water stays pale blue."""
    polygon = orient(polygon, sign=1)
    vertices, codes = [], []
    for ring in [polygon.exterior, *polygon.interiors]:
        coordinates = np.asarray(ring.coords)
        vertices.append(coordinates)
        code = np.full(len(coordinates), PlotPath.LINETO, dtype=np.uint8)
        code[0], code[-1] = PlotPath.MOVETO, PlotPath.CLOSEPOLY
        codes.append(code)
    return PlotPath(np.concatenate(vertices), np.concatenate(codes))


def basemap_views(regions):
    features = json.loads(BOUNDARY.read_text())["features"]
    assert len(features) == 47
    prefectures = [(item["properties"]["Prefecture Name"], shape(item["geometry"])) for item in features]
    reader = shapefile.Reader(str(BASEMAP_DIR / "neighbour_context/ne_110m_admin_0_countries.shp"))
    field = [x[0] for x in reader.fields[1:]].index("ADM0_A3")
    neighbours = [shape(item.shape.__geo_interface__) for item in reader.iterShapeRecords()
                  if item.record[field] != "JPN"]
    views = {}
    displayed = set()
    for name, _, bounds, _ in regions:
        xmin, xmax, ymin, ymax = bounds
        viewport = box(xmin, ymin, xmax, ymax)
        paths, outlines, context = [], [], []
        for prefecture, geometry in prefectures:
            if not geometry.intersects(viewport):
                continue
            displayed.add(prefecture)
            for polygon in polygons(geometry.intersection(viewport)):
                paths.append(polygon_path(polygon))
                outlines.extend(np.asarray(ring.coords) for ring in [polygon.exterior, *polygon.interiors])
        for geometry in neighbours:
            if geometry.intersects(viewport):
                context.extend(polygon_path(part) for part in polygons(geometry.intersection(viewport)))
        # One compound artist avoids creating thousands of island patch objects.
        land = [PlotPath.make_compound_path(*paths)] if paths else []
        views[name] = (land, outlines, context)
    assert len(displayed) == 47
    return views


def main(view="comparison", *, dpi=150, figure_size_mm=None):
    """Render separate manuscript/appendix maps or the retained comparison preview."""
    assert view in ("comparison", "medical", "coverage")
    title = {"comparison": "National Candidate Medical Distance and Analysis Sample Coverage",
             "medical": "National Candidate Medical Distance",
             "coverage": "Analysis Sample Coverage"}[view]
    slug = {"comparison": SLUG, "medical": "national_candidate_medical_distance",
            "coverage": "analysis_sample_coverage"}[view]
    output = (EXP / "comparison_preview.png" if view == "comparison"
              else OUT.parent / f"Figure_{slug}.png")
    exp = EXP if view == "comparison" else EXP.parent / slug
    panels = [0, 1] if view == "comparison" else [0 if view == "medical" else 1]
    source_hashes_before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [BASE, SAMPLE, AGE, BOUNDARY]}
    baseline = pd.read_parquet(BASE)
    sample = pd.read_parquet(SAMPLE)
    age = pd.read_parquet(AGE)
    assert baseline["Mesh Code"].is_unique and sample["Mesh Code"].is_unique
    invalid_support = age.loc[age["Population Aged 65+ Share Among Known Ages"].isna(), "Publication Support Code"]
    age_invalid = baseline["Publication Support Code"].isin(invalid_support).to_numpy()
    retained = (~age_invalid) & baseline["Candidate Medical Network Distance"].notna().to_numpy()
    pd.testing.assert_frame_equal(sample, baseline.loc[retained].reset_index(drop=True))
    assert len(sample) == 175612 and int(sample["Local Total Population"].sum()) == 125874788
    vertices = mesh_vertices(baseline)
    longitude = baseline["Longitude"].to_numpy()
    latitude = baseline["Latitude"].to_numpy()
    # Mutually exclusive display classes: age exclusion takes precedence for overlap.
    classes = np.full(len(baseline), "retained", dtype=object)
    classes[~retained] = "no_path"
    classes[baseline["Medical Route Status"].eq("entry_pending")] = "pending"
    classes[age_invalid] = "undefined_age"
    counts = {name: int(np.sum(classes == name)) for name in ["retained", "undefined_age", "no_path", "pending"]}
    assert sum(counts.values()) == len(baseline)
    distance = baseline["Candidate Medical Network Distance"].to_numpy() / 1000
    norm = PowerNorm(gamma=.5, vmin=0, vmax=float(np.max(distance[retained])))
    cmap = plt.get_cmap("YlOrRd")
    raster_row, raster_col, class_values, raster_metadata = build_rasters(baseline, classes, distance, retained)
    class_cmap = ListedColormap([RETAINED, AGE_MISSING, NO_PATH, PENDING])
    class_norm = BoundaryNorm([.5, 1.5, 2.5, 3.5, 4.5], class_cmap.N)
    regions = [
        ("Main islands", latitude >= 30, (128.2, 146.4, 29.8, 46.1), None),
        ("Nansei islands", (latitude < 30) & (longitude < 134),
         (122.5, 131.7, 23.75, 30.15), [.10, .69, .38, .255]),
        ("Ogasawara", (latitude < 30) & (longitude >= 134) & (longitude < 150),
         (140.95, 142.55, 24.45, 27.45), [.10, .37, .17, .23]),
        ("Minamitorishima", (latitude < 30) & (longitude >= 150),
         (153.945, 154.02, 24.265, 24.325), [.35, .51, .165, .125]),
    ]
    coverage = np.zeros(len(baseline), dtype=int)
    for name, mask, bounds, _ in regions:
        coverage += mask.astype(int)
        xy = vertices[mask]
        assert np.all(xy[:,:,0] >= bounds[0]) and np.all(xy[:,:,0] <= bounds[1])
        assert np.all(xy[:,:,1] >= bounds[2]) and np.all(xy[:,:,1] <= bounds[3])
    assert np.all(coverage == 1), "Every mesh must be represented exactly once."
    detail_region, detail_receipt = exclusion_detail(longitude, latitude, classes)
    view_regions = [*regions, detail_region]
    basemaps = basemap_views(view_regions)
    raster_views = {}
    for name, mask, _, _ in view_regions:
        category, extent, _, _ = native_raster(
            raster_row[mask], raster_col[mask], class_values[mask], 0, np.uint8)
        medical, medical_extent, _, _ = native_raster(
            raster_row[mask], raster_col[mask], np.where(retained[mask], distance[mask], np.nan), np.nan, np.float64)
        assert extent == medical_extent
        raster_views[name] = (category, medical, extent)

    sns.set_theme(context="paper", style="white", font="DejaVu Sans")
    plt.rcParams.update({"font.size":9, "text.color":"#263746", "figure.facecolor":"white",
                         "axes.labelcolor":"#374151", "xtick.color":"#374151", "ytick.color":"#374151"})
    size_inches = ((7.4*len(panels), 8.8) if figure_size_mm is None
                   else tuple(value / 25.4 for value in figure_size_mm))
    fig, axes = plt.subplots(1, len(panels), figsize=size_inches, squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(left=.045 if len(panels) == 2 else .085,
                        right=.985 if len(panels) == 2 else .975,
                        top=.965, bottom=.16, wspace=.10)
    for panel, ax in zip(panels, axes):
        panel_regions = regions if panel == 0 else view_regions
        for name, mask, bounds, position in panel_regions:
            target = ax if position is None else ax.inset_axes(position)
            style_map(target, bounds, inset=position is not None)
            land_paths, outlines, neighbours = basemaps[name]
            for path in neighbours:
                target.add_patch(PathPatch(path, facecolor="#F0F1EF", edgecolor="#AAB4BA",
                                           linewidth=.3, zorder=.5))
            for path in land_paths:
                target.add_patch(PathPatch(path, facecolor=LAND, edgecolor="none", zorder=.8))
            if panel == 0:
                category, medical, extent = raster_views[name]
                target.imshow(np.ma.masked_where(category == 0, category),
                              cmap=ListedColormap(["#ECEFF1"]*4), norm=class_norm,
                              extent=extent, origin="upper", interpolation="nearest",
                              resample=False, aspect=target.get_aspect(), zorder=1)
                target.imshow(np.ma.masked_invalid(medical), cmap=cmap, norm=norm,
                              extent=extent, origin="upper", interpolation="nearest",
                              resample=False, aspect=target.get_aspect(), zorder=2)
            else:
                category, _, extent = raster_views[name]
                target.imshow(np.ma.masked_where(category == 0, category),
                              cmap=class_cmap, norm=class_norm,
                              extent=extent, origin="upper", interpolation="nearest",
                              resample=False, aspect=target.get_aspect(), zorder=2)
            target.add_collection(LineCollection(outlines, colors="#84929C", linewidth=.45,
                                                 alpha=.85, zorder=3))
            if position is not None:
                target.text(.02, 1.04, name, transform=target.transAxes, fontsize=7,
                            ha="left", va="bottom", color="#374151")
        if panel == 1:
            west, east, south, north = detail_region[2]
            ax.add_patch(Rectangle((west, south), east-west, north-south,
                                   fill=False, edgecolor="#374151", linewidth=.8,
                                   linestyle=(0, (4, 2)), zorder=5))
        if len(panels) > 1:
            ax.text(-.055, 1.04, "ab"[panel], transform=ax.transAxes, fontsize=12,
                    fontweight="bold", ha="left", va="top")
    if 0 in panels:
        color_ax = fig.add_axes([.06, .105, .423, .022] if len(panels) == 2 else [.085, .105, .89, .022])
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        bar = fig.colorbar(mappable, cax=color_ax, orientation="horizontal",
                           ticks=[0, 1, 5, 10, 20, 40, norm.vmax])
        bar.ax.set_xticklabels(["0", "1", "5", "10", "20", "40", f"{norm.vmax:.1f}"])
        bar.set_label("Candidate network distance to a registered hospital or clinic (km)", fontsize=9)
        bar.ax.tick_params(labelsize=8)
    handles = [Patch(facecolor=color, edgecolor="none", label=f"{label} ({counts[key]:,})")
               for key, label, color in [
                   ("retained", "Retained sample", RETAINED),
                   ("undefined_age", "Undefined age share", AGE_MISSING),
                   ("no_path", "No candidate medical path", NO_PATH),
                   ("pending", "Road entry pending", PENDING)]]
    if 1 in panels:
        fig.legend(handles=handles, loc="lower left",
                   bbox_to_anchor=(.545, .047) if len(panels) == 2 else (.07, .047),
                   ncol=2, frameon=False, fontsize=8, columnspacing=1.5, handlelength=1.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    coordinate_labels = []
    main_labels = []
    inset_names = []
    main_meshes_hidden_by_insets = 0
    for parent in axes:
        main_labels.extend(text.get_window_extent(renderer)
                           for text in [*parent.get_xticklabels(), *parent.get_yticklabels()])
        main_points = parent.transData.transform(np.column_stack([longitude[latitude >= 30], latitude[latitude >= 30]]))
        for child in parent.child_axes:
            frame = child.get_window_extent(renderer)
            main_meshes_hidden_by_insets += int(np.sum(
                (main_points[:,0] >= frame.x0) & (main_points[:,0] <= frame.x1)
                & (main_points[:,1] >= frame.y0) & (main_points[:,1] <= frame.y1)))
            inset_names.extend(text.get_window_extent(renderer) for text in child.texts)
            for text in [*child.get_xticklabels(), *child.get_yticklabels()]:
                bounds = text.get_window_extent(renderer)
                assert not frame.overlaps(bounds), "Inset coordinates must stay outside the frame."
                coordinate_labels.append(bounds)
    assert len(coordinate_labels) == sum(12 if panel == 0 else 16 for panel in panels)
    assert not any(first.overlaps(second) for i, first in enumerate(coordinate_labels)
                   for second in coordinate_labels[i+1:]), "Inset coordinate labels overlap."
    assert not any(label.overlaps(other) for label in coordinate_labels
                   for other in main_labels + inset_names), "Inset coordinates collide with other labels."
    assert main_meshes_hidden_by_insets == 0, "Repositioned insets cover main-map mesh origins."
    output.parent.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)
    if figure_size_mm is not None:
        check_publication_layout(fig, exp)
        fig.savefig(exp / "preview.png", dpi=150, facecolor="white")
    fig.savefig(output, dpi=dpi, bbox_inches="tight" if figure_size_mm is None else None,
                facecolor="white")
    plt.close(fig)
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
               for name, expected in source_hashes_before.items()), "Source changed during export."
    exp.mkdir(parents=True, exist_ok=True)
    receipt = {"title":title, "placement":{"comparison":"comparison_preview", "medical":"main_manuscript", "coverage":"appendix"}[view],
               "subpanels":len(panels), "panel_letters":len(panels) > 1,
               "status":"generated_and_validated", "framework":"AnaSOP Sections 5-7 selected-sample baseline",
               "baseline_meshes":len(baseline), "selected_meshes":len(sample),
               "selected_population":int(sample["Local Total Population"].sum()),
               "display_classes":counts, "all_meshes_represented_once":True,
               "unique_coverage_scope":"Main and island views partition the baseline; the exclusion detail inset intentionally repeats their original pixels",
               "region_counts":{name:int(mask.sum()) for name,mask,_,_ in regions},
               "distance_range_km":[float(np.min(distance[retained])), norm.vmax],
               "distance_values_clipped":False, "colour_normalization":"square root",
               "excluded_and_island_centres_enlarged_for_visibility":False,
               "mesh_rendering":"native categorical and continuous rasters, nearest-neighbour display",
               "raster_export":raster_metadata,
               "retained_mesh_colour":RETAINED,
               "exclusion_detail_inset":detail_receipt if 1 in panels else None,
               "administrative_boundary_style":{"colour":"#84929C", "linewidth":.45, "alpha":.85},
               "graticule_style":{"colour":"#97A6B1", "main_linewidth":.75,
                                   "inset_linewidth":.6, "alpha":.8, "dash_pattern":[4,3],
                                   "main_label_fontsize":9.5, "inset_label_fontsize":6.5,
                                   "label_fontweight":"bold", "ticks":"outward"},
               "basemap":"Official MLIT N03 2024 prefecture polygons; municipal polygons dissolved by prefecture",
               "prefecture_count":47, "sea_fill":SEA, "land_fill":LAND,
               "explanatory_footer_removed":True,
               "inset_coordinates_inside_frames":False,
               "inset_frames_repositioned":True,
               "inset_positions":{name:position for name,_,_,position in regions if position is not None},
               "main_mesh_origins_hidden_by_insets":main_meshes_hidden_by_insets,
               "inset_coordinate_labels":len(coordinate_labels),
               "inset_coordinate_overlap_check":"passed",
               "basemap_sha256":hashlib.sha256(BOUNDARY.read_bytes()).hexdigest(),
               "source_sha256":{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [BASE,SAMPLE,AGE]},
               "output":str(output.relative_to(ROOT)), "dpi":dpi,
               "figure_size_mm":[value * 25.4 for value in size_inches],
               "source_hashes_verified_after_export":source_hashes_before,
               "source_data_unchanged":True}
    (exp / "validation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    medical_caption = "Candidate network distance from each retained population mesh to the nearest included registered hospital or general clinic; grey cells are excluded baseline records. "
    coverage_caption = (
        "Geographic sample inclusion and mutually exclusive display classes. Undefined-age status takes precedence for the single overlapping age/path exclusion. "
        f"Retained cells are light grey. The dashed locator marks the Exclusion detail window, which repeats original 1-km raster cells without adding them to national counts. The window was selected as the main-island 1-degree geographic cell containing the most excluded meshes ({detail_receipt['excluded_meshes']}); it is a visibility illustration, not a representative regional comparison. "
    )
    panel_caption = ("(a) "+medical_caption+"(b) "+coverage_caption if view == "comparison"
                     else medical_caption if view == "medical" else coverage_caption)
    output.with_suffix(".caption.md").write_text(title+". "+panel_caption+
        "The main sample contains 175,612 meshes and 125,874,788 residents. Observed census population mesh cells are overlaid on official 2024 prefecture land polygons and boundaries; pale blue indicates water. "
        "The main and island views partition all 176,964 original mesh locators exactly once. Island views have different map scales. "
        "Native JIS third-level geographic mesh rasters use one pixel per mesh with no point-marker enlargement, interpolation, smoothing or cell aggregation. Unobserved locations remain transparent over the basemap; excluded records are distinct from unobserved locations. "
        + ("Distance colours use square-root normalization over the full finite range without clipping or tail deletion. " if 0 in panels else "") +
        "Population and registered facilities refer to 2020; roads refer to 2024. The undirected candidate-network distance is not observed travel time, operating-service availability or individual older-resident access. "
        "Administrative layer: Ministry of Land, Infrastructure, Transport and Tourism, National Land Numerical Information, Administrative Areas (2024-01-01), CC BY 4.0, "
        "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2024.html. Municipal polygons were dissolved by prefecture and simplified by 0.0005 degrees for display only. "
        "Coarse neighbouring-land context: Natural Earth Admin 0 Countries, "
        "https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/.\n")
    print(f"Saved: {output.relative_to(ROOT)}")
    print(f"Validation: {len(sample):,} selected meshes; {len(baseline):,} locators represented once in the main/island partition.")


if __name__ == "__main__":
    main()
