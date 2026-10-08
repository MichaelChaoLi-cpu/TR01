"""Prepare the adopted Appendix A/B/C review index from item-specific outputs.

No model is estimated and no scientific values are changed. Final figures and
workbooks stay at their stable paths; contact sheets are review aids only.
Run locally: .venv/bin/python src/exp/prepare_appendix_delivery.py
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "data/exp/.matplotlib-cache"))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont
from matplotlib.font_manager import FontProperties
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data/exp/figure-table-generation/appendix-production"
INDEX = ROOT / "data/results/appendices/appendix_index.md"
STATE = ROOT / "data/exp/figure-table-generation/production_state.json"
SCOPES = {
    "A": "Baseline sample and input documentation",
    "B": "Independent observed-pavement comparison: 70,049 common roads",
    "C": "Independent bridge comparison: 46,688 conditional units",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contact_sheet(group, items):
    """Display faithful reduced previews; never change scientific figure files."""
    width, margin, header = 900, 24, 108
    pictures = []
    for item in items:
        with Image.open(ROOT / item["image"]) as image:
            picture = image.convert("RGB")
        picture.thumbnail((width - 2 * margin, 1300), Image.Resampling.LANCZOS)
        pictures.append(picture)
    sheet = Image.new("RGB", (width * len(items), max(p.height for p in pictures) + header + margin), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(findfont("DejaVu Sans"), 21)
    bold = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", weight="bold")), 25)
    for i, (item, picture) in enumerate(zip(items, pictures)):
        x = i * width
        draw.text((x + margin, 12), f"Appendix {group} | Figure {group}{i + 1}", fill="black", font=bold)
        draw.text((x + margin, 48), SCOPES[group].split(":")[0], fill="#263746", font=font)
        draw.text((x + margin, 77), "600 dpi originals; reduced review preview", fill="#4A5560", font=font)
        sheet.paste(picture, (x + (width - picture.width) // 2, header))
    path = EXP / f"appendix_{group.lower()}_preview.png"
    sheet.save(path)
    return str(path.relative_to(ROOT))


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text())
    outputs = {item["title"]: item for item in state["outputs"]}
    sop = (ROOT / "docs/AnaSOP.md").read_text()
    section = sop.split("## 8. Figure and Table Plan", 1)[1]
    entries, previews = [], {}
    appendix_items = [item for item in state["outputs"] if item.get("placement") == "appendix"]
    figure_count = sum("image" in item for item in appendix_items)
    table_count = sum("workbook" in item for item in appendix_items)
    pending_figures = sum("image" in item and item["status"] != "done" for item in appendix_items)
    pending_tables = sum("workbook" in item and item["status"] != "done" for item in appendix_items)
    review_note = (f"{pending_figures} appendix figures and {pending_tables} appendix tables await final human review."
                   if pending_figures + pending_tables else
                   f"All appendix outputs are human-accepted under {state['latest_output_acceptance_decision']}; earlier acceptance records are retained.")
    lines = ["# Appendices A, B and C", "", f"{figure_count} figures and {table_count} tables from the adopted plan. Scientific outputs retain their individual scripts and stable file paths. Contact sheets are reduced review aids; all appendix figure originals are exported at 600 dpi.", "", f"Main manuscript figures and its policy table were accepted under MILI-D-20261008-004. {review_note}", ""]
    if state.get("critique_resolution_authorization"):
        lines += ["Publication export/readability revisions were authorized under " + state["critique_resolution_authorization"] + ". Revised PNGs have separate agent visual-review and validation records; the preceding human-acceptance records are preserved as history, without inferring additional human acceptance.", ""]
    for group, members in state["appendix_groups"].items():
        lines += [f"## Appendix {group}: {SCOPES[group]}", ""]
        group_figures = [outputs[title] for title in members["figures"]]
        previews[group] = contact_sheet(group, group_figures)
        for kind in ["figures", "tables"]:
            for number, title in enumerate(members[kind], 1):
                item = outputs[title]
                assert item["placement"] == "appendix" and item["appendix_group"] == group
                row = next(line for line in section.splitlines() if line.startswith(f"| {title} |"))
                assert row.endswith("| done |" if item["status"] == "done" else "| pending |")
                path = ROOT / item["image" if kind == "figures" else "workbook"]
                validation_path = ROOT / item["validation"]
                validation = json.loads(validation_path.read_text())
                caption = ROOT / item["caption"]
                assert path.is_file() and caption.is_file() and caption.read_text().strip()
                script = ROOT / item.get("script", f"src/analyses/{'figure' if kind == 'figures' else 'table'}_{path.stem.split('_', 1)[1]}.py")
                assert script.is_file()
                entry = {"label": f"{'Figure' if kind == 'figures' else 'Table'} {group}{number}", "title": title, "appendix": group, "status": item["status"], "artifact": str(path.relative_to(ROOT)), "caption": item["caption"], "script": str(script.relative_to(ROOT)), "validation": item["validation"], "sha256": digest(path)}
                if kind == "figures":
                    with Image.open(path) as image:
                        dpi = list(image.info["dpi"])
                        dimensions = list(image.size)
                    assert all(abs(value - 600) < .1 for value in dpi), (title, dpi)
                    entry.update(dpi=dpi, dimensions=dimensions, figure_size_mm=[v / d * 25.4 for v, d in zip(dimensions, dpi)])
                    item.update(dpi=dpi, dimensions=dimensions)
                    if "figure_size_mm" in validation:
                        item["figure_size_mm"] = validation["figure_size_mm"]
                else:
                    workbook = load_workbook(path, data_only=False)
                    assert not workbook._external_links
                    assert not any(cell.data_type == "f" for sheet in workbook for row in sheet for cell in row)
                    data = pd.read_excel(path, sheet_name=0)
                    assert data.notna().all().all()
                    csv_path = validation_path.with_name("table_data.csv")
                    if csv_path.exists():
                        pd.testing.assert_frame_equal(data, pd.read_csv(csv_path), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
                    if "Requested budget (%)" in data:
                        np.testing.assert_allclose(data["Actual spending (%)"] + data["Unused budget (%)"], data["Requested budget (%)"], rtol=0, atol=1e-10)
                    if title == "Baseline Analysis Sample Selection":
                        spec = importlib.util.spec_from_file_location("tr01_table_preview", ROOT / "src/analyses/table_allocation_rules_and_shared_scenario_assumptions.py")
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        formatted = data.copy()
                        for column in ["Mesh count", "Resident count"]:
                            formatted[column] = formatted[column].map(lambda value: f"{value:,.0f}")
                        formatted["Share of baseline population"] = formatted["Share of baseline population"].map(lambda value: f"{value * 100:.6f}%")
                        item["preview"] = module.preview_table(formatted, [47, 10, 16, 19, 27], EXP / "table_a1_preview.png")["path"]
                    if item.get("preview"):
                        entry["preview"] = item["preview"]
                    entry.update(rows=len(data), columns=len(data.columns), local_workbook_readback="passed", formula_cells=0, external_links=0)
                if item.get("human_acceptance_decision"):
                    entry["human_acceptance_decision"] = item["human_acceptance_decision"]
                entries.append(entry)
                label = entry["label"]
                status = "Human-accepted" if item["status"] == "done" else "Ready for human review"
                lines += [f"### {label}: {title}", "", f"Status: {status}.", "", f"[{'600 dpi PNG' if kind == 'figures' else 'Excel workbook'}]({path}) | [Caption and interpretation limits]({caption}) | [Reproducible Python script]({script})", ""]
                if kind == "figures":
                    lines += [f"![{label}]({path})", ""]
        if group == "A":
            lines += ["Exclusion rows overlap in the sample table. Map classes are mutually exclusive; the undefined-age class takes precedence. The Kyoto window illustrates input relationships rather than national representativeness.", ""]
        elif group == "B":
            lines += ["Only observed common-sample roads deteriorate in this supplement; all other network roads remain at baseline cost. Its budget denominator and outcomes differ from the replacement national pavement model. Residual-burden maps retain the adopted linear 0–30 resident reference-kilometre colour scale, with a pale-blue zero and stronger low-value palette transitions; values above 30 are saturated and underlying values are unchanged.", ""]
        else:
            lines += ["Bridge units use area-relative costs and equally weighted single-road closure events. They do not share a pavement budget. Demographic figures use 175,485 matched origins and 125,874,510 census residents at the matched 2020 reference; 127 unmatched origins and 278 residents are excluded only from those comparisons. Original area groups stay fixed; future elderly-resident locations are not inferred.", ""]
    assert len(entries) == figure_count + table_count
    assert sum("dpi" in item for item in entries) == figure_count
    assert sum(item["status"] != "done" for item in entries) == pending_figures + pending_tables
    INDEX.write_text("\n".join(lines))
    report = {"status": "passed_appendix_delivery_checks", "figures": figure_count, "tables": table_count, "all_appendix_figures_600dpi": True, "human_review_pending": pending_figures + pending_tables, "main_outputs_accepted": 7, "index": str(INDEX.relative_to(ROOT)), "review_previews": previews, "entries": entries, "scope": "adopted Appendices A/B/C and human-requested figure/table revisions; no new estimation"}
    (EXP / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    state.update(output_review="main_manuscript_accepted_appendices_ready_for_review" if pending_figures + pending_tables else "all_planned_outputs_human_accepted", next_ready_action=f"human review of {pending_figures} appendix figures and {pending_tables} appendix tables" if pending_figures + pending_tables else "critique-research-outputs before manuscript-input assembly", appendix_delivery=str(INDEX.relative_to(ROOT)), appendix_delivery_validation=str((EXP / "validation.json").relative_to(ROOT)))
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
