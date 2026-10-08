"""Confirm the local N13-2024 inventory and pilot provenance without edits to raw data."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
ACQ = ROOT / "data/exp/data-acquisition"
RAW = ROOT / "data/raw/roads/n13_2024_geojson"


def main():
    page = (ACQ / "n13_2024_source_page.html").read_text()
    advertised = set(re.findall(r"N13-24_\d+_GEOJSON\.zip", page))
    with (ACQ / "n13_2024_download_manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    names = {row["filename"] for row in manifest}
    assert len(manifest) == len(names) == 160
    assert names == advertised == {path.name for path in RAW.glob("*.zip")}
    for row in manifest:
        path = RAW / row["filename"]
        assert row["status"] == "validated" and not row["error"]
        assert row["url"] == "https://nlftp.mlit.go.jp/ksj/gml/data/N13/N13-24/" + path.name
        assert path.stat().st_size == int(row["bytes"])
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
    pilot = json.loads((ROOT / "data/exp/vehicle-network-pilot/source_hashes.json").read_text())
    roads = [row for row in pilot if row["path"].startswith("data/raw/roads/")]
    byname = {row["filename"]: row for row in manifest}
    assert {Path(row["path"]).name for row in roads} == {
        "N13-24_4830_GEOJSON.zip", "N13-24_4930_GEOJSON.zip"}
    assert all(row["sha256"] == byname[Path(row["path"]).name]["sha256"] for row in roads)
    review = json.loads((ROOT / "data/exp/network-map-review/summary.json").read_text())
    assert review["graph_modified"] is False and review["raw_data_modified"] is False
    result = dict(status="passed", official_source="https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N13-2024.html",
                  reference_date="2024-09", nationwide_archives=160,
                  inventory_matches_saved_official_page=True, current_file_hashes_match_acquisition=True,
                  pilot_road_archives=[Path(row["path"]).name for row in roads],
                  imagery_review_deleted_roads=False,
                  interpretation="N13-2024 is the road baseline. Absence in older or undated auxiliary imagery is not evidence that a source road was absent in 2024. Unresolved review status concerns insufficient evidence for graph connectivity, not road existence.")
    (ACQ / "n13_2024_source_confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
