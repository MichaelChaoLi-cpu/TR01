"""Acquire the 47 official 2021 traffic tables and source documentation.

Downloads remain under data/exp; no observation filtering or graph assignment.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import time
import urllib.request
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/national-traffic-acquisition"
BASE = "https://www.mlit.go.jp/road/census/r3/"


def fetch(filename, url):
    path = OUT / "sources" / filename
    if not path.exists():
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "TR01 research source acquisition"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                temp = path.with_suffix(path.suffix+".part")
                temp.write_bytes(data)
                temp.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1+attempt)
    data = path.read_bytes()
    return {"file": filename, "url": url, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main():
    (OUT / "sources").mkdir(parents=True, exist_ok=True)
    manifest = [fetch("index.html", BASE)]
    html = (OUT / "sources/index.html").read_text(encoding="utf-8")
    links = re.findall(r'href=["\']([^"\']+)["\']', html, re.I)
    tables = {int(re.search(r"kasyo(\d{2})\.csv", link).group(1)): urljoin(BASE, link)
              for link in links if re.search(r"kasyo\d{2}\.csv", link)}
    assert set(tables) == set(range(1, 48)), "Official index must list all 47 prefecture tables"
    with ThreadPoolExecutor(max_workers=2) as pool:
        for entry in pool.map(lambda code: fetch(f"kasyo{code:02d}.csv", tables[code]), sorted(tables)):
            manifest.append(entry)
            print(f"Acquired {entry['file']}: {entry['bytes']:,} bytes", flush=True)
    manifest.append(fetch("kasyorep.pdf", urljoin(BASE, "data/pdf/kasyorep.pdf")))
    manifest.append(fetch("mlit_reuse_terms.html", "https://www.mlit.go.jp/link.html"))
    # Preserve the official geometry viewer metadata; this is not bulk geometry acquisition.
    manifest.append(fetch("webmap.html", "https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/webmap.html"))
    manifest.append(fetch("property.js", "https://www.mlit.go.jp/road/ir/ir-data/census_visualizationR3/property.js"))
    coverage = []
    for code in sorted(tables):
        path = OUT / "sources" / f"kasyo{code:02d}.csv"
        data = list(csv.reader(path.read_text(encoding="cp932").splitlines()))
        header, rows = data[0], data[1:]
        assert len(header) == 159 and all(len(row) == 159 for row in rows)
        assert len({row[0] for row in rows}) == len(rows)
        coverage.append({"Prefecture Code": code, "Source Rows": len(rows), "Source Columns": len(header),
                         "Unique Header Labels": len(set(header)),
                         "Rows With Both Direction Observation Flags 1": sum(row[34] == row[46] == "1" for row in rows),
                         "Distinct Traffic Observation Units": len({(row[30], row[31]) for row in rows if row[31].strip()})})
    with (OUT / "prefecture_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0])); writer.writeheader(); writer.writerows(coverage)
    for entry in manifest:
        assert hashlib.sha256((OUT / "sources" / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    receipt = {"status": "passed", "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
               "prefectures": 47, "traffic_table_rows": sum(row["Source Rows"] for row in coverage),
               "source_files": len(manifest), "source_hashes_verified": True,
               "geometry_acquired_nationally": False, "graph_assignments": 0,
               "observation_filter_adopted": False, "processed_variables_created": False,
               "raw_folder_modified": False}
    (OUT / "validation.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
