"""Acquire the human-approved official population forecast without changing raw data."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-acquisition/demographic-projection"
URL = "https://nlftp.mlit.go.jp/ksj/gml/data/m1kr6/m1kr6-24/1km_mesh_2024_SHP.zip"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "1km_mesh_2024_SHP.zip"
    if not destination.exists():
        partial = destination.with_suffix(".zip.part")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as target:
            received = 0
            next_report = 32 * 1024 * 1024
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                received += len(chunk)
                if received >= next_report:
                    print(f"Downloaded {received / 1024**2:.0f} MiB", flush=True)
                    next_report += 32 * 1024 * 1024
        with zipfile.ZipFile(partial) as archive:
            assert archive.testzip() is None
        partial.replace(destination)
    with zipfile.ZipFile(destination) as archive:
        members = [{"name": x.filename, "bytes": x.file_size} for x in archive.infolist()]
    with destination.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"status": "acquired", "authorization": "MILI-D-20261007-035", "source_url": URL,
              "official_catalog": "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-mesh1000r6.html",
              "archive": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
              "sha256": sha, "members": members, "raw_data_modified": False}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
