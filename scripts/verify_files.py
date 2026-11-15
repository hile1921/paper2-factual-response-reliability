"""Verify the distributed file manifest; uses Python 3 standard library only."""
from pathlib import Path
import csv, hashlib
ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "FILE_MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    path = ROOT / row["path"]
    assert path.is_file(), row["path"]
    content = path.read_bytes()
    # Git may normalise text line endings; hashes use LF for text and raw bytes for binaries.
    if row["hash_mode"] == "text_lf":
        content = content.replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == row["sha256"], row["path"]
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            assert header, path.name
            for index, record in enumerate(reader, 2):
                assert len(record) == len(header), (path.name, index)
print(f"Verified {len(rows)} manifest entries and CSV structure. This does not validate statistical claims.")
