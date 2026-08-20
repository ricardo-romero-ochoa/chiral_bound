#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest of source, data, and reproducibility artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MANIFEST.sha256"
EXCLUDE = {"MANIFEST.sha256", ".DS_Store"}
EXCLUDE_SUFFIX = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


files = []
for path in ROOT.rglob("*"):
    if not path.is_file() or path.name in EXCLUDE or path.suffix in EXCLUDE_SUFFIX:
        continue
    if path.name.endswith("Notes.bib"):
        continue
    if any(part in {".git", ".pytest_cache", "__pycache__", ".venv"} for part in path.parts):
        continue
    files.append(path)

with OUT.open("w") as handle:
    for path in sorted(files):
        handle.write(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n")
print(f"wrote {OUT} with {len(files)} entries")
