"""Create a source-only ZIP, excluding local environments, bytecode, and user datasets."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent.parent / "outputs" / "offline-vision-x.zip"
SKIP = {".venv", "__pycache__", ".git"}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.relative_to(ROOT).parts):
            continue
        if path.name.endswith((".pyc", ".pyo")):
            continue
        archive.write(path, path.relative_to(ROOT))
print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
