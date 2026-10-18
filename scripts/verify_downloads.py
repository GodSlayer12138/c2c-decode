"""Check LFS weight SHA-256 against content-addressed Hugging Face blob names."""
import hashlib
import json
from pathlib import Path
import re

lock = json.loads(Path("models.lock.json").read_text())
rows = []
for role, entry in lock.items():
    weights = sorted(Path(entry["path"]).glob("*.safetensors")) + sorted(Path(entry["path"]).glob("*.pt"))
    if not weights:
        raise RuntimeError(f"No weights for {role}")
    for file in weights:
        expected = file.resolve().name
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"Not a content-addressed LFS blob: {file}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8*2**20), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {file}")
        rows.append({"role": role, "file": file.name, "bytes": file.stat().st_size, "sha256": actual})
        print("Verified", role, file.name, flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/download-integrity.json").write_text(json.dumps({"passed": True, "weights": rows}, indent=2)+"\n")
