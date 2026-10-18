"""Fetch the exact upstream commits; never track moving main branches."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "upstream.lock.json").read_text())
for key, name in (("c2c", "C2C"), ("nano_vllm", "nano-vllm")):
    spec, path = lock[key], root / "upstream" / name
    if not path.exists():
        subprocess.run(["git", "clone", spec["url"], str(path)], check=True)
    if not (path / ".git").exists():
        raise RuntimeError(f"{path} is an exported tree. Verify its source archive before use.")
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", spec["commit"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    assert actual == spec["commit"]
    print(name, actual)
