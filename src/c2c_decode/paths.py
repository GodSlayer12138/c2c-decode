import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def enable_upstreams():
    for name in ("C2C", "nano-vllm"):
        path = ROOT / "upstream" / name
        if not path.is_dir():
            raise RuntimeError(f"Missing locked upstream checkout: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def model_paths(lock_path="models.lock.json"):
    lock = json.loads(Path(lock_path).read_text())
    expected = {"receiver": "Qwen/Qwen3-0.6B", "sender": "Qwen/Qwen3-4B", "fuser": "nics-efc/C2C_Fuser"}
    for role, repo in expected.items():
        if lock[role]["repo"] != repo:
            raise ValueError(f"Wrong {role}: expected {repo}")
        if not Path(lock[role]["path"]).is_dir():
            raise FileNotFoundError(lock[role]["path"])
    if Path(lock["fuser"]["path"]).parts[-2:] != ("qwen3_0.6b+qwen3_4b_Fuser", "final"):
        raise ValueError("Fuser must match the exact non-Base Qwen3-4B pair")
    return {k: v["path"] for k, v in lock.items()}
