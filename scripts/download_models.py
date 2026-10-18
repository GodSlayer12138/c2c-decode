"""Download only the selected pair; record immutable revisions and local paths."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="models.lock.json")
    args = p.parse_args()
    lock = json.loads(Path("upstream.lock.json").read_text())
    output = Path(args.output)
    resolved = json.loads(output.read_text()) if output.exists() else {}
    api = HfApi()
    for role in ("receiver", "sender", "fuser"):
        repo = lock[role]
        entry = resolved.get(role, {})
        revision = entry.get("revision") or api.model_info(repo).sha
        patterns = ([lock["fuser_subdir"] + "/*"] if role == "fuser" else
                    ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        resolved[role] = {"repo": repo, "revision": revision}
        output.write_text(json.dumps(resolved, indent=2) + "\n")
        print(f"Downloading {repo}@{revision}", flush=True)
        path = snapshot_download(repo, revision=revision, allow_patterns=patterns, max_workers=4)
        if role == "fuser":
            path = str(Path(path) / lock["fuser_subdir"])
        resolved[role]["path"] = path
        output.write_text(json.dumps(resolved, indent=2) + "\n")
    print("Model lock saved:", output, flush=True)


if __name__ == "__main__":
    main()
