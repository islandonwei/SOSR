"""Create the local paths for the recorded Qwen-City SFT recipe."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--london-json", type=Path, required=True)
    args = p.parse_args()
    target = ROOT / "PLAN.json"
    if target.exists():
        p.error("PLAN.json already exists; preserve any prepared/trained run")
    config = json.loads((ROOT / "PLAN.template.json").read_text())
    with args.london_json.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != config["data"]["sha256"]:
        p.error("London input does not match the recorded corpus")
    for name, expected in config["base_weight_sha256"].items():
        with (args.model / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                p.error(f"Base model shard differs: {name}")
    config["base_model"] = str(args.model.resolve())
    config["data"]["path"] = str(args.london_json.resolve())
    target.write_text(json.dumps(config, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
