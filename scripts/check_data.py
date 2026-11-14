"""Verify frozen semantic files and, optionally, extracted road inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roads", type=Path)
    args = p.parse_args()
    semantic = ROOT / "data/semantic"
    count = 0
    for name, expected in json.loads((semantic / "checksums.json").read_text()).items():
        with (semantic / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Semantic file differs: {name}")
        count += 1
    if args.roads:
        for name, entry in json.loads((ROOT / "data/roads/manifest.json").read_text()).items():
            with (args.roads / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != entry["sha256"]:
                raise ValueError(f"Road file differs: {name}")
            count += 1
    print(json.dumps({"verified_files": count}))


if __name__ == "__main__":
    main()
