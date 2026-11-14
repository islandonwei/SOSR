"""Download and verify the saved predictions and routing records."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest():
    return json.loads((ROOT / "configs/results.json").read_text(encoding="utf-8"))


def record_path(directory, relative):
    destination = (directory / relative).resolve()
    if not destination.is_relative_to(directory.resolve()):
        raise ValueError(f"Invalid record path: {relative}")
    return destination


def verify_records(directory, manifest):
    for relative, expected in manifest["files"].items():
        path = record_path(directory, relative)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing reference record: {path}. Run python scripts/download_results.py first."
            )
        if sha256(path) != expected:
            raise ValueError(f"Reference record checksum differs: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reference")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded ZIP instead of the network")
    args = parser.parse_args()
    manifest = load_manifest()
    output = args.output.resolve()
    try:
        missing = []
        for relative, expected in manifest["files"].items():
            path = record_path(output, relative)
            if path.exists():
                if not path.is_file() or sha256(path) != expected:
                    raise ValueError(f"Refusing to overwrite a different local file: {path}")
            else:
                missing.append(relative)
        if missing:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".sosr-results-", dir=output.parent) as temp:
                staging = Path(temp)
                archive = args.archive
                if archive is None:
                    archive = staging / "results.zip"
                    print(f"Downloading {manifest['url']}", flush=True)
                    with urlopen(manifest["url"], timeout=60) as response, archive.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                if archive.stat().st_size != manifest["size_bytes"] or sha256(archive) != manifest["sha256"]:
                    raise ValueError("Archive checksum or size differs from configs/results.json")
                extracted = staging / "records"
                with zipfile.ZipFile(archive) as bundle:
                    names = bundle.namelist()
                    if len(names) != len(manifest["files"]) or set(names) != set(manifest["files"]):
                        raise ValueError("Archive contents differ from configs/results.json")
                    for relative in names:
                        destination = record_path(extracted, relative)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(relative) as source, destination.open("wb") as stream:
                            shutil.copyfileobj(source, stream)
                verify_records(extracted, manifest)
                for relative in missing:
                    destination = record_path(output, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(extracted / relative, destination)
        verify_records(output, manifest)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"{error}\n")
    print(f"Verified {len(manifest['files'])} reference files in {output}")


if __name__ == "__main__":
    main()
