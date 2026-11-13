"""Compile the Linux routing driver and run its existing correctness checks."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build")
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this command on Linux or WSL")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = (args.output / "sosr_route").resolve()
    flags = ["-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off"]
    sources = [ROOT / "driver.cpp"] + [ROOT / "source" / name for name in ["h2h.cpp", "spf.cpp", "EmbeddingService.cpp"]]
    subprocess.run([args.compiler, *flags, "-I", str(ROOT / "source"), *map(str, sources), "-o", str(binary)], check=True)
    for case in ["selftest", "timeout-test"]:
        subprocess.run([str(binary), case], check=True, timeout=120)
    geometry = (args.output / "check_geometry").resolve()
    subprocess.run([args.compiler, *flags, str(ROOT / "test_safe_geometry.cpp"), "-o", str(geometry)], check=True)
    subprocess.run([str(geometry)], check=True, timeout=60)
    print(binary)


if __name__ == "__main__":
    main()
