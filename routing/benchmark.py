"""Run the frozen, paired 1,800-query experiment with a 30-second query budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import time

from report_stats import METHODS, LABELS, write_report

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Worker:
    def __init__(self, binary, dataset, inputs, work, log_path, cpu):
        self.log = log_path.open("xb")
        def affinity():
            os.sched_setaffinity(0, {cpu})
        self.process = subprocess.Popen(
            [str(binary), "worker", dataset, str(inputs)], cwd=work,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            bufsize=0, start_new_session=True, preexec_fn=affinity,
        )
        self.buffer = b""
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def line(self, seconds):
        end = time.monotonic() + seconds
        while b"\n" not in self.buffer:
            remaining = end - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("Worker setup/protocol timeout")
            part = os.read(self.process.stdout.fileno(), 65536)
            if not part:
                raise RuntimeError("Routing worker terminated")
            self.log.write(part)
            self.buffer += part
        row, self.buffer = self.buffer.split(b"\n", 1)
        return row.decode("utf-8")

    def send(self, line):
        self.process.stdin.write((line + "\n").encode())

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("QUIT ignored")
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.selector.close()
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/roads")
    parser.add_argument("--binary", type=Path, default=ROOT / "routing/build/sosr_route")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["NYC", "BAY", "COL", "FLA"], default=["NYC", "BAY", "COL", "FLA"])
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--limit", type=int, help="Diagnostic subset per network; omit for formal results")
    args = parser.parse_args()
    if not hasattr(os, "sched_setaffinity"):
        parser.error("Linux or WSL is required")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory")
    if not args.binary.is_file():
        parser.error("Compile routing/build.py first")
    affinity = os.sched_getaffinity(0)
    cpu = min(affinity) if args.cpu is None else args.cpu
    if cpu not in affinity:
        parser.error("Requested CPU is outside this process's allowed affinity")
    manifest = json.loads((ROOT / "data/roads/manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir()
    for key, item in manifest.items():
        if key.split("/")[0] not in args.datasets:
            continue
        source = (args.data_root / key).resolve()
        if not source.is_file() or sha(source) != item["sha256"]:
            raise ValueError(f"Missing or mismatched road input: {key}")
        target = output / "data" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    all_queries = json.loads((ROOT / "routing/QUERIES.json").read_text())["queries"]
    selected, refs = [], {}
    for dataset in args.datasets:
        rows = [q for q in all_queries if q["dataset"] == dataset]
        selected.extend(rows[:args.limit] if args.limit else rows)
        path = ROOT / "routing/references" / (dataset + ".jsonl")
        refs.update({r["id"]: r for r in map(json.loads, path.read_text().splitlines())})
    saved = []
    for repeat in [1, 2, 3]:
        for dataset in args.datasets:
            rows = [q for q in selected if q["dataset"] == dataset]
            inputs = ROOT / "routing/inputs" / dataset / "queries.txt"
            folder = output / f"r{repeat}_{dataset}"
            folder.mkdir()
            worker = Worker(args.binary.resolve(), dataset, inputs.resolve(), work, folder / "worker.log", cpu)
            try:
                while not worker.line(3600).startswith("READY "):
                    pass
                with (folder / "measurements.jsonl").open("x", encoding="utf-8") as stream:
                    for index, query in enumerate(rows):
                        offset = (index + repeat - 1) % len(METHODS)
                        for method in METHODS[offset:] + METHODS[:offset]:
                            worker.send(f"{method} {query['id']}")
                            result = json.loads(worker.line(45))
                            if result.get("id") != query["id"] or result.get("algorithm") != method:
                                raise ValueError("Worker response does not match the request")
                            result.update(repeat=repeat, dataset=dataset, length=query["k"],
                                          frequency=query["frequency"], distance_bin=query["distance_bin"],
                                          candidate_counts=query["candidate_counts"], display_name=LABELS[method])
                            if result["status"] == "ok":
                                expected = refs[query["id"]]["distance"]
                                if abs(result["distance"] - expected) > 1e-6 + 1e-9 * abs(expected):
                                    raise ValueError(f"Answer mismatch: {query['id']} {method}")
                                points = result["waypoints"]
                                worker.send(f"VERIFY {query['id']} {result['distance']:.17g} {len(points)} " +
                                            " ".join(map(str, points)))
                                if worker.line(30) != f"VERIFY_OK {query['id']}":
                                    raise ValueError("Invalid reconstructed service vertices")
                                result["distance_and_waypoints_verified"] = True
                                if result["planner_ms"] > 30000:
                                    result.update(observed_after_budget_ms=result["planner_ms"],
                                                  planner_ms=None, status="timeout", censor_ms=30000)
                            elif result["status"] != "timeout" or result["planner_ms"] is not None:
                                raise ValueError("Query failed")
                            stream.write(json.dumps(result) + "\n")
                            stream.flush()
                            saved.append(result)
                        print(dataset, repeat, index + 1, "/", len(rows), flush=True)
            finally:
                worker.close()
    report = write_report(output, saved, selected)
    (output / "RUN_SCOPE.json").write_text(json.dumps({
        "formal_full_manifest": args.limit is None and set(args.datasets) == {"NYC", "BAY", "COL", "FLA"},
        "cpu": cpu, "unique_queries": len(selected), "measurements": len(saved),
    }, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
