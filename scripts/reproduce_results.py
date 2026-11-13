"""Recompute reported metrics from saved predictions and raw route measurements."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsm.metrics import score_archive, score_name_archive
sys.path.insert(0, str(ROOT / "routing"))
from report_stats import write_report

NAMES = {"bert": "xlm-r", "sentence_t": "sentence-t", "hsm": "hsm", "no_level_lambda1": "no-level",
         "random_neg_lambda1": "random-neg", "no_contrastive": "no-contrastive",
         "qwen_zero": "qwen-zero", "qwen_few": "qwen-few", "city_zero": "qwen-city-zero", "city_few": "qwen-city-few"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reproduced")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cats, test = ROOT / "data/semantic/categories.csv", ROOT / "data/semantic/test.json"
    metrics = {}
    for name in sorted(set(NAMES.values())):
        path = ROOT / "results/hsm/predictions" / (name + ".npz")
        metrics[name] = (score_archive(path, cats, test) if path.exists()
                         else score_name_archive(ROOT / "results/hsm/name_only" / (name + ".json"), cats, test))
    with (ROOT / "results/hsm/publication_metrics.csv").open(encoding="utf-8-sig") as stream:
        expected = list(csv.DictReader(stream))
    for row in expected:
        result = metrics[NAMES[row["source_key"]]]
        for key, field in [("D-Acc", "D-Acc"), ("Acc@1", "Acc@1"), ("MRR", "MRR_exact"), ("Level Acc", "Level Acc")]:
            if row[field] and result[key] is not None:
                if abs(result[key] - float(row[field])) > 1e-12:
                    raise ValueError(f"{row['model']} {key} differs from the reference")
    (args.output / "hsm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    raw = []
    for path in sorted((ROOT / "results/routing/raw").glob("*.jsonl")):
        raw.extend(json.loads(line) for line in path.read_text().splitlines())
    queries = json.loads((ROOT / "routing/QUERIES.json").read_text())["queries"]
    routes = args.output / "routing"
    routes.mkdir(exist_ok=True)
    report = write_report(routes, raw, queries)
    reference = json.loads((ROOT / "results/routing/RESULTS.json").read_text())
    if report["results"] != reference["results"]:
        raise ValueError("Recomputed routing statistics differ")
    print(json.dumps({"verified": True, "prediction_models": len(metrics),
                      "routing_queries": len(queries), "routing_measurements": len(raw),
                      "name_only_MRR": "archived reference only; full ID ranks are unavailable for XLM-R and Sentence-T",
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
