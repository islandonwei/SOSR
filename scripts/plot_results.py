"""Draw semantic metric panels from the versioned publication table."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "runs/figures")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results/hsm/publication_metrics.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for table, stem in [("II", "model_comparison"), ("IV", "ablation")]:
        group = [r for r in rows if r["table"] == table]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
        for ax, metric in zip(axes, ["D-Acc", "Acc@1", "MRR_display", "Level Acc"]):
            vals = [float(r[metric]) if r[metric] else np.nan for r in group]
            labels = [r["model"].replace("BERT", "XLM-R") for r in group]
            ax.bar(np.arange(len(group)), vals, color=["#376a92" if r["source_key"] == "hsm" else "#c5cbd1" for r in group])
            ax.set_xticks(np.arange(len(group)), labels, rotation=65, ha="right", fontsize=8)
            ax.set_title(metric.replace("_display", ""))
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
        for ext in ["pdf", "png"]:
            fig.savefig(args.output / (stem + "." + ext), dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
