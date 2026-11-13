"""Metrics on fixed category IDs; a descendant is credited only for D-Acc."""
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_taxonomy(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [r["category_id"] for r in rows]
    paths = [tuple(x.strip() for x in r["category_label"].split(">")) for r in rows]
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise ValueError("Category IDs and full paths must be unique")
    available = set(paths)
    for row, path in zip(rows, paths):
        if len(path) != int(row["category_level"]) or any(path[:k] not in available for k in range(1, len(path))):
            raise ValueError("Incomplete or inconsistent taxonomy")
    return rows, ids, paths


def score_rankings(order, targets, paths, levels=None, true_levels=None):
    order = np.asarray(order)
    targets = np.asarray(targets)
    if order.shape != (len(targets), len(paths)):
        raise ValueError("Expected one complete category ranking per sample")
    if not np.all(np.sort(order, axis=1) == np.arange(len(paths))):
        raise ValueError("Each ranking must be a permutation of category indices")
    ranks = (order == targets[:, None]).argmax(axis=1) + 1
    top = order[:, 0]
    compatible = [paths[p][:len(paths[t])] == paths[t] for p, t in zip(top, targets)]
    result = {"n": len(targets), "D-Acc": float(np.mean(compatible)),
              "Acc@1": float(np.mean(top == targets)), "MRR": float(np.mean(1.0 / ranks)),
              "Level Acc": None, "exact_hits": int(np.sum(top == targets)),
              "compatible_hits": int(np.sum(compatible))}
    if levels is not None and len(levels):
        if true_levels is None or len(levels) != len(targets):
            raise ValueError("Explicit level predictions must align with samples")
        result["Level Acc"] = float(np.mean(np.asarray(levels) == np.asarray(true_levels)))
    return result


def score_archive(path, categories_path, samples_path):
    _, ids, paths = load_taxonomy(categories_path)
    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    lookup = {cid: i for i, cid in enumerate(ids)}
    targets = np.array([lookup[r["original_id"]] for r in samples])
    with np.load(path, allow_pickle=False) as data:
        if data["category_ids"].tolist() != ids or not np.array_equal(data["target_indices"], targets):
            raise ValueError("Predictions use a different category order or test set")
        truth = np.array([r["level"] for r in samples])
        if not np.array_equal(data["true_levels"], truth):
            raise ValueError("Level labels differ from the frozen split")
        return score_rankings(data["order"], targets, paths, data["predicted_levels"], truth)


def score_name_archive(path, categories_path, samples_path):
    """Recover hit counts without inventing IDs for old name-only predictions."""
    cats, ids, paths = load_taxonomy(categories_path)
    by_id = dict(zip(ids, paths))
    names = defaultdict(list)
    for row in cats:
        names[row["category_name"]].append(row["category_id"])
    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(saved["predictions"]) != len(samples):
        raise ValueError("Prediction count differs")
    legacy_targets = {r["category_name"]: r["category_id"] for r in cats}
    options = []
    base = [0, 0, 0]
    ambiguous = []
    for i, (p, t) in enumerate(zip(saved["predictions"], samples)):
        if p["query"] != t["query"] or p["target"] != t["categories"][0]:
            raise ValueError("Name-only records do not align with the test split")
        opts = names[p["top5"][0]]
        options.append(opts)
        if len(opts) > 1:
            ambiguous.append(i)
            continue
        pred = opts[0]
        gold = t["original_id"]
        base[0] += pred == legacy_targets[t["categories"][0]]
        base[1] += pred == gold
        base[2] += by_id[pred][:len(by_id[gold])] == by_id[gold]
    if len(ambiguous) > 12:
        raise ValueError("Too many ambiguous predictions for exhaustive ID reconstruction")
    legacy_hits = round(saved["metrics"]["acc@1"] * len(samples))
    feasible = set()
    for assignment in itertools.product(*(options[i] for i in ambiguous)):
        counts = list(base)
        for i, pred in zip(ambiguous, assignment):
            t = samples[i]
            gold = t["original_id"]
            counts[0] += pred == legacy_targets[t["categories"][0]]
            counts[1] += pred == gold
            counts[2] += by_id[pred][:len(by_id[gold])] == by_id[gold]
        if counts[0] == legacy_hits:
            feasible.add((counts[1], counts[2]))
    if not feasible:
        raise ValueError("Saved aggregate hits contradict the available names")
    exact = sorted({a for a, _ in feasible})
    compatible = sorted({b for _, b in feasible})
    return {"n": len(samples), "Acc@1": exact[0] / len(samples) if len(exact) == 1 else None,
            "D-Acc": compatible[0] / len(samples) if len(compatible) == 1 else None,
            "exact_hits": exact[0] if len(exact) == 1 else None,
            "compatible_hits": compatible[0] if len(compatible) == 1 else None,
            "MRR": None, "archived_name_mapping_MRR": saved["metrics"]["mrr"], "Level Acc": None,
            "MRR_status": "Complete ID rankings were not retained for this historical baseline.",
            "possible_exact_hits": exact, "possible_compatible_hits": compatible}
