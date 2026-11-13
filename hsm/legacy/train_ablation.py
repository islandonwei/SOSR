#!/usr/bin/env python3
"""Validation-only Table-IV ablations for the SOSR0904 HSM protocol.

Variants match the original paper's Table IV: no level loss/scoring,
random-negative contrastive learning, and three values of lambda.  The real
test split is never opened here; checkpoints are selected only on validation.
"""
import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig

import negative_sampling_train_wrapper as base
import train
from train_sosr0904 import documented_level_loss, evaluate_log_joint, install_pooling


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", required=True)
    p.add_argument("--variant", required=True, choices=("no-level", "random-neg", "lambda-0.5", "lambda-1.0", "lambda-3.0", "lambda-5.0"))
    p.add_argument("--semantic-temperature", type=float, default=.01)
    p.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    return p.parse_args()


@torch.no_grad()
def evaluate_semantic(model, dataloader, category_embeddings):
    model.eval(); ranks_all = []
    for batch in dataloader:
        ids = batch["input_ids"].to(train.config.device); mask = batch["attention_mask"].to(train.config.device)
        targets = batch["positive_idx"].to(train.config.device)
        vectors, _ = model(ids, mask)
        rankings = (vectors @ category_embeddings.T).argsort(dim=1, descending=True)
        ranks_all.append((rankings == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1)
    ranks = torch.cat(ranks_all).float(); model.train()
    return {"acc@1": (ranks <= 1).float().mean().item(), "acc@3": (ranks <= 3).float().mean().item(),
            "acc@5": (ranks <= 5).float().mean().item(), "mrr": (1. / ranks).mean().item()}


def install_random_sampler(count=20):
    """Replace hierarchy-aware pools with deterministic random category pools."""
    original_init = train.QueryDataset.__init__
    def patched_init(self, data, tokenizer, category_tree, id_to_idx, is_train=True):
        original_init(self, data, tokenizer, category_tree, id_to_idx, is_train)
        self._random_negatives = {}
        if is_train:
            all_indices = list(id_to_idx.values())
            for category_id, positive in id_to_idx.items():
                candidates = [x for x in all_indices if x != positive]
                # category-specific seeded sampling keeps fixed pools independent of loader order
                rng = random.Random(f"SOSR0904-random-neg-{category_id}")
                self._random_negatives[category_id] = rng.sample(candidates, min(count, len(candidates)))
    def get_negatives(self, category_id, level):
        return self._random_negatives[category_id]
    train.QueryDataset.__init__ = patched_init
    train.QueryDataset.get_negative_samples = get_negatives


def install_stage2_selection(semantic_temperature, no_level):
    def stage2(model, train_loader, val_loader, category_embeddings, category_names, tokenizer):
        tag = "semantic MRR" if no_level else "log-joint MRR"
        print(f"\nSTAGE 2 ABLATION: validation {tag} selection")
        params = [{"params": model.encoder.parameters(), "lr": train.config.stage2_lr_encoder}]
        if not no_level:
            params.append({"params": model.level_predictor.parameters(), "lr": train.config.stage2_lr_predictor})
        optimizer = torch.optim.AdamW(params, weight_decay=train.config.weight_decay)
        total = len(train_loader) * train.config.stage2_epochs; warmup = int(total * train.config.warmup_ratio)
        def lr_lambda(step):
            if step < warmup: return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return max(0., .5 * (1. + train.np.cos(train.np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        category_levels = torch.tensor([train_loader.dataset.category_tree.nodes[c].level - 1 for c in train_loader.dataset.id_to_idx], device=train.config.device)
        best, history = float("-inf"), []
        for epoch in range(train.config.stage2_epochs):
            model.train(); tot = lev_tot = con_tot = 0.
            category_embeddings = train.compute_category_embeddings(model, category_names, tokenizer)
            for batch in train.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train.config.stage2_epochs}"):
                ids = batch["input_ids"].to(train.config.device); mask = batch["attention_mask"].to(train.config.device)
                pos = batch["positive_idx"].to(train.config.device); neg = batch["negative_indices"].to(train.config.device)
                levels = batch["level"].to(train.config.device)
                vectors, logits = model(ids, mask)
                contrastive = train.compute_contrastive_loss(vectors, category_embeddings, pos, neg, train.config.temperature)
                level_loss = vectors.sum() * 0. if no_level else train.compute_level_loss(logits, levels)
                loss = level_loss + train.config.lambda_weight * contrastive
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), train.config.max_grad_norm)
                optimizer.step(); scheduler.step(); tot += loss.item(); lev_tot += level_loss.item(); con_tot += contrastive.item()
            category_embeddings = train.compute_category_embeddings(model, category_names, tokenizer)
            metrics = evaluate_semantic(model, val_loader, category_embeddings) if no_level else evaluate_log_joint(model, val_loader, category_embeddings, category_levels, semantic_temperature)
            record = {"epoch": epoch + 1, "stage": 2, "train_loss": tot / len(train_loader), "train_level_loss": lev_tot / len(train_loader), "train_contrastive_loss": con_tot / len(train_loader), **{f"val_{k}": v for k, v in metrics.items()}}
            history.append(record); print(json.dumps(record), flush=True)
            if metrics["mrr"] > best:
                best = metrics["mrr"]
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "validation_metrics": metrics,
                            "selection": "validation_semantic_mrr" if no_level else "validation_log_joint_mrr",
                            "semantic_temperature": semantic_temperature}, os.path.join(train.config.checkpoint_dir, "stage2_best.pt"))
        return history
    train.train_stage2 = stage2


def main():
    args = arguments(); no_level = args.variant == "no-level"
    lambdas = {"lambda-0.5": .5, "lambda-1.0": 1., "lambda-3.0": 3., "lambda-5.0": 5.}
    c = train.config; split_dir = Path(args.split_dir)
    c.categories_path = str(split_dir / "categories.csv"); c.model_path = os.environ["SOSR_ENCODER"]
    c.output_dir = "./trained_model"; c.checkpoint_dir = "./checkpoints"; c.log_dir = "./logs"
    c.lambda_weight = lambdas.get(args.variant, 1.0); c.stage1_lr = 1e-3; c.stage2_lr_encoder = args.encoder_lr; c.stage2_lr_predictor = 2e-3; c.temperature = .05
    c.negative_mode = "random" if args.variant == "random-neg" else "hierarchy_hard"; c.hierarchy_scope = "level"; c.same_level_k = 10; c.parent_level_k = 5; c.child_level_k = 5
    c.embedding_dim = AutoConfig.from_pretrained(c.model_path).hidden_size
    for path in (c.output_dir, c.checkpoint_dir, c.log_dir): os.makedirs(path, exist_ok=True)
    def load_validation_only():
        cats = pd.read_csv(c.categories_path); tree = train.CategoryTree(cats); ids = cats.category_id.tolist(); index = {x:i for i,x in enumerate(ids)}; names = dict(zip(cats.category_id, cats.category_name))
        def read(name):
            return [(x["query"], x["original_id"], int(x["level"])) for x in json.loads((split_dir / f"{name}.json").read_text())]
        training, validation = read("train"), read("val")
        print(f"SOSR0904 {args.variant}: train={len(training)}, val={len(validation)}; test not read")
        return training, validation, validation, tree, index, names, ids, cats
    train.load_data = load_validation_only
    if args.variant == "random-neg": install_random_sampler()
    else: base.install_hierarchy_candidate_sampler("level")
    install_pooling(args.pooling); train._original_compute_contrastive_loss = train.compute_contrastive_loss
    # No-Level removes only the level-prediction loss and level-aware ranking;
    # it deliberately retains the paper HSM hierarchy-hard negatives.  Random-
    # Neg instead supplies a flat, deterministic pool, so sampled InfoNCE must
    # consume it directly rather than interpreting it as packed level groups.
    contrastive_mode = "sampled" if args.variant == "random-neg" else "hierarchy_hard"
    train.compute_contrastive_loss = base.make_contrastive_loss(contrastive_mode, 50, 10, 5, 5)
    if not no_level: train.compute_level_loss = documented_level_loss
    else: train.train_stage1 = lambda *unused: []
    install_stage2_selection(args.semantic_temperature, no_level)
    train.set_seed(args.seed); train.main()
    with open(os.path.join(c.log_dir, "training_metrics.json")) as f:
        records = json.load(f)
    stage2 = [r for r in records if r.get("stage") == 2]
    best = max(stage2, key=lambda r: r["val_mrr"])
    summary = {
        "split": "validation_only", "test_evaluated": False,
        "variant": args.variant, "seed": args.seed,
        "encoder_lr": args.encoder_lr, "lambda_weight": c.lambda_weight,
        "selection": "validation_mrr", "selected_epoch": best["epoch"],
        "metrics": {k: v for k, v in best.items() if k.startswith("val_")},
        "checkpoint": os.path.abspath(os.path.join(c.checkpoint_dir, "stage2_best.pt")),
        "note": "Legacy train.main final test-labelled output uses the validation alias, not the held-out test split."
    }
    with open("validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("VALIDATION_RUN_COMPLETE " + json.dumps(summary), flush=True)

if __name__ == "__main__": main()
