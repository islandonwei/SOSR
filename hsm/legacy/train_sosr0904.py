#!/usr/bin/env python3
"""Validation-only trainer for the SOSR0904 protocol.

This module deliberately reuses the verified v4 training implementation and
overrides only the changes documented in README_REPRODUCTION.md:
  * hierarchy-hard online negatives (10/5/5);
  * CE(label_smoothing=0.1) + 0.5 * CDF-EMD for the level loss;
  * contrastive temperature 0.05;
  * validation checkpoint selection using the documented log joint score.

The real test split is never opened by this program.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig

import negative_sampling_train_wrapper as base
import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--semantic-temperature", type=float, default=0.01)
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    return parser.parse_args()


def install_pooling(pooling):
    """Use the encoder's documented pooling without changing any other protocol item."""
    if pooling == "mean":
        return

    def encode_text(self, input_ids, attention_mask):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(hidden_states[:, 0], p=2, dim=1)

    train.SpecificityAwareModel.encode_text = encode_text


def documented_level_loss(logits, targets):
    """CE with label smoothing plus the six-bin CDF Earth Mover loss."""
    ce = F.cross_entropy(logits, targets, label_smoothing=0.1)
    probabilities = F.softmax(logits, dim=1)
    target_distribution = F.one_hot(targets, num_classes=logits.size(1)).to(logits.dtype)
    emd = (probabilities.cumsum(dim=1) - target_distribution.cumsum(dim=1)).square().mean(dim=1).mean()
    return ce + 0.5 * emd


@torch.no_grad()
def evaluate_log_joint(model, dataloader, category_embeddings, category_levels, semantic_temperature):
    """Rank by sim/tau_sem + log P(predicted category level | query)."""
    model.eval()
    all_ranks = []
    level_correct = 0
    sample_count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(train.config.device)
        attention_mask = batch["attention_mask"].to(train.config.device)
        targets = batch["positive_idx"].to(train.config.device)
        true_levels = batch["level"].to(train.config.device)
        query_embeddings, level_logits = model(input_ids, attention_mask)
        level_log_probabilities = F.log_softmax(level_logits, dim=1)
        scores = query_embeddings @ category_embeddings.T / semantic_temperature
        scores = scores + level_log_probabilities[:, category_levels]
        rankings = scores.argsort(dim=1, descending=True)
        all_ranks.append((rankings == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1)
        level_correct += (level_logits.argmax(dim=1) == true_levels).sum().item()
        sample_count += targets.numel()
    ranks = torch.cat(all_ranks).float()
    model.train()
    return {
        "acc@1": (ranks <= 1).float().mean().item(),
        "acc@3": (ranks <= 3).float().mean().item(),
        "acc@5": (ranks <= 5).float().mean().item(),
        "mrr": (1.0 / ranks).mean().item(),
        "level_acc": level_correct / sample_count,
    }


def install_documented_checkpoint_selection(semantic_temperature):
    def train_stage2(model, train_loader, val_loader, category_embeddings, category_names_list, tokenizer):
        print("\n" + "=" * 80)
        print("STAGE 2: joint fine-tuning (selection: validation log-joint MRR)")
        print("=" * 80)
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": train.config.stage2_lr_encoder},
            {"params": model.level_predictor.parameters(), "lr": train.config.stage2_lr_predictor},
        ], weight_decay=train.config.weight_decay)
        total_steps = len(train_loader) * train.config.stage2_epochs
        warmup_steps = int(total_steps * train.config.warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + train.np.cos(train.np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        category_levels = torch.tensor(
            [train_loader.dataset.category_tree.nodes[category_id].level - 1 for category_id in train_loader.dataset.id_to_idx],
            device=train.config.device,
        )
        best_mrr = float("-inf")
        history = []
        for epoch in range(train.config.stage2_epochs):
            model.train()
            total_loss = total_level_loss = total_contrastive_loss = 0.0
            print("Computing category embeddings...")
            category_embeddings = train.compute_category_embeddings(model, category_names_list, tokenizer)
            for batch in train.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train.config.stage2_epochs}"):
                input_ids = batch["input_ids"].to(train.config.device)
                attention_mask = batch["attention_mask"].to(train.config.device)
                positive_idx = batch["positive_idx"].to(train.config.device)
                negative_indices = batch["negative_indices"].to(train.config.device)
                level_targets = batch["level"].to(train.config.device)
                query_embeddings, level_logits = model(input_ids, attention_mask)
                level_loss = train.compute_level_loss(level_logits, level_targets)
                contrastive_loss = train.compute_contrastive_loss(
                    query_embeddings, category_embeddings, positive_idx, negative_indices, train.config.temperature
                )
                loss = level_loss + train.config.lambda_weight * contrastive_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train.config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_level_loss += level_loss.item()
                total_contrastive_loss += contrastive_loss.item()

            category_embeddings = train.compute_category_embeddings(model, category_names_list, tokenizer)
            metrics = evaluate_log_joint(model, val_loader, category_embeddings, category_levels, semantic_temperature)
            record = {
                "epoch": epoch + 1,
                "stage": 2,
                "train_loss": total_loss / len(train_loader),
                "train_level_loss": total_level_loss / len(train_loader),
                "train_contrastive_loss": total_contrastive_loss / len(train_loader),
                **{f"val_{name}": value for name, value in metrics.items()},
            }
            history.append(record)
            print(f"\nEpoch {epoch + 1}/{train.config.stage2_epochs}")
            print("  Validation log-joint: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": metrics,
                    "selection": "validation_log_joint_mrr",
                    "semantic_temperature": semantic_temperature,
                }, os.path.join(train.config.checkpoint_dir, "stage2_best.pt"))
                print(f"  -> saved best model (validation log-joint MRR: {best_mrr:.4f})")
        return history

    train.train_stage2 = train_stage2


def main():
    args = parse_args()
    split_dir = Path(args.split_dir)
    config = train.config
    config.categories_path = "./categories.csv"
    config.model_path = "./paraphrase-multilingual-mpnet-base-v2-local"
    config.output_dir = "./trained_model"
    config.checkpoint_dir = "./checkpoints"
    config.log_dir = "./logs"
    config.lambda_weight = 1.5
    config.stage1_lr = 1e-3
    config.stage2_lr_encoder = 1e-5
    config.stage2_lr_predictor = 2e-3
    config.temperature = 0.05
    config.negative_mode = "hierarchy_hard"
    config.hierarchy_scope = "level"
    config.same_level_k = 10
    config.parent_level_k = 5
    config.child_level_k = 5
    config.embedding_dim = AutoConfig.from_pretrained(config.model_path).hidden_size
    for path in (config.output_dir, config.checkpoint_dir, config.log_dir):
        os.makedirs(path, exist_ok=True)

    def load_validation_only():
        categories_df = pd.read_csv(config.categories_path)
        category_tree = train.CategoryTree(categories_df)
        all_category_ids = categories_df["category_id"].tolist()
        id_to_idx = {category_id: index for index, category_id in enumerate(all_category_ids)}
        id_to_name = {row["category_id"]: row["category_name"] for _, row in categories_df.iterrows()}

        def read_split(name):
            records = json.loads((split_dir / f"{name}.json").read_text(encoding="utf-8"))
            return [(item["query"], item["original_id"], int(item["level"])) for item in records]

        train_data = read_split("train")
        val_data = read_split("val")
        print(
            f"SOSR0904 validation-only experiment: train={len(train_data)}, val={len(val_data)}; "
            "real test split is not read"
        )
        return train_data, val_data, val_data, category_tree, id_to_idx, id_to_name, all_category_ids, categories_df

    train.load_data = load_validation_only
    base.install_hierarchy_candidate_sampler("level")
    install_pooling(args.pooling)
    train._original_compute_contrastive_loss = train.compute_contrastive_loss
    train.compute_contrastive_loss = base.make_contrastive_loss("hierarchy_hard", 50, 10, 5, 5)
    train.compute_level_loss = documented_level_loss
    install_documented_checkpoint_selection(args.semantic_temperature)
    train.set_seed(42)
    train.main()


if __name__ == "__main__":
    main()
