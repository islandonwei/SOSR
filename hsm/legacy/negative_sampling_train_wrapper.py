#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument(
        "--negative-mode",
        choices=("sampled", "full", "topk", "hybrid", "hierarchy_hard"),
        required=True,
    )
    parser.add_argument("--semantic-top-k", type=int, default=50)
    parser.add_argument("--hierarchy-scope", choices=("level", "strict"), default="level")
    parser.add_argument("--same-level-k", type=int, default=10)
    parser.add_argument("--parent-level-k", type=int, default=5)
    parser.add_argument("--child-level-k", type=int, default=5)
    parser.add_argument("--lambda-weight", type=float, required=True)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--predictor-lr", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("semantic_combined", "joint_mrr"),
        default="semantic_combined",
        help="Validation-only checkpoint selection criterion.",
    )
    parser.add_argument("--level-input", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--level-head", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--balanced-level-sampling", action="store_true")
    parser.add_argument("--ordinal-beta", type=float, default=0.0)
    parser.add_argument("--selection-level-accuracy", action="store_true")
    return parser.parse_args()


def make_contrastive_loss(
    mode,
    semantic_top_k,
    same_level_k=10,
    parent_level_k=5,
    child_level_k=5,
):
    def compute_loss(query_embeddings, category_embeddings, positive_indices, negative_indices, temperature):
        all_logits = torch.matmul(query_embeddings, category_embeddings.T) / temperature

        if mode == "full":
            return F.cross_entropy(all_logits, positive_indices)

        batch_size, num_categories = all_logits.shape
        hard_logits = all_logits.clone()
        hard_logits.scatter_(1, positive_indices.unsqueeze(1), float("-inf"))
        top_k = min(semantic_top_k, num_categories - 1)
        hard_indices = torch.topk(hard_logits, k=top_k, dim=1).indices

        if mode == "topk":
            positive_logits = all_logits.gather(1, positive_indices.unsqueeze(1))
            selected_negative_logits = all_logits.gather(1, hard_indices)
            logits = torch.cat((positive_logits, selected_negative_logits), dim=1)
            targets = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
            return F.cross_entropy(logits, targets)

        if mode == "hybrid":
            losses = []
            for row in range(batch_size):
                hierarchy_indices = negative_indices[row]
                hierarchy_indices = hierarchy_indices[hierarchy_indices >= 0]
                combined = torch.cat((hierarchy_indices, hard_indices[row]))
                combined = torch.unique(combined)
                combined = combined[combined != positive_indices[row]]
                positive_logit = all_logits[row, positive_indices[row]].unsqueeze(0)
                selected_negative_logits = all_logits[row, combined]
                logits = torch.cat((positive_logit, selected_negative_logits))
                losses.append(-F.log_softmax(logits, dim=0)[0])
            return torch.stack(losses).mean()

        if mode == "hierarchy_hard":
            losses = []
            quotas = (same_level_k, parent_level_k, child_level_k)
            for row in range(batch_size):
                packed = negative_indices[row]
                packed = packed[packed != -1]
                same_separator = (packed == -2).nonzero(as_tuple=False).flatten()
                child_separator = (packed == -3).nonzero(as_tuple=False).flatten()
                if len(same_separator) != 1 or len(child_separator) != 1:
                    raise RuntimeError("Malformed packed hierarchy candidates")
                first = int(same_separator.item())
                second = int(child_separator.item())
                candidate_groups = (
                    packed[:first],
                    packed[first + 1 : second],
                    packed[second + 1 :],
                )
                selected = []
                for candidates, quota in zip(candidate_groups, quotas):
                    candidates = candidates[candidates >= 0]
                    if quota <= 0 or candidates.numel() == 0:
                        continue
                    scores = all_logits[row, candidates]
                    count = min(quota, candidates.numel())
                    selected.append(candidates[torch.topk(scores, k=count).indices])
                if not selected:
                    continue
                selected_indices = torch.unique(torch.cat(selected))
                selected_indices = selected_indices[selected_indices != positive_indices[row]]
                if selected_indices.numel() == 0:
                    continue
                positive_logit = all_logits[row, positive_indices[row]].unsqueeze(0)
                selected_negative_logits = all_logits[row, selected_indices]
                logits = torch.cat((positive_logit, selected_negative_logits))
                losses.append(-F.log_softmax(logits, dim=0)[0])
            if not losses:
                return query_embeddings.sum() * 0.0
            return torch.stack(losses).mean()

        return train._original_compute_contrastive_loss(
            query_embeddings,
            category_embeddings,
            positive_indices,
            negative_indices,
            temperature,
        )

    return compute_loss


def install_hierarchy_candidate_sampler(scope):
    original_init = train.QueryDataset.__init__

    def unique_ids(category_ids, positive_id):
        return list(dict.fromkeys(category_id for category_id in category_ids if category_id != positive_id))

    def patched_init(self, data, tokenizer, category_tree, id_to_idx, is_train=True):
        original_init(self, data, tokenizer, category_tree, id_to_idx, is_train)
        self._hierarchy_hard_candidates = {}
        if not is_train:
            return
        for category_id, node in category_tree.nodes.items():
            if category_id == "ROOT" or category_id not in id_to_idx:
                continue
            same_ids = [item.category_id for item in category_tree.get_siblings(category_id)]
            parent_ids = [item.category_id for item in category_tree.get_parent_siblings(category_id)]
            child_ids = [item.category_id for item in category_tree.get_sibling_children(category_id)]
            if scope == "level":
                same_ids.extend(self.level_to_categories.get(node.level, []))
                parent_ids.extend(self.level_to_categories.get(node.level - 1, []))
                child_ids.extend(self.level_to_categories.get(node.level + 1, []))
            same_ids = unique_ids(same_ids, category_id)
            parent_ids = unique_ids(parent_ids, category_id)
            child_ids = unique_ids(child_ids, category_id)
            same_indices = [id_to_idx[item] for item in same_ids if item in id_to_idx]
            parent_indices = [id_to_idx[item] for item in parent_ids if item in id_to_idx]
            child_indices = [id_to_idx[item] for item in child_ids if item in id_to_idx]
            self._hierarchy_hard_candidates[category_id] = (
                same_indices + [-2] + parent_indices + [-3] + child_indices
            )

    def get_hierarchy_candidates(self, category_id, level):
        return self._hierarchy_hard_candidates[category_id]

    train.QueryDataset.__init__ = patched_init
    train.QueryDataset.get_negative_samples = get_hierarchy_candidates


@torch.no_grad()
def evaluate_joint(model, dataloader, category_embeddings, category_levels):
    """Paper score: cosine(query, category) * P(category level | query)."""
    model.eval()
    total_samples = 0
    total_mrr = 0.0
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(train.config.device)
        attention_mask = batch["attention_mask"].to(train.config.device)
        targets = batch["positive_idx"].to(train.config.device)
        query_embeddings, level_logits = model(input_ids, attention_mask)
        scores = (query_embeddings @ category_embeddings.T) * torch.softmax(level_logits, dim=1)[:, category_levels]
        ranks = (scores.argsort(dim=1, descending=True) == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        total_samples += ranks.numel()
        total_mrr += (1.0 / ranks.float()).sum().item()
        total_top1 += (ranks <= 1).sum().item()
        total_top3 += (ranks <= 3).sum().item()
        total_top5 += (ranks <= 5).sum().item()
    model.train()
    return {
        "joint_acc@1": total_top1 / total_samples,
        "joint_acc@3": total_top3 / total_samples,
        "joint_acc@5": total_top5 / total_samples,
        "joint_mrr": total_mrr / total_samples,
    }


def install_joint_mrr_checkpoint_selection():
    """Replace stage-2 selection only; all training hyperparameters stay unchanged."""
    def train_stage2(model, train_loader, val_loader, category_embeddings, category_names_list, tokenizer):
        print("\n" + "=" * 80)
        print("STAGE 2: Joint Fine-tuning (checkpoint selection: validation joint MRR)")
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
        best_joint_mrr = float("-inf")
        metrics_history = []
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
                contrastive_loss = train.compute_contrastive_loss(query_embeddings, category_embeddings, positive_idx, negative_indices, train.config.temperature)
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
            semantic_metrics = train.evaluate(model, val_loader, category_embeddings, "Val")
            joint_metrics = evaluate_joint(model, val_loader, category_embeddings, category_levels)
            epoch_metrics = {
                "epoch": epoch + 1,
                "stage": 2,
                "train_loss": total_loss / len(train_loader),
                "train_level_loss": total_level_loss / len(train_loader),
                "train_contrastive_loss": total_contrastive_loss / len(train_loader),
                **{f"val_{key}": value for key, value in semantic_metrics.items()},
                **{f"val_{key}": value for key, value in joint_metrics.items()},
            }
            metrics_history.append(epoch_metrics)
            print(f"\nEpoch {epoch + 1}/{train.config.stage2_epochs}")
            print(f"  Val Joint MRR: {joint_metrics['joint_mrr']:.4f}; Joint Acc@1: {joint_metrics['joint_acc@1']:.4f}")
            print(f"  Val Semantic Top-1: {semantic_metrics['top1_accuracy']:.4f}; Level Acc: {semantic_metrics['level_accuracy']:.4f}")
            if joint_metrics["joint_mrr"] > best_joint_mrr:
                best_joint_mrr = joint_metrics["joint_mrr"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "semantic_metrics": semantic_metrics,
                    "joint_metrics": joint_metrics,
                    "selection": "validation_joint_mrr",
                }, train.os.path.join(train.config.checkpoint_dir, "stage2_best.pt"))
                print(f"  → Saved best model (validation joint MRR: {best_joint_mrr:.4f})")
        return metrics_history

    train.train_stage2 = train_stage2


def install_level_model_variant(level_input, level_head, ordinal_beta):
    """Keep normalized embeddings for retrieval; optionally feed raw pooling to the level head."""
    original_init = train.SpecificityAwareModel.__init__

    def patched_init(self, model_path, num_levels, num_categories):
        original_init(self, model_path, num_levels, num_categories)
        if level_head == "mlp":
            self.level_predictor = torch.nn.Sequential(
                torch.nn.LayerNorm(self.embedding_dim),
                torch.nn.Linear(self.embedding_dim, 256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, num_levels),
            )

    def patched_forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        raw_embeddings = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        semantic_embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        level_embeddings = raw_embeddings if level_input == "raw" else semantic_embeddings
        return semantic_embeddings, self.level_predictor(level_embeddings)

    train.SpecificityAwareModel.__init__ = patched_init
    train.SpecificityAwareModel.forward = patched_forward

    if ordinal_beta > 0:
        original_loss = train.compute_level_loss

        def ordinal_loss(logits, targets):
            ce = original_loss(logits, targets)
            positions = torch.arange(logits.size(1), device=logits.device, dtype=logits.dtype)
            predicted_position = (torch.softmax(logits, dim=1) * positions).sum(dim=1)
            return ce + ordinal_beta * F.mse_loss(predicted_position, targets.to(logits.dtype))

        train.compute_level_loss = ordinal_loss


def install_balanced_level_sampling():
    original_loader = train.DataLoader

    def patched_loader(dataset, *args, **kwargs):
        if kwargs.get("shuffle") and getattr(dataset, "is_train", False):
            levels = [item[2] for item in dataset.data]
            counts = pd.Series(levels).value_counts().to_dict()
            weights = torch.DoubleTensor([1.0 / counts[level] for level in levels])
            kwargs.pop("shuffle", None)
            kwargs["sampler"] = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return original_loader(dataset, *args, **kwargs)

    train.DataLoader = patched_loader


def install_level_accuracy_selection():
    """The existing stage-2 score is level accuracy plus Top-1; zero Top-1 for level-only selection."""
    original_evaluate = train.evaluate

    def level_only_evaluate(*args, **kwargs):
        metrics = original_evaluate(*args, **kwargs)
        metrics["top1_accuracy"] = 0.0
        metrics["top5_accuracy"] = 0.0
        metrics["level_aware_accuracy"] = 0.0
        return metrics

    train.evaluate = level_only_evaluate


def main():
    args = parse_args()
    split_dir = Path(args.split_dir)
    config = train.config
    config.categories_path = "./categories.csv"
    config.model_path = "./paraphrase-multilingual-mpnet-base-v2-local"
    config.output_dir = "./trained_model"
    config.checkpoint_dir = "./checkpoints"
    config.log_dir = "./logs"
    config.lambda_weight = args.lambda_weight
    config.stage1_lr = args.stage1_lr
    config.stage2_lr_encoder = args.encoder_lr
    config.stage2_lr_predictor = args.predictor_lr
    config.temperature = args.temperature
    config.negative_mode = args.negative_mode
    config.semantic_top_k = args.semantic_top_k
    config.hierarchy_scope = args.hierarchy_scope
    config.same_level_k = args.same_level_k
    config.parent_level_k = args.parent_level_k
    config.child_level_k = args.child_level_k
    config.checkpoint_selection = args.checkpoint_selection
    config.level_input = args.level_input
    config.level_head = args.level_head
    config.balanced_level_sampling = args.balanced_level_sampling
    config.ordinal_beta = args.ordinal_beta
    for path in (config.output_dir, config.checkpoint_dir, config.log_dir):
        os.makedirs(path, exist_ok=True)

    def load_validation_only():
        categories_df = pd.read_csv(config.categories_path)
        category_tree = train.CategoryTree(categories_df)
        all_category_ids = categories_df["category_id"].tolist()
        id_to_idx = {category_id: idx for idx, category_id in enumerate(all_category_ids)}
        id_to_name = {row["category_id"]: row["category_name"] for _, row in categories_df.iterrows()}

        def read_split(name):
            records = json.loads((split_dir / f"{name}.json").read_text(encoding="utf-8"))
            return [(item["query"], item["original_id"], int(item["level"])) for item in records]

        train_data = read_split("train")
        val_data = read_split("val")
        print(
            f"Validation-only negative-sampling experiment: train={len(train_data)}, val={len(val_data)}; "
            f"mode={args.negative_mode}, semantic_top_k={args.semantic_top_k}; real test split is not read"
        )
        return (
            train_data,
            val_data,
            val_data,
            category_tree,
            id_to_idx,
            id_to_name,
            all_category_ids,
            categories_df,
        )

    train.load_data = load_validation_only
    if args.negative_mode == "hierarchy_hard":
        install_hierarchy_candidate_sampler(args.hierarchy_scope)
    install_level_model_variant(args.level_input, args.level_head, args.ordinal_beta)
    if args.balanced_level_sampling:
        install_balanced_level_sampling()
    if args.selection_level_accuracy:
        install_level_accuracy_selection()
    if args.checkpoint_selection == "joint_mrr":
        install_joint_mrr_checkpoint_selection()
    train._original_compute_contrastive_loss = train.compute_contrastive_loss
    train.compute_contrastive_loss = make_contrastive_loss(
        args.negative_mode,
        args.semantic_top_k,
        args.same_level_k,
        args.parent_level_k,
        args.child_level_k,
    )
    train.set_seed(42)
    train.main()


if __name__ == "__main__":
    main()
