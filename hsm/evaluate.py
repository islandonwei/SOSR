"""Evaluate one selected checkpoint using complete category-ID rankings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsm.metrics import load_taxonomy, score_rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/semantic")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--variant", choices=["hsm", "no-level", "random-neg", "no-contrastive", "embedding"], default="hsm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, help="Query batch size; validation defaults to 64, test to 128")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.variant != "embedding" and args.checkpoint is None:
        parser.error("A selected checkpoint is required")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    cats, ids, paths = load_taxonomy(args.data / "categories.csv")
    samples = json.loads((args.data / (args.split + ".json")).read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[:args.limit]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(args.device)
    head = torch.nn.Linear(encoder.config.hidden_size, 6).to(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        weights = checkpoint["model_state_dict"]
        encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in weights.items() if k.startswith("encoder.")}, strict=True)
        head.load_state_dict({k.removeprefix("level_predictor."): v for k, v in weights.items() if k.startswith("level_predictor.")}, strict=True)
    encoder.eval()
    head.eval()
    def encode(texts, batch_size, padding):
        values = []
        for i in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[i:i+batch_size], padding=padding, truncation=True, max_length=64, return_tensors="pt").to(args.device)
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            values.append(F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9), dim=1))
        return torch.cat(values)
    with torch.inference_mode():
        category_vectors = encode([c["category_name"] for c in cats], 128, True)
        query_batch = args.batch_size or (64 if args.split == "val" else 128)
        query_padding = "max_length" if args.split == "val" else True
        query_vectors = encode([r["query"] for r in samples], query_batch, query_padding)
        scores = query_vectors @ category_vectors.T
        predicted_levels = np.array([], dtype=np.int8)
        if args.variant not in ["no-level", "embedding"]:
            logp = F.log_softmax(head(query_vectors), dim=1)
            levels = torch.tensor([int(c["category_level"]) - 1 for c in cats], device=args.device)
            scores = scores / 0.01 + logp[:, levels]
            predicted_levels = logp.argmax(1).cpu().numpy() + 1
        order = scores.argsort(dim=1, descending=True).cpu().numpy()
    lookup = {cid: i for i, cid in enumerate(ids)}
    target = np.array([lookup[r["original_id"]] for r in samples])
    truth = np.array([r["level"] for r in samples])
    metrics = score_rankings(order, target, paths, predicted_levels, truth)
    np.savez_compressed(output / "predictions.npz", order=order.astype(np.uint16), target_indices=target,
                        category_ids=np.array(ids), true_levels=truth, predicted_levels=predicted_levels)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
