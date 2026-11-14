"""Frozen Zero/Few full-ranking protocol for Qwen and Qwen-City."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsm.metrics import load_taxonomy, score_rankings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["zero", "few"], required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data/semantic")
    p.add_argument("--candidate-batch", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cache_score import cached_scores, fork_cache
    from prompts import base_messages, user_text
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    _, ids, paths = load_taxonomy(args.data / "categories.csv")
    lookup = {c: i for i, c in enumerate(ids)}
    taxonomy = json.loads((ROOT / "baselines/prompts/taxonomy.json").read_text())
    examples = json.loads((ROOT / "baselines/prompts/few_shot_examples.json").read_text()) if args.mode == "few" else []
    mapping = np.array([lookup[r["id"]] for r in taxonomy])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    samples = json.loads((args.data / (args.split + ".json")).read_text())
    samples = samples[:args.limit] if args.limit else samples
    order, level_predictions = [], []
    with torch.inference_mode():
        contexts = {}
        completions = {
            "category": [tokenizer.encode(r["alias"], add_special_tokens=False) + [tokenizer.eos_token_id] for r in taxonomy],
            "level": [tokenizer.encode(str(i), add_special_tokens=False) + [tokenizer.eos_token_id] for i in range(1, 7)],
        }
        for task in ["category", "level"]:
            messages = base_messages(taxonomy, examples, task)
            prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
            result = model(input_ids=torch.tensor([prefix], device=args.device), use_cache=True, logits_to_keep=1)
            contexts[task] = (messages, prefix_text, prefix, result.past_key_values)
        for index, row in enumerate(samples):
            scores = {}
            for task in ["category", "level"]:
                messages, prefix_text, prefix, cache = contexts[task]
                text = tokenizer.apply_chat_template(messages + [{"role": "user", "content": user_text(row["query"], task)}],
                                                     tokenize=False, add_generation_prompt=True)
                if not text.startswith(prefix_text):
                    raise ValueError("Chat-template prefix changed")
                tail = tokenizer.encode(text[len(prefix_text):], add_special_tokens=False)
                if len(prefix) + len(tail) + max(map(len, completions[task])) > model.config.max_position_embeddings:
                    raise ValueError("Prompt exceeds the model context")
                result = model(input_ids=torch.tensor([tail], device=args.device), past_key_values=fork_cache(cache),
                               use_cache=True, logits_to_keep=1)
                scores[task] = cached_scores(model, result.past_key_values, result.logits[:, -1],
                                             completions[task], args.candidate_batch)
            sorted_indices = np.argsort(-np.array(scores["category"]), kind="stable")
            order.append(mapping[sorted_indices])
            level_predictions.append(1 + int(np.argmax(scores["level"])))
            print(index + 1, "/", len(samples), flush=True)
    targets = np.array([lookup[r["original_id"]] for r in samples])
    truth = np.array([r["level"] for r in samples])
    metrics = score_rankings(np.array(order), targets, paths, level_predictions, truth)
    np.savez_compressed(args.output / "predictions.npz", order=np.array(order, dtype=np.uint16),
                        target_indices=targets, category_ids=np.array(ids), true_levels=truth,
                        predicted_levels=np.array(level_predictions))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
