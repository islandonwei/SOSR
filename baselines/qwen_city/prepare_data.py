import collections
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer

from audit_inputs import canonical, digest, sha

ROOT = Path(__file__).resolve().parent
TOK = None
CUTOFF = 4096


def messages_for(row):
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    for user, assistant in row.get("history", []):
        messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    user = row["instruction"]
    if row.get("input"):
        user += "\n" + row["input"]
    messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}])
    return messages


def init_worker(model, cutoff):
    global TOK, CUTOFF
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOK = AutoTokenizer.from_pretrained(model, local_files_only=True)
    CUTOFF = cutoff


def encode_messages(messages, tokenizer, cutoff):
    def render(ms, add=False):
        return tokenizer.apply_chat_template(ms, tokenize=False, add_generation_prompt=add)
    text = render(messages)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    labels = [-100] * len(ids)
    spans = []
    for k, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        header = render(messages[:k], True)
        end = render(messages[:k + 1])
        if not text.startswith(header) or not text.startswith(end):
            raise ValueError("Chat-template text prefix is not stable")
        spans.append((len(header), len(end)))
    j = 0
    for t, (a, b) in enumerate(encoded["offset_mapping"]):
        while j < len(spans) and a >= spans[j][1]:
            j += 1
        if j < len(spans) and spans[j][0] <= a < b <= spans[j][1]:
            labels[t] = ids[t]
    if len(ids) != len(labels) or not spans:
        raise ValueError("Invalid token spans")
    return ids[:cutoff], labels[:cutoff], len(ids), spans


def encode_row(item):
    i, row = item
    group = digest({k: row.get(k, [] if k == "history" else "") for k in ("system", "history", "instruction", "input")})
    split_hash = hashlib.sha256(("42|" + group).encode()).hexdigest()
    split = "validation" if int(split_hash, 16) % 1000 < 100 else "train"
    ids, labels, original_len, spans = encode_messages(messages_for(row), TOK, CUTOFF)
    supervised = sum(v != -100 for v in labels[1:])
    return i, group, split, ids, labels, original_len, supervised, spans


def content_key(r):
    return digest({k: r.get(k, [] if k == "history" else "") for k in ("system", "history", "instruction", "input", "output")})


def composition_audit(london_rows):
    return {"note": "Optional cross-city provenance audit omitted; training rows unchanged."}


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    plan = json.loads((ROOT / "PLAN.json").read_text(encoding="utf-8"))
    out = ROOT / plan["tokenization"]["prepared_directory"]
    out.mkdir(exist_ok=False)
    source = Path(plan["data"]["path"])
    if sha(source) != plan["data"]["sha256"]:
        raise ValueError("Training input hash mismatch")
    rows = json.loads(source.read_text(encoding="utf-8"))
    if len(rows) != plan["data"]["records_before_processing"]:
        raise ValueError("Unexpected source record count")
    (out / "COMPOSITION_AUDIT.json").write_text(
        json.dumps(composition_audit(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tokenizer = AutoTokenizer.from_pretrained(plan["base_model"], local_files_only=True)
    toy = [{"role": "user", "content": "User text."}, {"role": "assistant", "content": "Assistant one."},
           {"role": "user", "content": "Second user."}, {"role": "assistant", "content": "Assistant two."}]
    ids, labels, _, spans = encode_messages(toy, tokenizer, 4096)
    assert len(spans) == 2
    text = tokenizer.apply_chat_template(toy, tokenize=False, add_generation_prompt=False)
    off = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    assert all(y == (ids[i] if any(s <= a < b <= e for s, e in spans) else -100)
               for i, ((a, b), y) in enumerate(zip(off, labels)))
    assert ids == list(tokenizer.apply_chat_template(toy, tokenize=True, return_dict=False))
    white = [{"role": "user", "content": "Whitespace test."},
             {"role": "assistant", "content": "\n\n  Answer with leading spaces.\n"}]
    wi, wl, _, ws = encode_messages(white, tokenizer, 4096)
    assert any(x != -100 for x in wl)
    checks = {"multi_turn_mask_check": True, "single_encoding_matches_chat_template": True,
              "leading_whitespace_check": True, "character_spans": spans, "rendered": text}
    (out / "TOKENIZER_CHECK.json").write_text(json.dumps(checks, indent=2) + "\n")
    offsets, lengths, targets, raw_indices = [], [], [], []
    train, valid, valid_keys = [], [], []
    train_groups, valid_groups = set(), set()
    counters = collections.Counter()
    cursor, started = 0, time.time()
    context = mp.get_context("spawn")
    with open(out / "tokens.u32", "xb") as tf, open(out / "labels.i32", "xb") as lf, \
         open(out / "records.jsonl", "x", encoding="utf-8") as audit, \
         context.Pool(plan["tokenization"]["workers"], initializer=init_worker,
                      initargs=(plan["base_model"], plan["tokenization"]["cutoff_len"])) as pool:
        for i, group, split, ids, labels, original_len, n_targets, spans in pool.imap(encode_row, enumerate(rows), chunksize=32):
            info = {"source_index": i, "group_hash": group, "split": split, "original_tokens": original_len,
                    "retained_tokens": len(ids), "supervised_tokens": n_targets, "assistant_turns": len(spans)}
            counters["source_records"] += 1
            counters["truncated_records"] += int(original_len > len(ids))
            if n_targets == 0:
                info["excluded_reason"] = "zero_surviving_supervised_targets_after_fixed_right_truncation"
                counters["excluded_zero_targets"] += 1
            else:
                ix = len(offsets)
                info["prepared_index"] = ix
                offsets.append(cursor)
                lengths.append(len(ids))
                targets.append(n_targets)
                raw_indices.append(i)
                tf.write(np.asarray(ids, dtype="<u4").tobytes())
                lf.write(np.asarray(labels, dtype="<i4").tobytes())
                cursor += len(ids)
                if split == "train":
                    train.append(ix)
                    train_groups.add(group)
                else:
                    valid.append(ix)
                    valid_keys.append((group, i, ix))
                    valid_groups.add(group)
                counters[split + "_records"] += 1
                counters[split + "_tokens"] += len(ids)
                counters[split + "_supervised_tokens"] += n_targets
            audit.write(json.dumps(info) + "\n")
            if (i + 1) % 10000 == 0:
                print(json.dumps({"processed": i + 1, "seconds": time.time() - started, **dict(counters)}), flush=True)
    if train_groups & valid_groups:
        raise ValueError("Prompt-group split leakage")
    for name, values, dtype in [
        ("offsets", offsets, "<u8"), ("lengths", lengths, "<u4"), ("targets", targets, "<u4"),
        ("source_indices", raw_indices, "<u4"), ("train_indices", train, "<u4"),
        ("validation_indices", valid, "<u4"),
        ("probe_indices", [t[2] for t in sorted(valid_keys)[:512]], "<u4"),
    ]:
        np.save(out / (name + ".npy"), np.asarray(values, dtype=dtype), allow_pickle=False)
    manifest = {
        "plan_sha256": sha(ROOT / "PLAN.json"),
        "preprocessor_sha256": sha(Path(__file__)),
        "source_sha256": sha(source),
        "counters": dict(counters),
        "prepared_records": len(offsets),
        "flat_token_count": cursor,
        "group_split_overlap": 0,
        "sequence_length_quantiles": dict(zip(["0", "25", "50", "75", "90", "95", "99", "100"],
                                            np.percentile(lengths, [0,25,50,75,90,95,99,100]).tolist())),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(out.iterdir()) if p.is_file()},
        "seconds": time.time() - started,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared_complete": True, **manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
