"""Independent, finite, full-parameter SFT. Never opens an SOSR split."""
import argparse
import contextlib
from datetime import datetime, timedelta
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, MixedPrecision,
    ShardingStrategy, StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from audit_inputs import sha

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class PreparedData:
    def __init__(self, plan):
        self.root = ROOT / plan["tokenization"]["prepared_directory"]
        self.manifest = json.loads((self.root / "MANIFEST.json").read_text())
        if self.manifest["plan_sha256"] != sha(ROOT / "PLAN.json"):
            raise ValueError("Plan changed since data preparation")
        self.tokens = np.memmap(self.root / "tokens.u32", mode="r", dtype="<u4")
        self.labels = np.memmap(self.root / "labels.i32", mode="r", dtype="<i4")
        for key in ("offsets", "lengths", "targets", "train_indices", "probe_indices"):
            setattr(self, key, np.load(self.root / (key + ".npy"), mmap_mode="r", allow_pickle=False))

    def example(self, idx, device):
        i = int(idx)
        a, n = int(self.offsets[i]), int(self.lengths[i])
        return {
            "input_ids": torch.tensor(np.array(self.tokens[a:a+n], dtype=np.int64), device=device)[None],
            "labels": torch.tensor(np.array(self.labels[a:a+n], dtype=np.int64), device=device)[None],
            "attention_mask": torch.ones((1, n), dtype=torch.long, device=device),
        }


def setup():
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    expected = os.environ.get("EXPECTED_GPU_UUIDS", "").split(",")
    actual = str(torch.cuda.get_device_properties(local).uuid)
    if expected == [""] or actual.removeprefix("GPU-").lower() != expected[local].removeprefix("GPU-").lower():
        raise RuntimeError(f"CUDA/NVML device binding mismatch: actual={actual}, expected={expected}")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    return rank, dist.get_world_size(), torch.device("cuda", local)


def make_model(plan, device):
    base = AutoModelForCausalLM.from_pretrained(
        plan["base_model"], local_files_only=True, dtype=torch.float32,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = FSDP(
        base,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.bfloat16),
        device_id=device, sync_module_states=True, use_orig_params=True, limit_all_gathers=True,
    )
    return base, model


def run_update(model, optimizer, data, indices, device, rank, world, lr):
    n = len(indices)
    total_targets = sum(int(data.targets[int(i)]) for i in indices)
    if total_targets <= 0:
        raise ValueError("No targets in global batch")
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    model.train()
    stats = torch.zeros(2, device=device, dtype=torch.float64)
    micro_count = math.ceil(n / world)
    for micro in range(micro_count):
        position = micro * world + rank
        real = position < n
        idx = indices[position] if real else indices[0]
        batch = data.example(idx, device)
        n_local = int(data.targets[int(idx)]) if real else 0
        sync = contextlib.nullcontext() if micro == micro_count - 1 else model.no_sync()
        with sync, torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch, use_cache=False)
            mean_loss = output.loss
            scaled = mean_loss * (n_local * world / total_targets)
            scaled.backward()
        stats[0] += mean_loss.detach().double() * n_local
        stats[1] += n_local
        del output, mean_loss, scaled, batch
    norm = model.clip_grad_norm_(1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError(f"Nonfinite gradient norm: {norm}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    dist.all_reduce(stats)
    if int(stats[1].item()) != total_targets or not torch.isfinite(stats[0]):
        raise FloatingPointError("Target-count or loss audit failed")
    return stats[0].item() / total_targets, float(norm.item()), total_targets


@torch.no_grad()
def evaluate(model, data, indices, device, rank, world, return_rows=False):
    model.eval()
    accum = torch.zeros(2, dtype=torch.float64, device=device)
    local_rows = []
    for offset in range(0, len(indices), world):
        pos = offset + rank
        real = pos < len(indices)
        idx = int(indices[pos] if real else indices[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**data.example(idx, device), use_cache=False)
        loss = float(out.loss.item())
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        if real:
            targets = int(data.targets[idx])
            accum[0] += loss * targets
            accum[1] += targets
            if return_rows:
                local_rows.append({"prepared_index": idx, "loss": loss, "targets": targets})
        del out
    dist.all_reduce(accum)
    result = {"records": len(indices), "supervised_tokens": int(accum[1].item()),
              "token_mean_loss": float((accum[0] / accum[1]).item())}
    if return_rows:
        gathered = [None] * world
        dist.all_gather_object(gathered, local_rows)
        result["rows"] = sorted([r for group in gathered for r in group], key=lambda r: r["prepared_index"])
    return result


def save_checkpoint(base, model, output, step, rank, plan, data):
    folder = output / f"checkpoint_step_{step:06d}"
    if folder.exists():
        raise FileExistsError(folder)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()
    if rank == 0:
        folder.mkdir()
        state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}
        base.config.use_cache = True
        base.save_pretrained(folder, state_dict=state, safe_serialization=True, max_shard_size="4GB")
        base.config.use_cache = False
        AutoTokenizer.from_pretrained(plan["base_model"], local_files_only=True).save_pretrained(folder)
        write_json(folder / "EXPERIMENT_METADATA.json", {
            "claim": plan["claim"], "step": step, "created": now(),
            "plan_sha256": sha(ROOT / "PLAN.json"), "trainer_sha256": sha(Path(__file__)),
            "prepared_manifest_sha256": sha(data.root / "MANIFEST.json"),
            "checkpoint_tensor_dtype": "bfloat16", "optimizer_state_saved": False,
            "not_an_exact_resume_checkpoint": True,
        })
        hashes = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(folder.iterdir()) if p.is_file()}
        write_json(folder / "FILES.json", hashes)
    del state
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return str(folder)


def train(args):
    rank, world, device = setup()
    plan = json.loads((ROOT / "PLAN.json").read_text())
    if world != plan["training"]["world_size"]:
        raise ValueError("Wrong world size")
    output = Path(args.output).resolve()
    if output.parent != ROOT:
        raise ValueError("Output must be a new child of the isolated experiment root")
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    data = PreparedData(plan)
    if rank == 0:
        write_json(output / "STATUS.json", {"state": "loading_base", "started": now(), "phase": args.phase})
    base, model = make_model(plan, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan["training"]["learning_rate"],
                                  betas=tuple(plan["training"]["betas"]), eps=1e-8,
                                  weight_decay=0.0, fused=True)
    if rank == 0:
        print(json.dumps({"event": "model_loaded", "phase": args.phase, "time": now(),
                          "trainable_parameter_shards_rank0": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    train_indices = np.array(data.train_indices, dtype=np.int64)
    if args.phase == "real":
        # Every rank sees a 4096-token example before an optimizer update, followed by
        # ordinary-length samples. Preflight updates are discarded, never reused.
        long_ids = sorted(train_indices, key=lambda i: (-int(data.lengths[i]), int(i)))[:4]
        ordinary = [int(i) for i in train_indices if int(data.lengths[i]) <= 1024][:120]
        order = np.asarray(ordinary[:60] + long_ids + ordinary[60:] + long_ids, dtype=np.int64)
        if len(order) != 128 or min(int(data.lengths[i]) for i in long_ids) != 4096:
            raise ValueError("Preflight could not cover full context length")
        probe = np.asarray(ordinary[:4], dtype=np.int64)
    else:
        order = np.random.default_rng(plan["seed"]).permutation(train_indices)
        probe = np.array(data.probe_indices, dtype=np.int64)
    before = evaluate(model, data, probe, device, rank, world)
    if rank == 0:
        write_json(output / "BASE_DOMAIN_LOSS.json", before)
    total_steps = math.ceil(len(order) / 64)
    warmup = math.ceil(total_steps * plan["training"]["warmup_ratio"])
    save_at = ({total_steps} if args.phase == "real"
               else {math.ceil(total_steps * f) for f in plan["training"]["checkpoint_fractions"]})
    checkpoints, processed, started = [], 0, time.time()
    for step in range(1, total_steps + 1):
        indices = order[(step-1)*64:step*64].tolist()
        if args.phase == "real":
            lr = plan["training"]["learning_rate"]
        elif step <= warmup:
            lr = plan["training"]["learning_rate"] * step / max(warmup, 1)
        else:
            lr = plan["training"]["learning_rate"] * 0.5 * (
                1 + math.cos(math.pi * (step-warmup) / max(1, total_steps-warmup)))
        tick = time.time()
        loss, norm, n_targets = run_update(model, optimizer, data, indices, device, rank, world, lr)
        torch.cuda.synchronize()
        elapsed = time.time() - tick
        processed += len(indices)
        if rank == 0:
            record = {"time": now(), "phase": args.phase, "step": step, "total_steps": total_steps,
                      "processed_records": processed, "planned_records": len(order),
                      "loss": loss, "grad_norm": norm, "lr": lr, "supervised_tokens": n_targets,
                      "step_seconds": elapsed, "elapsed_seconds": time.time()-started,
                      "peak_allocated_gib_rank0": torch.cuda.max_memory_allocated()/1024**3}
            with open(output / "steps.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            write_json(output / "STATUS.json", {"state": "training", **record})
            if step <= 3 or step % 10 == 0:
                print(json.dumps(record), flush=True)
        if step in save_at:
            if rank == 0:
                write_json(output / "STATUS.json", {"state": "saving_checkpoint", "step": step, "time": now()})
            checkpoint = save_checkpoint(base, model, output, step, rank, plan, data)
            checkpoints.append(checkpoint)
    after = evaluate(model, data, probe, device, rank, world, return_rows=args.phase == "real")
    if processed != len(order):
        raise ValueError("Training coverage mismatch")
    if rank == 0:
        write_json(output / "FINAL_DOMAIN_LOSS.json", after)
        write_json(output / "RESULTS.json", {
            "state": "complete_stopped", "phase": args.phase, "completed": now(),
            "steps": total_steps, "records": processed, "plan_sha256": sha(ROOT / "PLAN.json"),
            "selected_checkpoint": checkpoints[-1], "checkpoint_selection": "fixed_final",
            "all_checkpoints": checkpoints, "base_domain_loss": before, "final_domain_loss": after,
            "sosr_validation_evaluated": False, "sosr_test_evaluated": False,
            "preflight_updates_discarded": args.phase == "real",
            "training_seconds": time.time()-started,
        })
        write_json(output / "STATUS.json", {"state": "complete_stopped", "phase": args.phase,
                                            "completed": now(), "steps": total_steps, "records": processed})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["real", "train"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        train(args)
    except BaseException as exc:
        if int(os.environ.get("RANK", "0")) == 0:
            output = Path(args.output)
            if output.is_dir():
                write_json(output / "FAILURE.json", {"time": now(), "exception": repr(exc), "phase": args.phase})
        raise
