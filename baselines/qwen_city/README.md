# Qwen-City full SFT

This is the independent Qwen2.5-7B-Instruct adaptation on CityInstruction-London used by this project. It is distinct from an official CityGPT checkpoint.

Obtain the London Alpaca JSON from the resources linked by the [CityGPT authors](https://github.com/tsinghua-fib-lab/CityGPT). The filename, corpus hash and base-shard hashes are in PLAN.template.json.

Use the LLM environment and run from the repository root.

    python baselines/qwen_city/configure.py --model /absolute/path/to/Qwen2.5-7B-Instruct --london-json /absolute/path/to/citygpt-London-v24.6-SF-v2.3-alpaca.json
    python baselines/qwen_city/prepare_data.py
    torchrun --standalone --nproc_per_node=4 baselines/qwen_city/train_fsdp.py --phase train --output baselines/qwen_city/train_run

Preparation masks assistant tokens using character offsets from one rendered conversation. Identical prompts are grouped before the deterministic domain train/validation split. Corpus duplicate rows are retained. The optional cross-city source audit is omitted from the portable preprocessor because it does not change training samples.

The trainer is the final successful FSDP version. It uses one epoch, global batch 64, learning rate 1e-5, a cosine schedule, BF16 computation/FP32 reduction and context 4096. Selection uses the predeclared final-epoch rule. SOSR splits are not training inputs. Intermediate model-only checkpoints do not contain optimizer state for exact resumption.

The recorded run trained 124,944 records in 1,953 updates using four large-memory GPUs. The raw London corpus and 7B model weights are external assets.
