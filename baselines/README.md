# Semantic baselines

This directory provides the category-prediction baselines used alongside HSM. Commands below run from the repository root.

Routing baselines are documented in [routing/README.md](../routing/README.md).

## Embedding baselines

Activate the HSM environment and install its dependencies:

```bash
pip install -r requirements-hsm.txt
```

Evaluate XLM-R mean pooling:

```bash
python -m hsm.evaluate --variant embedding --model FacebookAI/xlm-roberta-base --split test --output runs/xlmr_test
```

Evaluate Sentence-T:

```bash
python -m hsm.evaluate --variant embedding --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --split test --output runs/sentence_t_test
```

Both methods encode category names, rank the complete category list, and write `metrics.json` and `predictions.npz`. Level Acc is inapplicable to these embedding-only variants. The historical manuscript label BERT corresponds to the XLM-R implementation.

## Qwen Zero/Few

Use a separate environment for the LLM dependencies:

```bash
python -m venv .venv-llm
source .venv-llm/bin/activate
pip install -r requirements-llm.txt
```

Run validation with the fixed zero-shot or few-shot prompt:

```bash
python baselines/evaluate_llm.py --model Qwen/Qwen2.5-7B-Instruct --mode zero --split val --output runs/qwen_zero_val
python baselines/evaluate_llm.py --model Qwen/Qwen2.5-7B-Instruct --mode few --split val --output runs/qwen_few_val
```

After fixing the protocol, evaluate the test split:

```bash
python baselines/evaluate_llm.py --model Qwen/Qwen2.5-7B-Instruct --mode zero --split test --output runs/qwen_zero_test
python baselines/evaluate_llm.py --model Qwen/Qwen2.5-7B-Instruct --mode few --split test --output runs/qwen_few_test
```

The default device is CUDA. The complete category-ranking procedure can be expensive. For a diagnostic subset, pass `--limit 2` and use a separate output directory.

The evaluator uses category aliases, complete hierarchy paths, and a separate level-prediction task. The frozen taxonomy and training examples are in `baselines/prompts/`. Outputs have the same metric and prediction filenames as the embedding evaluator.

## Qwen-City

Qwen-City is the project's Qwen2.5-7B-Instruct model fine-tuned on CityInstruction-London. It is an independent adaptation, with the recipe and data preparation described in [qwen_city/README.md](qwen_city/README.md).

After training the final checkpoint, use its directory as `--model`:

```bash
python baselines/evaluate_llm.py --model baselines/qwen_city/train_run/checkpoint_step_001953 --mode zero --split test --output runs/qwen_city_zero_test
```

The recorded run ended at step 1,953. Use the actual final checkpoint directory produced by your matching run. The same evaluator supports `--mode few`.

Saved results, metric definitions, and historical record limitations are documented in [results/README.md](../results/README.md).
