# HSM

HSM maps a natural-language requirement to one category in the POI hierarchy. It combines semantic similarity with an explicit prediction of the requirement's category level.

Run commands from the repository root. Use a new output directory for each run.

## Setup

The default encoder is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. The `--model` argument accepts this Hugging Face identifier or a local base-encoder directory.

```bash
python -m venv .venv-hsm
source .venv-hsm/bin/activate
pip install -r requirements-hsm.txt
python scripts/check_data.py
```

The default device is CUDA. Use a compatible PyTorch build for the machine and pass `--device cpu` for CPU execution.

Data are read from `data/semantic/` by default. See the [data guide](../data/README.md) for formats and split information.

## Evaluate a checkpoint

Pass the selected checkpoint through `--checkpoint`. The example below uses the output of the [training command](#training).

```bash
python -m hsm.evaluate --checkpoint runs/hsm_seed42/checkpoints/stage2_best.pt --variant hsm --split val --output runs/hsm_val

python -m hsm.evaluate --checkpoint runs/hsm_seed42/checkpoints/stage2_best.pt --variant hsm --split test --output runs/hsm_test
```

Each run writes `metrics.json` and `predictions.npz`. Use the same base encoder through `--model` if training used a local model directory.

Reference checkpoints are identified by their hashes in [models/checkpoints.json](../models/checkpoints.json). Weight files are external model artifacts or outputs of training. The included predictions can be evaluated directly with the [result reconstruction command](../results/README.md#reconstruct-saved-results).

## Training

```bash
python -m hsm.train --variant hsm --output runs/hsm_seed42
```

The two training phases use the following default configuration.

| Setting | Value |
|---|---|
| Seed | 42 |
| Epochs | 50 for the level head, then 50 for joint training |
| Batch size | 64 |
| Stage 1 head learning rate | 0.001 |
| Stage 2 encoder/head learning rates | 0.00001 / 0.002 |
| Contrastive weight | 1 |
| Contrastive temperature | 0.05 |
| Semantic temperature | 0.01 |
| Checkpoint selection | Highest full validation MRR, earliest tie |

The selected checkpoint is saved as `checkpoints/stage2_best.pt` within the output directory. Training reads the train and validation splits. The retained final diagnostic also uses a validation alias. Test evaluation is performed through the separate evaluation command.

To inspect the configuration before running:

```bash
python -m hsm.train --variant hsm --output runs/hsm_seed42 --dry-run
```

Loss definitions, category caching, padding, and optimizer settings are described in [docs/hsm.md](../docs/hsm.md).

## Ablations

The three ablations share the training entry point.

| Variant | Modification |
|---|---|
| `no-level` | Removes the level loss and the level probability term in ranking |
| `random-neg` | Uses a fixed global random pool of 20 negatives per category |
| `no-contrastive` | Removes the semantic contrastive loss |

```bash
python -m hsm.train --variant no-level --output runs/no_level_seed42
python -m hsm.train --variant random-neg --output runs/random_neg_seed42
python -m hsm.train --variant no-contrastive --output runs/no_contrastive_seed42
```

No-Level omits the first training phase. Pass the same variant name when evaluating its checkpoint:

```bash
python -m hsm.evaluate --checkpoint runs/no_level_seed42/checkpoints/stage2_best.pt --variant no-level --split test --output runs/no_level_test
```

## Metrics and reference results

- **D-Acc** accepts the reference category or one of its descendants.
- **Acc@1** requires an exact category-ID match.
- **MRR** uses the reference category's position in the complete ranking.
- **Level Acc** measures the explicit level prediction.

The reference test results use 1,245 requirements.

| Variant | D-Acc | Acc@1 | MRR | Level Acc |
|---|---:|---:|---:|---:|
| HSM | 0.513 | 0.493 | 0.605 | 0.806 |
| No-Level | 0.482 | 0.463 | 0.590 | — |
| Random-Neg | 0.401 | 0.379 | 0.509 | 0.772 |
| No-Contrastive | 0.357 | 0.334 | 0.460 | 0.742 |

Checkpoint and data provenance are documented in [docs/provenance.md](../docs/provenance.md).
