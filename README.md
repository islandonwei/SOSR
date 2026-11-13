# SOSR

Code and experimental artifacts for **Efficiently Answering Semantic Optimal Sequenced Route Queries with Dynamic Candidates**.

The framework answers route queries with ordered natural-language requirements. The Hierarchical Semantic Matcher (HSM) predicts one POI category per requirement. Fast Sequenced Shortest Paths (FSSP) plans a minimum-distance route through the resulting candidate sets.

## Components

| Component | Contents | Guide |
|---|---|---|
| HSM | Training, checkpoint evaluation, and three ablations | [hsm/README.md](hsm/README.md) |
| Routing | FSSP, routing baselines, and query benchmarks | [routing/README.md](routing/README.md) |
| Baselines | Embedding and LLM category-prediction baselines | [baselines/README.md](baselines/README.md) |
| Data | Semantic splits, road inputs, and file formats | [data/README.md](data/README.md) |
| Results | Reference tables, raw records, and plotting | [results/README.md](results/README.md) |

## Quick start

With Python 3.11 or later, run the following from the repository root in a Linux or WSL shell:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/check_data.py
python scripts/reproduce_results.py
python scripts/plot_results.py
```

These commands check the included semantic data, reconstruct the saved results, and generate the semantic comparison plots. Outputs are written to `runs/reproduced/` and `runs/figures/`.

For model training and checkpoint evaluation, follow the [HSM guide](hsm/README.md). For compilation and new route measurements, follow the [routing guide](routing/README.md).

## Data and model artifacts

The semantic dataset is included. Download the four road networks from the [road-data release asset](https://github.com/islandonwei/SOSR/releases/download/reproducibility-2026-11-13/SOSR-road-data-20261113.zip). Dataset sizes, extraction paths, and source information are in the [data guide](data/README.md).

Selected checkpoints can be produced by training or supplied as external model artifacts. Reference hashes are in [models/checkpoints.json](models/checkpoints.json).

The released routing driver uses calibrated geometric bounds. Historical manuscript plots and the supplemental routing comparison have separate version records. See [result provenance](docs/provenance.md).

## Citation and questions

Citation metadata is provided in [CITATION.cff](CITATION.cff). Please use the repository's Issues page for questions.

Project licensing is pending author selection. Model and dataset sources are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
