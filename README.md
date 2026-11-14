# SOSR

Code for **Efficiently Answering Semantic Optimal Sequenced Route Queries with Dynamic Candidates**.

HSM maps each natural-language requirement to a POI category. FSSP then finds a shortest route through the predicted candidate sets in the required order.

## Requirements

- Python 3.11 or later
- Linux or WSL and a C++17 compiler for routing experiments
- HSM dependencies: [requirements-hsm.txt](requirements-hsm.txt)

Run the commands below from the repository root. The HSM examples use CUDA. The embedding and LLM baselines have separate [setup and evaluation instructions](baselines/README.md).

## Data

The semantic dataset is included in `data/semantic/`: 10,859 training requirements, 1,245 validation requirements, and 1,245 test requirements.

Download the [road data](https://github.com/islandonwei/SOSR/releases/download/reproducibility-2026-11-13/SOSR-road-data-20261113.zip) for NYC, BAY, COL, and FLA and extract the ZIP at the repository root. The files should appear under `data/roads/`.

See [data/README.md](data/README.md) for file formats and [THIRD_PARTY.md](THIRD_PARTY.md) for data and model sources.

## HSM

Create the environment and train HSM:

```bash
python -m venv .venv-hsm
source .venv-hsm/bin/activate
pip install -r requirements-hsm.txt

python scripts/check_data.py
python -m hsm.train --variant hsm --output runs/hsm_seed42
```

Evaluate the checkpoint selected on the validation set:

```bash
python -m hsm.evaluate \
    --checkpoint runs/hsm_seed42/checkpoints/stage2_best.pt \
    --variant hsm --split test --output runs/hsm_test
```

Trained weights are not included. Training settings and commands for No-Level, Random-Neg, and No-Contrastive are in [hsm/README.md](hsm/README.md).

## FSSP

After extracting the road data, compile and run the routing comparison:

```bash
python scripts/check_data.py --roads data/roads
python routing/build.py
python routing/benchmark.py --data-root data/roads --output runs/routing_1800
```

This runs the four methods on 1,800 fixed queries, with three repetitions. Baseline implementations, timing settings, and smaller test runs are described in [routing/README.md](routing/README.md).

## Results

The final CSV tables are in [results/](results/README.md). To recompute metrics from saved predictions and routing measurements, without rerunning the experiments:

```bash
pip install -r requirements.txt
python scripts/download_results.py
python scripts/reproduce_results.py
python scripts/plot_results.py
```

The download is about 20 MB. Tables and semantic plots are written to `runs/reproduced/` and `runs/figures/`. The [results README](results/README.md) explains the metrics and how to regenerate the other figures. Differences between the historical plots and the later routing comparison are listed in [docs/provenance.md](docs/provenance.md).

## Citation

If you use this code, please cite our work. See [CITATION.cff](CITATION.cff) for citation information.
