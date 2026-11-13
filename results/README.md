# Experimental results

This directory contains reference semantic predictions, model-efficiency records, and raw measurements from the fixed routing comparison.

## Reconstruct saved results

From the repository root, install the lightweight dependencies and run:

```bash
pip install -r requirements.txt
python scripts/reproduce_results.py
```

The command reconstructs semantic metrics and routing statistics, then checks them against the stored reference records. It writes:

```text
runs/reproduced/
├── hsm_metrics.json
└── routing/
    ├── RESULTS.csv
    ├── RESULTS.json
    └── RESULTS.txt
```

## Semantic comparison and ablations

| Location | Contents |
|---|---|
| `hsm/publication_metrics.csv` | Main comparison and ablation table values |
| `hsm/predictions/` | Complete ID rankings for HSM, its three ablations, and four LLM conditions |
| `hsm/name_only/` | Historical XLM-R and Sentence-T Top-5 name predictions |
| `hsm/provenance.json` | Source hashes and ranking-conversion records |

D-Acc accepts the reference category or one of its descendants. Acc@1 requires exact ID equality. MRR uses the reference category's full rank. Level Acc uses the explicit level prediction.

The HSM reference configuration uses seed 42, contrastive weight 1, and semantic temperature 0.01.

| Variant | D-Acc | Acc@1 | MRR | Level Acc |
|---|---:|---:|---:|---:|
| HSM | 0.513 | 0.493 | 0.605 | 0.806 |
| No-Level | 0.482 | 0.463 | 0.590 | — |
| Random-Neg | 0.401 | 0.379 | 0.509 | 0.772 |
| No-Contrastive | 0.357 | 0.334 | 0.460 | 0.742 |

The XLM-R and Sentence-T archives contain Top-5 names and aggregate MRR. Their complete ID rankings are unavailable. The reconstruction script derives D-Acc/Acc@1 where uniquely determined and marks their full-ID MRR as unavailable. Historical displayed MRR and its bounds remain in the reference CSV.

## Routing comparison

`routing/raw/` contains 21,600 measurements for 1,800 fixed queries and three paired repetitions. Aggregates are in `routing/RESULTS.csv` and `routing/RESULTS.json`.

The comparison uses FSSP with calibrated geometric bounds, an RNII-guided ROSE implementation, the existing Candidate-A* implementation labelled DROMC, and layered multi-source Dijkstra.

Reported latency covers distance and ordered service-vertex reconstruction. Each run has a 30-second query budget. Censored values remain unknown above that budget when computing median and P95. Details are in [docs/routing.md](../docs/routing.md).

## Model efficiency

`efficiency/measurements.csv` contains existing latency and storage measurements. Encoder timings use all 1,245 test requirements. LLM timings use 24 stratified requirements and three repetitions, weighted by the test set's level distribution. These measurements come from different batches.

LLM time includes complete category ranking and separate level prediction. Storage is decimal GB. See [docs/provenance.md](../docs/provenance.md) for the original measurement scope and environments.

## Plots

Generate semantic comparison and ablation plots:

```bash
python scripts/plot_results.py
```

The generated PDFs and PNGs are saved in `runs/figures/`.

Historical manuscript plot sources are in `figures/archived/`:

| Stems | Manuscript figure |
|---|---|
| `0–3` | Figure 6, source–target distance groups |
| `4–7` | Figure 7, requirement sequence length |
| `8–11` | Figure 8, category-frequency combinations |
| `16–17` | Figure 9, pruning distance-call counts |
| `18–19` | Figure 10, historical index costs |

To regenerate a historical plot, install Gnuplot and run its script from that directory:

```bash
cd figures/archived
gnuplot 8
```

The historical plot inputs and the supplemental calibrated routing comparison use different experiment versions. Their data should be interpreted with the [version and provenance notes](../docs/provenance.md).
