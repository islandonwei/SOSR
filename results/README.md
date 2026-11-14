# Experimental results

This directory contains three reference tables:

| File | Contents |
|---|---|
| `hsm/publication_metrics.csv` | Semantic comparison and ablation results |
| `routing/RESULTS.csv` | Fixed four-method routing comparison |
| `efficiency/measurements.csv` | Model latency and storage measurements |

Full predictions and individual routing measurements are in [SOSR-reference-results-20261114.zip](https://github.com/islandonwei/SOSR/releases/download/reproducibility-2026-11-13/SOSR-reference-results-20261114.zip) (about 20 MB). They are downloaded separately; generated outputs are not tracked in Git.

## Reconstruct saved results

From the repository root, install the lightweight dependencies and run:

```bash
pip install -r requirements.txt
python scripts/download_results.py
python scripts/reproduce_results.py
```

The downloader checks the archive and all 25 files against [configs/results.json](../configs/results.json), then stores them in `runs/reference/`. Verified existing files are reused. For an offline download, use `python scripts/download_results.py --archive /path/to/SOSR-reference-results-20261114.zip`.

The reconstruction command checks semantic metrics and routing statistics against the references without rerunning models or route queries. Outputs go to `runs/reproduced/`: `hsm_metrics.json` and `routing/RESULTS.{csv,json,txt}`.

## Semantic comparison and ablations

| Location | Contents |
|---|---|
| `runs/reference/hsm/predictions/` | Complete ID rankings for HSM, its three ablations, and four LLM conditions |
| `runs/reference/hsm/name_only/` | Historical XLM-R and Sentence-T Top-5 name predictions |
| `runs/reference/hsm/provenance.json` | Source hashes and ranking-conversion records |

D-Acc accepts the reference category or one of its descendants. Acc@1 requires exact ID equality. MRR uses the reference category's full rank. Level Acc uses the explicit level prediction.

The HSM reference configuration uses seed 42, contrastive weight 1, and semantic temperature 0.01.

The XLM-R and Sentence-T archives contain Top-5 names and aggregate MRR. Their complete ID rankings are unavailable. The reconstruction script derives D-Acc/Acc@1 where uniquely determined and marks their full-ID MRR as unavailable. Historical displayed MRR and its bounds remain in the reference CSV.

## Routing comparison

The download includes 21,600 measurements for 1,800 fixed queries and three paired repetitions in `runs/reference/routing/raw/`. The CSV summary remains in `routing/RESULTS.csv`; the detailed JSON summary and original audit are in the download.

The comparison uses FSSP with calibrated geometric bounds, an RNII-guided ROSE implementation, the existing Candidate-A* implementation labelled DROMC, and layered multi-source Dijkstra.

Reported latency covers distance and ordered service-vertex reconstruction. Each run has a 30-second query budget. Censored values remain unknown above that budget when computing median and P95. Details are in [docs/routing.md](../docs/routing.md).

## Model efficiency

`efficiency/measurements.csv` contains existing latency and storage measurements. Encoder timings use all 1,245 test requirements. LLM timings use 24 stratified requirements and three repetitions, weighted by the test set's level distribution. These measurements come from different batches.

LLM time includes complete category ranking and separate level prediction. Storage is decimal GB. See [docs/provenance.md](../docs/provenance.md) for the original measurement scope and environments.

## Plots

Semantic comparison and ablation plots use the included CSV and do not require the record download:

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
