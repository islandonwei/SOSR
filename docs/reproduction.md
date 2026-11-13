# Reproduction Guide

Run all commands from the repository root. The guides below provide the environment, input files, commands, and expected outputs for each component.

| Task | Guide | Main output |
|---|---|---|
| Prepare and verify input data | [Data](../data/README.md) | Verified semantic files and road inputs |
| Train HSM and evaluate a checkpoint | [HSM](../hsm/README.md) | Checkpoint, metrics, and complete rankings |
| Train and evaluate the ablations | [HSM ablations](../hsm/README.md#ablations) | One selected checkpoint per variant |
| Run embedding and LLM baselines | [Baselines](../baselines/README.md) | Metrics and category rankings |
| Compile and measure routing methods | [Routing](../routing/README.md) | Per-query measurements and aggregates |
| Reconstruct tables and draw plots | [Results](../results/README.md) | Metric JSON, routing CSV, PDF/PNG plots |

## HSM

Use the [HSM guide](../hsm/README.md) for training and checkpoint evaluation. Losses, optimizer settings, category caching, and ranking details are documented in [hsm.md](hsm.md).

### Ablation experiments

No-Level, Random-Neg, and No-Contrastive use the shared training entry point. Their definitions and commands are in [HSM ablations](../hsm/README.md#ablations).

## Routing experiments

The [routing guide](../routing/README.md) covers compilation, data preparation, and the fixed benchmark. Detailed measurement semantics are in [routing.md](routing.md).

## Semantic baselines

See [baselines/README.md](../baselines/README.md) for XLM-R, Sentence-T, Qwen Zero/Few, and Qwen-City. The CityInstruction preparation and full-SFT recipe are in [Qwen-City training](../baselines/qwen_city/README.md).

## Reproducing tables and plots

```bash
python scripts/reproduce_results.py
python scripts/plot_results.py
```

Output filenames, metric definitions, historical record limitations, and manuscript figure mappings are documented in [results/README.md](../results/README.md).

## Version information

The manuscript plot inputs and the supplemental calibrated routing comparison use different experiment versions. Read [provenance.md](provenance.md) when interpreting the saved results.

Packaging checks are documented in [packaging_checks.md](packaging_checks.md). Citation metadata and source information are in [CITATION.cff](../CITATION.cff) and [THIRD_PARTY.md](../THIRD_PARTY.md).
