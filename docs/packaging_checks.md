# Packaging checks

The 2026-11-13 package was checked at its public entry points.

- All four HSM variants completed one applicable epoch and checkpoint evaluation with a synthetic tiny encoder, 12 categories and CPU execution. This validates packaging and checkpoint flow; it is not a repeat of the paper's full training.
- The routing driver compiled on Linux. Its 420 small graphs produced 1,680 successful four-method comparisons, including same-visit and endpoint semantics. The timeout fixture produced a censored result.
- Geometric arithmetic checks passed 20,000 edge interval checks and 20,000 rectangle checks, plus invalid/zero/coincident cases.
- Cached completion scores matched uncached tiny-Qwen scores on CPU, with maximum error zero.
- Ten saved semantic variants and all 21,600 routing measurements were reconstructed and compared with the selected result records.
- The four semantic input hashes and 20 road input hashes match the frozen sources.
- Plot generation and Python syntax checks passed. Generated figures are standard PDF/PNG outputs.

HSM smoke checks used Python 3.12.3, PyTorch 2.13.0 and Transformers 4.57.1. The archived full-ranking LLM measurements used Transformers 5.15.1. The full 7B SFT, 7B test evaluation and full road-network benchmark were not rerun during packaging.

Legacy C++ entry points still emit unused-return-value compiler warnings. The routing checks passed. GitHub Actions has been prepared locally; its hosted run occurs after upload.
