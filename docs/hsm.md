# HSM protocol

The encoder uses mean pooling and L2 normalization, followed by a linear level classifier. Category names alone are encoded. Hierarchy paths form negative pools and define D-Acc.

Stage1 freezes encoder parameters and trains the head for 50 epochs at learning rate0.001. The original forward/dropout behavior is retained. Stage2 starts from the final stage 1 state, trains the encoder at 1e-5 and head at 0.002, and selects the earliest checkpoint with highest full validation MRR. Batch size is 64.

The level loss is label-smoothed cross entropy (0.1) plus 0.5 times the mean squared difference of cumulative six-level distributions. InfoNCE temperature is 0.05 and matching weight is 1. Category embeddings are detached and refreshed before each epoch and validation. Gradient clipping is norm 1 on head parameters in stage 1 and encoder parameters in stage 2. AdamW decay is 0.01. Stage2 uses 10% warmup and a cosine schedule.

Ranking uses cosine /0.01 + log P(category level | requirement). The final prediction is one category ID.

Validation follows the trainer's fixed 64-token query padding and query batch 64. Test queries use dynamic padding and batch 128 as in the saved checkpoint evaluations. Category encoding uses dynamic padding and batch 128 in both cases.

| CLI variant | Change |
|---|---|
| hsm | hierarchy-hard caps 10/5/5 for same/upper/lower levels |
| no-level | omit stage 1, level loss and level ranking term |
| random-neg | fixed global pool of 20 distinct negatives per category, seeded by category ID |
| no-contrastive | zero matching loss; encoder/head trained by the level loss |

These are the selected archived runs. Random-Neg does not denote the later within-hierarchy-pool random experiment.

The legacy functions are retained. Packaging exposes model/data paths and renames the old final test-labelled diagnostic to its actual validation-alias role. Selection remains validation MRR. Implementation versions are recorded in the Git history.

## Metrics

- D-Acc credits the reference category or a strict descendant along its full taxonomy path.
- Acc@1 requires exact category-ID equality.
- MRR uses the reference ID's rank over all categories.
- Level Acc uses the explicit level prediction. No-Level and embedding retrieval have no level accuracy.

Archived ranking order, including numerical tie order, is preserved. Release indices are remapped to categories.csv order without re-sorting model scores.

XLM-R and Sentence-T retain category names and Top-5 only. Evaluation enumerates ambiguous name-to-ID assignments consistent with the saved legacy hit count and reports D-Acc/Acc@1 only when uniquely determined. Their historical three-decimal MRR and bounds remain in publication_metrics.csv. It is not a newly reconstructed exact full-ID MRR.

Checkpoints and selected epochs are recorded in configs/checkpoints.json where available. Fixed seeds do not promise bitwise identical retraining across GPU/library versions. Saved rankings provide exact table reconstruction.
