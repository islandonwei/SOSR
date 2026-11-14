# Provenance and scope

Assembled on 2026-11-13 from the research workspace and completed records.

## Data

The hierarchy has 1,245 categories. Train/validation/test sizes are 10,859/1,245/1,245. Each category has one validation and one test requirement. Training extends a 9,960-item split with 899 synthetic level2/3 requirements.

Archived sources identify 1,230 original and 15 Codex items in validation, and 1,244 original plus 1 Codex item in test. Original denotes the prior query collection. Its original author/generator and exact model version were not retained. Augmentation also lacks an archived underlying model version. The data are not described as newly collected real-user queries.

The test set has been used in historical experiments. Future configuration decisions should use validation.

## Semantic results

The selected original HSM uses contrastive weight 1 and semantic temperature 0.01, with strict-ID Acc@1=0.493173 and D-Acc=0.513253. Later prototype, EMA, sibling-quota, modified-data and ancestor-aggregation experiments are outside this release's primary table.

No-Level and Random-Neg use corrected weight 1 versions. publication_metrics.csv is the consistent table, which differs from several older drafts.

## Efficiency

The efficiency CSV combines complete-ranking LLM measurements with earlier encoder measurements. LLM means use 24 stratified queries and three repeats, weighted by test level distribution. Encoders use all 1,245 test items and three repeats. These are different batches. LLM timing includes category ranking and level prediction and excludes model loading and fixed prompt-prefix preparation. Storage is decimal GB.

The complete-ranking environment used AMD EPYC 9654, RTX PRO 6000 Blackwell, PyTorch 2.13.0 and Transformers 5.15.1. HSM packaging checks additionally use Transformers 4.57.1. Old free-generation times are not used to describe complete-ranking accuracy.

## Identities

The historical BERT row used xlm-roberta-base with mean pooling. Code/README call it XLM-R. The CSV retains the historical label for traceability.

Qwen-City is independently full-fine-tuned Qwen2.5 on CityInstruction-London, with final-epoch selection. It is not an official CityGPT checkpoint or SWFT reproduction.

DROMC denotes the existing Candidate-A* code. ROSE is an independent RNII-guided implementation.

## Plot versions

Archived Figures 6–10 have their original plotting data and Gnuplot scripts. Running a script generates its EPS output. The combination legends read FSSP. Earlier workloads/timings must remain distinct from the calibrated four-method comparison. The historical indexing-space plot alone does not establish whether it represents index-object memory, serialized disk size or peak RSS.

Frozen semantic and road inputs are identified by data/semantic/checksums.json and data/roads/manifest.json. Reference checkpoint hashes are recorded in configs/checkpoints.json. Git commits and release tags identify code versions. Compact prediction files preserve ranking order and labels.

The three reference CSVs remain in `results/`. Full predictions, raw routing measurements, and their original provenance records are in [SOSR-reference-results-20261114.zip](https://github.com/islandonwei/SOSR/releases/download/reproducibility-2026-11-13/SOSR-reference-results-20261114.zip). The 25 archived files are byte-identical to their copies in the original `reproducibility-2026-11-13` release. Archive and file checksums are in [configs/results.json](../configs/results.json).
