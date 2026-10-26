# Improvements log

Validation = held-out 10% of training entities (220,845 Source 1 records), never used for training or tuning.
Metric = macro F0.5 per Source 1 record.

| # | Add-on | Holdout macro F0.5 |
|---|---|---|
| 0 | Baseline pipeline: blocking + TF-IDF candidates, two-stage GBDT matcher, trained on 10% of train | 0.9753 |
| 1 | Trained on 50% of train, LightGBM stage 2 (chosen by out-of-fold blend search), expected-F0.5 decision rule | 0.9775 |
| 2 | **Cross-encoder A**: XLM-RoBERTa-base (multilingual, MIT) re-scores the ~6% of pairs the matcher is unsure about, reading both raw records together; a LightGBM stacker merges it with the matcher's score | 0.9836 |
| 3 | **Cross-encoder B + fixes**: second XLM-RoBERTa model on 500k pairs with name/address labelled separately and digit-safe augmentation; rival records counted per 10% slice for train, holdout and test alike; stacker uses both models plus missing-address, house-number and name/city/street evidence | 0.9846 |
| 4 | **Multilingual embedding search**: `intfloat/multilingual-e5-small` (MIT) fine-tuned on true pairs of four unused training slices, top-K name-embedding neighbours added to the candidates | 0.98652 |
| 5 | **Architecture v3 candidates**: three embedding spaces (name, address, fused), forward top-20 and backward top-10 searches, union with keys + TF-IDF, ~40 retrieval signals (ranks, CSLS, margins), learned pruning to 6.6 candidates / record at 99.1% pair recall; second cross-encoder with ASCII view + synthetic near-miss negatives; raw-string features in the stacker | 0.99004 |
| 6 | **Blend with the second pipeline's independent pipeline**: LightGBM stacker on both pipelines' scores, fitted on records out-of-fold for both | **0.99232** (on the 43,859 holdout records shared with its split C) |

## What each add-on fixed

- **Cross-encoder A:** catches matches the engineered features miss: native script vs transliteration, heavy typos, reordered words, padded numbers. Wrong matches fell from 4,052 to 2,216 and missed matches from 41,090 to 32,730.
- **Cross-encoder B:** a stronger model (AUC on the unsure pairs 0.963 → 0.973, combined 0.984 → 0.987). Wrong matches fell further, to 1,514.
- **Rival-count fix:** the first version let test pairs see about twice as many competing records as in training. This affects only test, so its gain shows on the leaderboard, not the holdout.

- **Embedding search:** catches matches with different wording or script that neither blocking keys nor TF-IDF see; candidates missing a true match fell from 2.6% to 1.6%.
- **v3 candidates:** better and smaller candidate sets (15 -> 6.6 per record, recall 98.4% -> 99.1%). Stage-2 matcher 0.9798 -> 0.9859 before any cross-encoder.
- **Blend:** the two pipelines make different mistakes. Ours alone 0.98992 and pipeline B alone 0.99143 on the shared records; blended 0.99232.

## Final numbers (holdout)

| | ce4 (our pipeline) | Blend (ours + second pipeline) |
|---|---|---|
| Macro F0.5 | 0.99004 | 0.99232 (shared records) |
| India / US | 0.9906 / 0.9896 | - |
| Pair precision / recall | 0.9981 / 0.9721 | - |
| True pairs never in the candidates | 6,779 (0.9%) | - |

## Compliance

Only the challenge data is used. Pretrained models: `FacebookAI/xlm-roberta-base` (MIT, 278M parameters, two cross-encoders) and `intfloat/multilingual-e5-small` (MIT, 118M parameters, retrieval), both fine-tuned on training pairs only. Nothing is trained on test records.

## Next ideas (not done yet)

- France: the holdout has no French records, so French handling is judged only by the leaderboard (`BLENDFR` / `CE4FR` variants treat France like the US).
- Train the matcher on 100% of the training data.
