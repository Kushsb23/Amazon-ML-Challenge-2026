# Business Entity Resolution — Amazon ML Challenge 2026

Given business records from three independent sources, find every Source 2 / Source 3 record that refers to the
same real-world business as each Source 1 record. Submissions are scored by **macro-averaged F0.5 per Source 1
entity**, where entities with no match count too.

| | Macro F0.5 |
|---|---|
| Validation: 220,845 held-out entities, single pipeline | **0.9900** |
| Validation: 43,859 held-out entities, final ensemble | **0.9923** |
| Public leaderboard | **0.988** |

---

## Approach

```
raw records ─► normalisation ─► candidate generation ─► learned pruning ─► two-stage GBDT matcher
            ─► cross-encoder re-scoring ─► stacker ─► ensemble ─► expected-F0.5 decision rule ─► submission
```

### 1. Normalisation
- **Names:** lower-cased and folded to ASCII. Indic-script names go through a token transliteration map learned from training pairs. Legal forms (LLC, Pvt Ltd, SARL…) and honorifics are separated out, and DBA / domain / hashtag names are resolved.
- **Addresses:** parsed into house number, street, unit, city, locality and state, with abbreviations expanded.

### 2. Candidate generation
Three generators run within each country, and their results are merged:
- **Blocking keys (9):** house number with street, city, locality, rare name tokens, whole name and alternate name. Oversized blocks are skipped.
- **TF-IDF nearest neighbours** on three views: name character n-grams, address tokens, and a weighted name + address view.
- **Dense retrieval:** `intfloat/multilingual-e5-small`, fine-tuned on true pairs from training slices that no other stage uses. It is searched in three spaces: name, address, and fused (mean of both cosines).
  - **Forward:** each entity's top 20 records per space.
  - **Backward:** each record's top 3 entities per space, kept only near the entity's own top-20 frontier.

A pair is kept if at least three generators find it, or if it is a forward top-2, backward top-1 or TF-IDF top-1 hit. This gives **99.15% pair recall**.

Each kept pair gets about 40 retrieval signals:
- ranks and scores per space and direction;
- blocking-key and TF-IDF hit flags;
- exact cosines and **CSLS** hubness-corrected cosines;
- margins to the next-best candidate on either side.

A LightGBM pruner on these signals keeps **6.6 candidates per entity at 99.11% recall**. These pairs form `candidate_pairs.tsv`.

### 3. Two-stage matcher
- **Stage 1 (XGBoost, GPU):** about 100 pairwise features. These cover name similarity (fuzzy ratios, Jaro-Winkler, token overlap, acronyms, alternate names, name frequency), address agreement (house number equality, difference and containment, street, city, state, unit), rank and gap among the entity's candidates, and all retrieval signals.
- **Stage 2 (LightGBM):** adds group-consistency features. Each candidate is compared with the entity's most confident match, and the model checks whether the entity is the candidate's own best match in return (reciprocal best).
- Both stages use 5-fold CV grouped by entity, so stage 2 only sees out-of-fold stage-1 scores.

### 4. Cross-encoder re-scoring
The uncertain band is pairs with a stage-2 probability between 0.02 and 0.995 that are among a record's three best entities, about 12% of pairs. Two fine-tuned `FacebookAI/xlm-roberta-base` cross-encoders read both raw records jointly:
- **Model A:** field-labelled input (`name: … | address: …`) with noise augmentation that never changes digits.
- **Model B:** an ASCII-folded view, trained with synthetic near-miss negatives (changed house number, swapped legal form). It is kept only when it improves the stacker's cross-fitted AUC.

A **LightGBM stacker** then combines:
- the stage-2 logit and both cross-encoder logits;
- margins to the best rival entity competing for the same record;
- entity-level ranks and gaps;
- structured evidence (missing address, house-number agreement, name, city and street equality);
- 14 raw-string features (case, digits, punctuation, token order).

All competitor features are computed **within each 10% slice** of entities, so training, validation and test see the same competitor density. Band AUC: GBDT 0.949, cross-encoder 0.941, **stacked 0.975**.

### 5. Ensemble
A LightGBM ensembler combines this pipeline with a second, independently developed matcher. It works over the union of both candidate sets and uses:
- both pipelines' probabilities and cross-encoder logits, with their mean, max, min, per-entity ranks and gaps, and missing-side flags;
- 20 pair features;
- 12 name-focused features: IDF-weighted token overlap, rarest shared and rarest unmatched token, name frequency, acronym, Jaro-Winkler / token-sort / partial ratio, skeleton equality, and similarity to the entity's best other candidate;
- a **no-address specialist**, a LightGBM trained only on candidate pairs whose record has no address. These pairs account for 80% of the remaining misses; the specialist's AUC on them is 0.968, against 0.963 for stage 2.

It is fitted only on entities that are out-of-fold for both pipelines, and variants are selected on out-of-fold scores.

### 6. Decision rule
1. **Exclusivity:** each Source 2 / 3 record is assigned to at most one entity, the highest-scoring one.
2. **Match count:** for each entity, the number of matches *k* maximises its **expected F0.5** given the candidate probabilities, 1.25·Σtop-k p / (0.25·(Σp + ε) + k). Predicting no match is scored as Π(1 − p).
3. **Caps:** at most 5 Source 2 and 6 Source 3 matches per entity.

Rule parameters are tuned on out-of-fold predictions.

---

## Validation protocol

Training entities are split into ten 10% slices by `id % 10`:

| Slice | Use |
|---|---|
| m0 | Held-out validation, never used for training or tuning |
| m1–m5 | Matcher training (5-fold, grouped by entity) |
| m1–m2 | Stacker / ensemble fitting (out-of-fold) and decision-rule tuning |
| m3–m5 | Cross-encoder training |
| m6–m9 | Dense-retrieval fine-tuning |

The test set is processed in ten slices of the same size, so slice-level context features match training.

---

## Results

All numbers are on held-out validation entities.

| Configuration | Macro F0.5 |
|---|---|
| Two-stage GBDT, 15 candidates / entity | 0.9775 |
| + cross-encoder + stacker | 0.9836 |
| + second cross-encoder, slice-consistent rivals, structured evidence | 0.9846 |
| + dense-retrieval candidates | 0.9865 |
| **+ three-space retrieval, learned pruning, raw-string features** | **0.9900** |
| **+ ensemble with name features and no-address specialist** | **0.9923** |

| Detail (single pipeline) | Value |
|---|---|
| Candidate recall (6.6 / entity) | 99.1% |
| Stage-1 / stage-2 matcher | 0.9827 / 0.9859 |
| Pair precision / recall | 0.9981 / 0.9721 |
| Macro F0.5, India / US | 0.9906 / 0.9896 |

The test set also contains French records, which never appear in training. Country is treated as an open set, and French records go through the same pipeline.

---

## Repository structure

```
src/
  config.py, io_utils.py, evaluation.py     paths, fast TSV IO, metric (macro F0.5, recall)
  learn_translit.py, normalize.py            transliteration map, record normalisation
  blocking_keys.py, run_blocking.py          blocking-key candidates
  tfidf_knn.py                               TF-IDF nearest-neighbour candidates
  emb_retrieval.py                           multilingual-e5 fine-tuning and encoding
  v3_retrieval.py                            three-space forward/backward search + retrieval signals
  prune.py                                   LightGBM candidate pruning
  union_trim.py                              candidate union and pair features
  matcher.py, train_big.py                   stage-1 / stage-2 matcher, test scoring
  cross_encoder.py                           uncertain band, cross-encoders, stacker
  name_feats.py, spec_empty.py, make_bspec.py name features, no-address specialist
  ensemble.py                                two-pipeline ensembler
  postprocess.py, decision_exact.py          decision rules, tuning, submission writer
  run_pipeline.py, build_normalized.py       end-to-end driver, cached normalisation
  blend.py                                   stage-2 blend weights (XGBoost / LightGBM / CatBoost)
notebooks/                                   EDA, evaluation and normalisation checks
docs/methodology.md                          full technical write-up
```

### Diagnostic and utility scripts

| Script | Purpose |
|---|---|
| `audit_france.py` | Parsing and matcher-confidence report for an unseen country (default France) |
| `check_scores.py` | Sanity check of test scores against holdout |
| `compare_submissions.py` | Per-country diff between two `matching_results.tsv` files |
| `loco.py`, `loco_pseudo.py` | Leave-one-country-out experiments and pseudo-label self-training |
| `missed_pairs.py` | Inspect holdout true pairs that no candidate source produces |
| `make_package.py` | Build the challenge submission zip |

---

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux / macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
set BER_DATA_DIR=C:\path\to\dataset                       # Linux / macOS: export BER_DATA_DIR=/path/to/dataset
set BER_V3=1 & set BER_EMB=1
```

A CUDA GPU is used for dense retrieval, XGBoost and the cross-encoders (tested on an RTX 5070 Ti, 12 GB).

## Reproducing the submission

> **Note:** steps up to the single-pipeline result (**0.9900**) are fully reproducible from this repository. The final **0.9923** blend also needs the score exports of a second, independently developed matcher ("pipeline B", `BER_PIPE_B_DIR`), whose code is not part of this repository.
>
> Commands below use Windows `set` syntax; on Linux / macOS use `export VAR=value` (or prefix the command with `VAR=value`). `python src/run_pipeline.py` runs the single-pipeline flow end to end.

Run from `src/`:

```bash
python learn_translit.py && python normalize.py                      # normalised records
python run_blocking.py && python tfidf_knn.py                        # key + TF-IDF candidates
python emb_retrieval.py train && python emb_retrieval.py encode      # dense retrieval model + vectors
python v3_retrieval.py search && python v3_retrieval.py recall       # three-space search, signals
python prune.py fit && python prune.py apply                         # learned pruning
python union_trim.py build train m1   # ... m2 m3 m4 m5 holdout, and: build test m0 ... m9
python matcher.py features train m1   # ... m2 m3 m4 m5 holdout
python train_big.py stage1 && python train_big.py stage2 lgb && python train_big.py score_test

# cross-encoders (model A: BER_CE_VER=2 recipe; model B: ASCII view + synthetic negatives)
set BER_CE_VER=2 & set BER_CE_ASCII=1 & set BER_CE_SYNTH=0.1 & set BER_CE_OUT=..\work\ce_v2\model
python cross_encoder.py train
set BER_CE_VER=4 & set BER_CE_SECOND=..\work\ce_v2 & set BER_CE_RIVAL_SLICE=1 & set BER_CE_OUT_TAG=ce4s
python cross_encoder.py band && python cross_encoder.py score && python cross_encoder.py stack
set BER_SCORES=ce4s & set BER_TUNE_SUBSETS=m1,m2 & python postprocess.py tune

# ensemble (needs pipeline B's pair scores in BER_PIPE_B_DIR)
python spec_empty.py
set BER_BLEND_BASE=ce4s & set BER_BLEND_RICH=1 & set BER_BLEND_NORID=1 & set BER_BLEND_NAME=1 & set BER_BLEND_SPEC=1
set BER_BLEND_TAG=final & python ensemble.py
set BER_SCORES=final & python postprocess.py write                   # -> output/matching_results.tsv, candidate_pairs.tsv
```

Validate with the checker shipped with the challenge resources:

```bash
python utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t dataset/test --check-ids
```

---

## License

Released under the MIT License (see `LICENSE`). The challenge dataset is not included and remains subject to the challenge's own terms.

## Compliance

- Only the provided challenge data is used. There are no external lookups, geocoding or APIs.
- Pretrained models:
  - `FacebookAI/xlm-roberta-base` (MIT, 278M parameters);
  - `intfloat/multilingual-e5-small` (MIT, 118M parameters).

  Both are well within the 8B limit and fine-tuned only on training pairs.
- LightGBM (MIT), XGBoost and CatBoost (Apache-2.0) and scikit-learn (BSD-3) models are trained from scratch.
- No model is trained on test records.
- `matching_results.tsv` is a subset of `candidate_pairs.tsv` by construction.
