# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

*Technical write-up originally prepared for the challenge submission (27 September 2026).*

---

## 1. Executive Summary

Every record is normalised once. Candidates are generated from three sources: nine blocking keys, a three-view TF-IDF nearest-neighbour search, and a fine-tuned multilingual-e5 embedding search in three spaces (name, address, fused), run forward and backward. A LightGBM pruner then keeps **6.6 candidates per Source 1 entity while retaining 99.1% of true pairs**.

A two-stage gradient-boosted matcher (XGBoost, then LightGBM with group-consistency features) scores every candidate. Two fine-tuned **XLM-RoBERTa cross-encoders** re-read the ~12% of pairs where the matcher is unsure. A LightGBM stacker merges them with competitor margins and raw-string evidence.

The final submission **blends our pipeline with a second, independently developed pipeline**. A stacked LightGBM takes both pipelines' scores, pair features, name-focused features and a specialist model for records with no address. An expected-F0.5 decision rule with record exclusivity then produces the matches.

| Result | Macro F0.5 |
|---|---|
| Holdout, 220,845 entities, our pipeline | **0.99004** |
| Holdout, 43,859 entities shared with the second pipeline's held-out split, blend | **0.99232** |
| Leaderboard | **0.988** |

---

## 2. Methodology

### 2.1 Problem Analysis

| Finding | Consequence for the design |
|---|---|
| Each Source 1 entity has 3.46 true matches on average (India 3.4645, US 3.4591); 5.6% have none | Per-entity decision rule, and singletons must be predicted as empty |
| Each Source 2/3 record belongs to at most one entity | Global record exclusivity: each record goes to its best-scoring Source 1 |
| Noise: typos, transliteration (Indic scripts), abbreviations, legal-form changes, reordered or missing address parts | Normalisation plus fuzzy, TF-IDF, embedding and cross-encoder evidence |
| 80% of the remaining missed pairs have an empty target address | Name-only evidence matters; a specialist model handles these pairs |
| Test contains France (15%), which is absent from training | Country treated as an open set; French handling judged only on the leaderboard |
| Metric is macro F0.5 (precision-weighted, averaged per entity) | Expected-F0.5 decision per entity; pair precision is kept at about 99.8% |

### 2.2 Solution Strategy

```
raw records → normalisation → candidates (keys ∪ TF-IDF ∪ e5 fwd/bwd, 3 spaces) → LightGBM pruning (6.6/entity)
  → stage-1 XGBoost → stage-2 LightGBM (group consistency) → cross-encoders on the uncertain band → stacker
  → blend with second pipeline (+ name features + no-address specialist) → expected-F0.5 rule + exclusivity
```

**Validation.** Training entities are split into 10% slices by `id % 10`. The holdout (slice m0) is never used for training.

| Stage | Slices used |
|---|---|
| Matcher training | m1–m5 (5-fold, grouped by entity) |
| Stacker and blender fitting | m1–m2 (out-of-fold) |
| Retrieval embedding fine-tuning | m6–m9 |

**Test.** Test is processed in ten slices so that every competitor-count feature has the same distribution as in training.

---

## 3. Candidate Generation (Architecture v3)

1. **Blocking keys (9):** house number with street, city, locality or rare name tokens, whole-name and alternate-name keys. Very large blocks are skipped.
2. **TF-IDF kNN (3 views):** name character n-grams, address tokens, and a weighted name + address view.
3. **Embedding search:** `intfloat/multilingual-e5-small` (MIT, 118M parameters), fine-tuned on true pairs from slices m6–m9.
   - Spaces: name (n), address (a), fused (mean of the n and a cosines), all computed from one pair of GPU matrix multiplies.
   - **Forward:** each entity's top 20 records per space.
   - **Backward:** each record's top 3 entities per space, kept only near the entity's own top-20 frontier.
4. **Union filter:** a pair is kept if at least 3 generators hit it, or it is a forward top-2, backward top-1 or TF-IDF top-1 hit.
   - Result: 45.5 pairs per entity at **99.15% recall**.
5. **Retrieval signals (~40 per pair):** ranks and scores per space and direction, key/TF-IDF hit flags, exact and **CSLS** hubness-corrected cosines, and margins to the next-best candidate on each side.
6. **LightGBM pruning** on these signals, with the threshold set for at most 0.05% recall loss.
   - Result: **6.6 pairs per entity at 99.11% recall** (the previous 15-candidate set reached 98.4%).

---

## 4. Matching Model

**Stage 1 (XGBoost, GPU).** Trained on 7.3M pairs from m1–m5 with 5-fold CV grouped by entity. It uses about 100 features:
- name similarities: fuzzy ratios, Jaro-Winkler, token set, acronym, alternate name, name frequency;
- address similarities: house-number equality, difference and containment, street/city/state, unit;
- rank and gap among the entity's candidates;
- all retrieval signals.

The top features are `prune_c`, `b_empty`, the fused-space scores and `num_diff`. Holdout F0.5: **0.9827**.

**Stage 2 (LightGBM).** Adds group-consistency features: agreement with the entity's best match, reciprocal best match, and rank and gap of the stage-1 probability. Holdout F0.5: **0.9859** (0.9798 on the old candidate set).

### Cross-encoder re-scoring of the uncertain band

- **Band:** stage-2 probability between 0.02 and 0.995 and among the top 3 of the record's competitors (about 12% of pairs). Competitor features are computed **within each 10% slice** for training, holdout and test alike.
- **Model B:** `FacebookAI/xlm-roberta-base` (MIT, 278M parameters) trained on 500k pairs.
  - Field-labelled input: `name: … | address: …`
  - Noise augmentation that never alters digits
- **Model v2:** the same base model with an ASCII-folded view and 10% synthetic near-miss negatives (house number changed, legal form swapped). It is kept only when it raises the stacker's cross-fitted AUC.
- **Stacker (LightGBM):**
  - inputs: stage-2 logit, both cross-encoder logits, margins to the best rival Source 1 for the same record, entity-level ranks and gaps, structured evidence, and 14 raw-string features (case, digits, punctuation, token order);
  - fitting: fitted on m1–m2, which neither the matcher training nor the cross-encoders' training ever touched.
  - Band AUC: stage-2 0.949, cross-encoder 0.941, **stacked 0.975**.

### Blend with the second pipeline (final)

- **Frame:** the union of both candidate sets, 14.7M test pairs.
- **Inputs:**
  - our final score, stage-1/stage-2 scores and cross-encoder logits;
  - the second pipeline's matcher probability, cross-encoder logit, final score and link flag;
  - means, max and min of the two pipelines' scores, per-entity ranks and gaps, and missing-side flags;
  - 20 pair features;
  - 12 **name-focused features:** IDF-weighted token overlap, rarest shared/missing token, name frequency, acronym, Jaro-Winkler / token-sort / partial ratio, skeleton equality, and similarity to the entity's best other candidate.
- **No-address specialist:** a LightGBM trained on the 700k no-address pairs of m1–m5, with 5-fold out-of-fold scores. On holdout no-address pairs its AUC is **0.9684** against 0.9626 for stage 2.
- **Fitting:** the blender is fitted only on entities that are out-of-fold for both pipelines (split C ∩ m1–m2, 88k entities) with 2-fold cross-fitting. Variants are chosen on out-of-fold data.

### Decision rule

1. **Exclusivity:** each record goes to its highest-scoring entity.
2. **Match count:** per entity, pick the number k that maximises expected F0.5, 1.25·Σtop-k p / (0.25·(Σp + extra) + k). Predicting nothing scores Π(1 − p).
3. **Caps:** at most 5 Source 2 and 6 Source 3 matches per entity.

The parameters are tuned on out-of-fold data.

---

## 5. Results & Error Analysis

| Configuration (holdout) | Macro F0.5 |
|---|---|
| Rule-based baseline | 0.638 |
| Two-stage GBDT, 15-candidate set | 0.9775 |
| + cross-encoder A + stacker | 0.9836 |
| + cross-encoder B, per-slice rivals, structured evidence | 0.9846 |
| + e5 embedding candidates (ce3) | 0.98652 |
| **+ architecture v3 candidates, second cross-encoder, raw-string features (ce4)** | **0.99004** |
| Blend with second pipeline (shared 43,859 entities): ours / pipeline B / ensemble | 0.98992 / 0.99143 / **0.99232** |

**ce4 detail:**
- By country: India 0.9906, US 0.9896.
- Pair precision / recall: 0.9981 / 0.9721.
- Singletons: F 0.993.

**Remaining errors:**

| Missed true pairs | Count | Share with an empty target address |
|---|---|---|
| Never in the candidates | 6,779 | 6,104 |
| In the candidates but rejected | 14,531 | 10,886 |
| **Total** | 21,310 | 80% |

- **Candidate ceiling:** perfect selection among the current candidates would score 0.9974 on holdout.
- **Leaderboard gap:** holdout is 0.992 on India/US; leaderboard is 0.988. The gap is attributed to **France** (15% of test, ~0.964 estimated). Both pipelines, built independently, agree on most French predictions.
- **Label-free France fixes tested:** country recoding, French→English vocabulary mapping, and count calibration. None moved the leaderboard.
- **Test-only fix:** competitor features were made slice-consistent in the stacker. Before the fix, the share of cross-encoder pairs with a rival was 23% in stacker training, 9% on holdout and 31% on test.

---

## 6. Conclusion

The largest gains came from three changes:
1. Better candidates: multi-space embedding retrieval with learned pruning (+0.0035).
2. Cross-encoders on the uncertain band with a stacker (+0.007 over GBDT alone).
3. Blending two independent pipelines (+0.002).

Keeping every competitor feature consistent between training and test mattered as much as model capacity.

**Next steps:**
1. Name-only retrieval for records with no address.
2. A cross-encoder trained on mined real hard negatives.
3. Scoring both pipelines on the full candidate union.
4. Cross-fitting the pruner.
5. A leave-one-country-out study toward French generalisation.

---

## Appendix

### A. Code Artefacts

| File | Role |
|---|---|
| `normalize.py`, `learn_translit.py` | Normalisation, Indic transliteration map |
| `run_blocking.py`, `tfidf_knn.py` | Blocking keys, TF-IDF kNN |
| `emb_retrieval.py`, `v3_retrieval.py`, `prune.py` | e5 fine-tuning/encoding, three-space forward/backward search + signals, LightGBM pruning |
| `union_trim.py`, `matcher.py`, `train_big.py` | Candidate features, stage-1/2 matcher, test scoring |
| `cross_encoder.py` | Band, cross-encoder training/scoring, stacker (slice-consistent rivals) |
| `postprocess.py`, `decision_exact.py` | Decision rules, tuning, submission writer |
| `ensemble.py`, `name_feats.py`, `spec_empty.py` | Two-pipeline blender, name features, no-address specialist |

### B. Compliance

- Only challenge data is used.
- Pretrained models: XLM-RoBERTa-base (MIT, 278M) and multilingual-e5-small (MIT, 118M), both fine-tuned on training pairs only.
- LightGBM, XGBoost, CatBoost and scikit-learn are trained from scratch.
- Nothing is trained on test records.
- The official validator passes with `--check-ids`.
