"""Matching model: pairwise features, stage-1 GBM, group-consistency features + stage-2 GBM.

Inputs: work/cand_{split}_{subset}.parquet from union_trim.py trim (top-N per S1 with union/context features).
    train m1      -> training data (5-fold CV grouped by S1 -> honest out-of-fold scores)
    train holdout -> evaluation (scored with the average of the 5 fold models)
    test m0..m9   -> submission scoring (each tenth of test looks like one train slice)

    python matcher.py features train m1          # pairwise features -> work/feat_train_m1.parquet
    python matcher.py features train holdout
    python matcher.py stage1                      # stage 1 -> OOF on m1, scores holdout, prints holdout F0.5
    python matcher.py stage2                      # group-consistency features + stage-2 model, prints holdout F0.5
    python matcher.py score_test                  # features + stage1 + stage2 for test m0..m9 -> work/scores_test.parquet

Models are XGBoost (Apache-2.0) on the GPU when available (device='cuda'), else CPU hist.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from blocking_keys import encode_id
import tfidf_knn as T
import union_trim as U

NFOLD = 5
MODEL_DIR = os.path.join(WORK_DIR, "models")
NORM_COLS = ["entity_id", "country", "n_full", "n_core", "n_alt", "n_nospace", "n_skel", "legal", "is_domain",
             "was_indic", "a_full", "a_street", "a_num", "a_city", "a_state", "a_unit", "a_empty"]


# ------------------------------------------------------------------ helpers
def _s(x):
    return x.fillna("").astype(str).to_numpy()


def _pair_scores(a, b, scorer, workers=-1):
    from rapidfuzz.process import cpdist
    out = cpdist(list(a), list(b), scorer=scorer, workers=workers, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)
    out[(a == "") | (b == "")] = -1
    return out


def _jaccard(a, b):
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            out[i] = -1
            continue
        sx, sy = set(x.split()), set(y.split())
        out[i] = len(sx & sy) / len(sx | sy)
    return out


def _initials(a):
    return np.array(["".join(t[0] for t in x.split()) if x.count(" ") >= 1 else "" for x in a], dtype=object)


_NORM_CACHE = {}


def load_norm(split):
    """S1 and pool normalised records indexed by int id, plus name-frequency tables (chain signal)."""
    if split in _NORM_CACHE:
        return _NORM_CACHE[split]
    t0 = time.time()
    s1 = pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s1.parquet"), columns=NORM_COLS)
    pool = pd.concat([pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet"), columns=NORM_COLS)
                      for k in (2, 3)], ignore_index=True)
    for d in (s1, pool):
        d["id"] = encode_id(d.entity_id)
        d.drop(columns="entity_id", inplace=True)
        d.set_index("id", inplace=True)
    key_p = pool.country.astype(str) + "|" + pool.n_core.astype(str)
    key_s = s1.country.astype(str) + "|" + s1.n_core.astype(str)
    freq_pool = key_p.value_counts()
    freq_s1 = key_s.value_counts()
    _NORM_CACHE[split] = (s1, pool, freq_pool, freq_s1)
    print(f"  norm {split} loaded ({time.time()-t0:.0f}s)", flush=True)
    return _NORM_CACHE[split]


# ------------------------------------------------------------------ pairwise features
def t7_features(c: pd.DataFrame, split: str) -> pd.DataFrame:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    t0 = time.time()
    s1, pool, freq_pool, freq_s1 = load_norm(split)
    A = s1.reindex(c.s1.to_numpy())
    B = pool.reindex(c.rid.to_numpy())
    f = {}
    an, bn = _s(A.n_core), _s(B.n_core)
    f["n_ratio"] = _pair_scores(an, bn, fuzz.ratio)
    f["n_partial"] = _pair_scores(an, bn, fuzz.partial_ratio)
    f["n_tsort"] = _pair_scores(an, bn, fuzz.token_sort_ratio)
    f["n_tset"] = _pair_scores(an, bn, fuzz.token_set_ratio)
    f["n_jw"] = _pair_scores(an, bn, JaroWinkler.normalized_similarity)
    f["n_full_ratio"] = _pair_scores(_s(A.n_full), _s(B.n_full), fuzz.ratio)
    f["n_skel_ratio"] = _pair_scores(_s(A.n_skel), _s(B.n_skel), fuzz.ratio)
    f["n_jacc"] = _jaccard(an, bn)
    f["n_nospace_eq"] = ((_s(A.n_nospace) == _s(B.n_nospace)) & (an != "")).astype(np.int8)
    f["n_first_eq"] = (pd.Series(an).str.split(" ").str[0].to_numpy() ==
                       pd.Series(bn).str.split(" ").str[0].to_numpy()).astype(np.int8)
    ia, ib = _initials(an), _initials(bn)
    ans, bns = _s(A.n_nospace), _s(B.n_nospace)
    f["n_acronym"] = (((ia != "") & (ia == bns)) | ((ib != "") & (ib == ans))).astype(np.int8)
    aa, ba = _s(A.n_alt), _s(B.n_alt)
    f["n_alt_match"] = (((ba != "") & (ba == an)) | ((aa != "") & (aa == bn))).astype(np.int8)
    f["n_len_a"] = np.array([len(x) for x in an], np.int16)
    f["n_len_b"] = np.array([len(x) for x in bn], np.int16)
    f["was_indic_b"] = B.was_indic.fillna(False).astype(np.int8).to_numpy()
    f["is_domain_b"] = B.is_domain.fillna(False).astype(np.int8).to_numpy()
    ctry = _s(A.country)
    f["freq_pool_a"] = pd.Series(np.char.add(np.char.add(ctry.astype(str), "|"), an.astype(str))).map(freq_pool).fillna(0).to_numpy(np.float32)
    f["freq_pool_b"] = pd.Series(np.char.add(np.char.add(ctry.astype(str), "|"), bn.astype(str))).map(freq_pool).fillna(0).to_numpy(np.float32)
    f["freq_s1_a"] = pd.Series(np.char.add(np.char.add(ctry.astype(str), "|"), an.astype(str))).map(freq_s1).fillna(0).to_numpy(np.float32)
    # address
    af, bf = _s(A.a_full), _s(B.a_full)
    f["a_tset"] = _pair_scores(af, bf, fuzz.token_set_ratio)
    f["a_tsort"] = _pair_scores(af, bf, fuzz.token_sort_ratio)
    f["a_jacc"] = _jaccard(af, bf)
    f["a_street_ratio"] = _pair_scores(_s(A.a_street), _s(B.a_street), fuzz.token_sort_ratio)
    f["a_city_ratio"] = _pair_scores(_s(A.a_city), _s(B.a_city), fuzz.ratio)
    ua, ub = _s(A.a_unit), _s(B.a_unit)
    f["a_unit_eq"] = np.where((ua == "") | (ub == ""), -1, (ua == ub)).astype(np.int8)
    f["a_empty_a"] = A.a_empty.fillna(True).astype(np.int8).to_numpy()
    na, nb = _s(A.a_num), _s(B.a_num)
    lev = _pair_scores(na, nb, fuzz.ratio)
    f["num_ratio"] = lev
    f["num_has_a"] = (na != "").astype(np.int8)
    f["num_has_b"] = (nb != "").astype(np.int8)
    for k, v in f.items():
        c[k] = v
    # context on name score (rank of this pool record among the S1's candidates)
    g = c.groupby("s1", sort=False)
    c["s1_rank_ntsort"] = g.n_tsort.rank(ascending=False, method="min").astype(np.float32)
    c["s1_gap_ntsort"] = (g.n_tsort.transform("max") - c.n_tsort).astype(np.float32)
    c["s1_rank_atset"] = g.a_tset.rank(ascending=False, method="min").astype(np.float32)
    print(f"  pair features for {len(c):,} pairs ({time.time()-t0:.0f}s)", flush=True)
    return c


UNION_FEATS = [f for f in U.FEATS]
T7_FEATS = ["n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw", "n_full_ratio", "n_skel_ratio", "n_jacc",
            "n_nospace_eq", "n_first_eq", "n_acronym", "n_alt_match", "n_len_a", "n_len_b", "was_indic_b",
            "is_domain_b", "freq_pool_a", "freq_pool_b", "freq_s1_a", "a_tset", "a_tsort", "a_jacc",
            "a_street_ratio", "a_city_ratio", "a_unit_eq", "a_empty_a", "num_ratio", "num_has_a", "num_has_b",
            "s1_rank_ntsort", "s1_gap_ntsort", "s1_rank_atset"]
S1_FEATS = UNION_FEATS + T7_FEATS


def features(split, subset):
    c = pd.read_parquet(os.path.join(WORK_DIR, f"cand_{split}_{subset}.parquet"))
    c = t7_features(c, split)
    out = os.path.join(WORK_DIR, f"feat_{split}_{subset}.parquet")
    c.to_parquet(out)
    print(f"[features] {split}/{subset}: {len(c):,} pairs -> {out}", flush=True)
    return c


# ------------------------------------------------------------------ models
def _xgb_params():
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    return dict(n_estimators=600, learning_rate=0.06, max_depth=8, min_child_weight=5, subsample=0.8,
                colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist", device=dev, eval_metric="logloss",
                max_bin=256, n_jobs=os.cpu_count())


def cv_fit(df, feats, tag):
    """5-fold CV grouped by S1 -> OOF predictions; saves the fold models."""
    import xgboost as xgb
    os.makedirs(MODEL_DIR, exist_ok=True)
    fold = (pd.util.hash_array(df.s1.to_numpy()) % NFOLD).astype(np.int8)
    oof = np.zeros(len(df), np.float32)
    X = df[feats].astype(np.float32).to_numpy()
    y = df.y.to_numpy()
    for k in range(NFOLD):
        t0 = time.time()
        tr, va = fold != k, fold == k
        m = xgb.XGBClassifier(**_xgb_params(), early_stopping_rounds=40)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        m.save_model(os.path.join(MODEL_DIR, f"{tag}_fold{k}.json"))
        print(f"  [{tag}] fold {k}: best_iter={m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(MODEL_DIR, f"{tag}_feats.json"), "w") as f:
        json.dump(feats, f)
    return oof


def cv_predict(df, tag):
    import xgboost as xgb
    with open(os.path.join(MODEL_DIR, f"{tag}_feats.json")) as f:
        feats = json.load(f)
    X = df[feats].astype(np.float32).to_numpy()
    p = np.zeros(len(df), np.float32)
    for k in range(NFOLD):
        m = xgb.XGBClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"{tag}_fold{k}.json"))
        p += m.predict_proba(X)[:, 1] / NFOLD
    return p


def importance(tag, top=15):
    import xgboost as xgb
    with open(os.path.join(MODEL_DIR, f"{tag}_feats.json")) as f:
        feats = json.load(f)
    m = xgb.XGBClassifier(); m.load_model(os.path.join(MODEL_DIR, f"{tag}_fold0.json"))
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    print(f"  [{tag}] top features: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.head(top).items()))


# ------------------------------------------------------------------ decision + metric (simple rule; postprocess.py has the final one)
def decide(c, p, thr, exclusive=True):
    """keep pairs with p >= thr; with exclusivity each rid goes only to its best-scoring S1."""
    d = pd.DataFrame({"s1": c.s1.to_numpy(), "rid": c.rid.to_numpy(), "p": p})
    d = d[d.p >= thr]
    if exclusive:
        d = d.sort_values("p", ascending=False).drop_duplicates("rid")
    return d


def macro_f05(pred, g, s1_ids):
    s1_ids = pd.Index(np.unique(s1_ids))
    n_pred = pred.groupby("s1").size().reindex(s1_ids, fill_value=0).to_numpy()
    n_true = g.groupby("s1").size().reindex(s1_ids, fill_value=0).to_numpy()
    tp = pred.merge(g, on=["s1", "rid"]).groupby("s1").size().reindex(s1_ids, fill_value=0).to_numpy()
    import evaluation as ev
    return float(ev.f05_per_entity(n_pred, n_true, tp).mean())


def subset_ids(split, subset):
    ids = encode_id(pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s1.parquet"), columns=["entity_id"]).entity_id)
    return ids[T.subset_mask(ids, subset)]


def tune_threshold(c, p, subset):
    g = U.gt_subset(subset)
    ids = subset_ids("train", subset)
    best = (0, 0.5)
    for thr in np.arange(0.2, 0.9, 0.025):
        f = macro_f05(decide(c, p, thr), g, ids)
        if f > best[0]:
            best = (f, float(thr))
    return best


def report(c, p, thr, subset, label):
    g = U.gt_subset(subset)
    ids = subset_ids("train", subset)
    f_ex = macro_f05(decide(c, p, thr, True), g, ids)
    f_no = macro_f05(decide(c, p, thr, False), g, ids)
    print(f"[{label}] {subset} macro F0.5 @thr={thr:.3f}: {f_ex:.4f} (with rid exclusivity) / {f_no:.4f} (without)",
          flush=True)
    return f_ex


# ------------------------------------------------------------------ group consistency
def t10_features(c: pd.DataFrame, split: str, pcol="p1") -> pd.DataFrame:
    """Group consistency: every candidate vs the S1's most confident match (the anchor; for the anchor
    itself, the runner-up), plus score context and reciprocal-best signals from the stage-1 score."""
    from rapidfuzz import fuzz
    t0 = time.time()
    s1, pool, _, _ = load_norm(split)
    c = c.sort_values(["s1", pcol], ascending=[True, False]).reset_index(drop=True)
    g = c.groupby("s1", sort=False)
    rank = g.cumcount().to_numpy()
    c["p_rank"] = rank.astype(np.int16)
    c["p_max"] = g[pcol].transform("max").astype(np.float32)
    c["p_gap"] = (c.p_max - c[pcol]).astype(np.float32)
    c["p_sum"] = g[pcol].transform("sum").astype(np.float32)
    c["p_n50"] = (c[pcol] >= 0.5).groupby(c.s1, sort=False).transform("sum").astype(np.int16)
    sec_val = pd.Series(np.where(rank == 1, c[pcol].to_numpy(), 0.0)).groupby(c.s1.to_numpy()).transform("max")
    second = sec_val.to_numpy(np.float32)
    c["p_second"] = second
    # anchor rid per S1 (top) and runner-up
    first_idx = np.flatnonzero(rank == 0)
    top_rid = pd.Series(c.rid.to_numpy()[first_idx], index=c.s1.to_numpy()[first_idx])
    sec_idx = np.flatnonzero(rank == 1)
    sec_rid = pd.Series(c.rid.to_numpy()[sec_idx], index=c.s1.to_numpy()[sec_idx])
    ref = top_rid.reindex(c.s1.to_numpy()).to_numpy()
    ref_alt = sec_rid.reindex(c.s1.to_numpy()).to_numpy()
    ref = np.where(rank == 0, ref_alt, ref)             # anchor compares to the runner-up
    has_ref = ~np.isnan(ref.astype(float))
    ref_i = np.where(has_ref, ref, c.rid.to_numpy()).astype(np.int64)
    R = pool.reindex(ref_i); B = pool.reindex(c.rid.to_numpy())
    rn, bn = _s(R.n_core), _s(B.n_core)
    ra, ba = _s(R.a_full), _s(B.a_full)
    gn = _pair_scores(rn, bn, fuzz.token_sort_ratio)
    ga = _pair_scores(ra, ba, fuzz.token_set_ratio)
    rnum, bnum = _s(R.a_num), _s(B.a_num)
    gnum = np.where((rnum == "") | (bnum == ""), -1, rnum == bnum).astype(np.int8)
    c["g_name"] = np.where(has_ref, gn, -2).astype(np.float32)
    c["g_addr"] = np.where(has_ref, ga, -2).astype(np.float32)
    c["g_num_eq"] = np.where(has_ref, gnum, -2).astype(np.int8)
    c["g_same_src"] = np.where(has_ref, (ref_i // 10 ** 12) == (c.rid.to_numpy() // 10 ** 12), -1).astype(np.int8)
    c["g_ref_p"] = np.where(rank == 0, second, c.p_max).astype(np.float32)
    # reciprocal best: is this S1 the rid's best S1 (by stage-1 score), and the best competing score
    gr = c.groupby("rid", sort=False)[pcol]
    rid_max = gr.transform("max").to_numpy()
    c["r_is_best"] = (c[pcol].to_numpy() >= rid_max).astype(np.int8)
    rid_n = gr.transform("size").to_numpy()
    # best score among OTHER S1s for this rid
    srt = c[["rid", pcol]].sort_values(["rid", pcol], ascending=[True, False])
    rr = srt.groupby("rid", sort=False).cumcount().to_numpy()
    top2 = pd.Series(srt[pcol].to_numpy()[rr == 1], index=srt.rid.to_numpy()[rr == 1])
    other = np.where(c[pcol].to_numpy() >= rid_max, c.rid.map(top2).fillna(0).to_numpy(), rid_max)
    c["r_other_best"] = np.where(rid_n > 1, other, 0).astype(np.float32)
    c["r_recip"] = ((rank == 0) & (c.r_is_best == 1)).astype(np.int8)
    print(f"  group-consistency features ({time.time()-t0:.0f}s)", flush=True)
    return c


T10_FEATS = ["p1", "p_rank", "p_max", "p_gap", "p_sum", "p_n50", "p_second", "g_name", "g_addr", "g_num_eq",
             "g_same_src", "g_ref_p", "r_is_best", "r_other_best", "r_recip"]
S2_FEATS = S1_FEATS + T10_FEATS


# ------------------------------------------------------------------ drivers
def stage1():
    tr = pd.read_parquet(os.path.join(WORK_DIR, "feat_train_m1.parquet"))
    ho = pd.read_parquet(os.path.join(WORK_DIR, "feat_train_holdout.parquet"))
    print(f"[stage1] train {len(tr):,} pairs ({tr.y.mean():.3f} pos), holdout {len(ho):,}", flush=True)
    tr["p1"] = cv_fit(tr, S1_FEATS, "s1")
    importance("s1")
    ho["p1"] = cv_predict(ho, "s1")
    f, thr = tune_threshold(tr, tr.p1.to_numpy(), "m1")
    print(f"[stage1] m1 OOF best F0.5={f:.4f} at thr={thr:.3f}")
    report(ho, ho.p1.to_numpy(), thr, "holdout", "stage1")
    tr[["s1", "rid", "p1"]].to_parquet(os.path.join(WORK_DIR, "p1_train_m1.parquet"))
    ho[["s1", "rid", "p1"]].to_parquet(os.path.join(WORK_DIR, "p1_train_holdout.parquet"))
    json.dump({"thr1": thr}, open(os.path.join(MODEL_DIR, "thresholds_s1.json"), "w"))


def _with_p1(split, subset):
    c = pd.read_parquet(os.path.join(WORK_DIR, f"feat_{split}_{subset}.parquet"))
    p = pd.read_parquet(os.path.join(WORK_DIR, f"p1_{split}_{subset}.parquet"))
    return c.merge(p, on=["s1", "rid"], how="left")


def stage2():
    tr = t10_features(_with_p1("train", "m1"), "train")
    ho = t10_features(_with_p1("train", "holdout"), "train")
    tr["p2"] = cv_fit(tr, S2_FEATS, "s2")
    importance("s2")
    ho["p2"] = cv_predict(ho, "s2")
    f, thr = tune_threshold(tr, tr.p2.to_numpy(), "m1")
    print(f"[stage2] m1 OOF best F0.5={f:.4f} at thr={thr:.3f}")
    fh = report(ho, ho.p2.to_numpy(), thr, "holdout", "stage2")
    ho[["s1", "rid", "y", "p1", "p2"]].to_parquet(os.path.join(WORK_DIR, "scores_train_holdout.parquet"))
    tr[["s1", "rid", "y", "p1", "p2"]].to_parquet(os.path.join(WORK_DIR, "scores_train_m1_oof.parquet"))
    json.dump({"thr2": thr, "holdout_f05": fh}, open(os.path.join(MODEL_DIR, "thresholds_s2.json"), "w"))


def score_test(parts=range(10)):
    out = []
    for k in parts:
        t0 = time.time()
        c = pd.read_parquet(os.path.join(WORK_DIR, f"cand_test_m{k}.parquet"))
        c = t7_features(c, "test")
        c["p1"] = cv_predict(c, "s1")
        c = t10_features(c, "test")
        c["p2"] = cv_predict(c, "s2")
        out.append(c[["s1", "rid", "p1", "p2"]])
        print(f"[test m{k}] {len(c):,} pairs scored ({time.time()-t0:.0f}s)", flush=True)
    s = pd.concat(out, ignore_index=True)
    s.to_parquet(os.path.join(WORK_DIR, "scores_test.parquet"))
    print(f"[test] {len(s):,} scored pairs -> work/scores_test.parquet")


def _ready(path, settle=30):
    """file exists, has not been modified for `settle` s and is a complete parquet file."""
    import pyarrow.parquet as pq
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) < settle:
        return False
    try:
        pq.ParquetFile(path).metadata
        return True
    except Exception:
        return False


def score_test_incremental(parts=range(10), poll=30):
    """Score each test part as soon as its trimmed candidates exist (overlaps with candidate generation)."""
    todo = list(parts)
    while todo:
        for k in list(todo):
            if not _ready(os.path.join(WORK_DIR, f"cand_test_m{k}.parquet")):
                continue
            t0 = time.time()
            c = pd.read_parquet(os.path.join(WORK_DIR, f"cand_test_m{k}.parquet"))
            c = t7_features(c, "test")
            c["p1"] = cv_predict(c, "s1")
            c = t10_features(c, "test")
            c["p2"] = cv_predict(c, "s2")
            c[["s1", "rid", "p1", "p2"]].to_parquet(os.path.join(WORK_DIR, f"scores_test_m{k}.parquet"))
            print(f"[test m{k}] {len(c):,} pairs scored ({time.time()-t0:.0f}s)", flush=True)
            todo.remove(k)
        if todo:
            time.sleep(poll)
    s = pd.concat([pd.read_parquet(os.path.join(WORK_DIR, f"scores_test_m{k}.parquet")) for k in parts],
                  ignore_index=True)
    s.to_parquet(os.path.join(WORK_DIR, "scores_test.parquet"))
    print(f"[test] {len(s):,} scored pairs, {s.s1.nunique():,} S1 -> work/scores_test.parquet", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "features":
        features(sys.argv[2], sys.argv[3])
    elif cmd == "stage1":
        stage1()
    elif cmd == "stage2":
        stage2()
    elif cmd == "score_test_inc":
        score_test_incremental()
    elif cmd == "score_test":
        score_test([int(x) for x in sys.argv[2:]] or range(10))
