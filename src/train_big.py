"""Matcher trained on several train slices + stage-2 ensemble (XGBoost, LightGBM, CatBoost).

    set BER_TRAIN_SUBSETS=m1,m2,m3,m4,m5      (default)
    python train_big.py stage1        # stage-1 XGBoost, 5-fold CV grouped by S1 on all slices
    python train_big.py stage2        # group-consistency features per slice + 3 stage-2 models, averaged
    python train_big.py score_test    # test parts m0..m9 -> work/scores_test_b.parquet

Outputs (tag 'b'): work/models/{s1b,s2b,s2b_lgb,s2b_cat}_fold*.*, work/scores_train_oof_b.parquet,
work/scores_train_holdout_b.parquet, work/scores_test_b.parquet (columns s1, rid, p1, p2_xgb, p2_lgb, p2_cat, p2).
Group-consistency / reciprocal-best features are computed per slice, exactly like on test (one tenth at a time).
"""
import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matcher as M
from config import WORK_DIR

SUBSETS = os.environ.get("BER_TRAIN_SUBSETS", "m1,m2,m3,m4,m5").split(",")
MD = M.MODEL_DIR
NF = M.NFOLD


def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_feat(sub, cols):
    c = pd.read_parquet(os.path.join(WORK_DIR, f"feat_train_{sub}.parquet"), columns=cols)
    for k in c.columns:
        if c[k].dtype == np.float64:
            c[k] = c[k].astype(np.float32)
    return c


def _folds(s1):
    return (pd.util.hash_array(np.asarray(s1)) % NF).astype(np.int8)


# ------------------------------------------------------------------ stage 1
def stage1():
    t0 = time.time()
    cols = ["s1", "rid", "y"] + M.S1_FEATS
    tr = pd.concat([_load_feat(s, cols) for s in SUBSETS], ignore_index=True)
    print(f"[stage1b] {len(tr):,} pairs from {SUBSETS} ({tr.y.mean():.3f} pos)", flush=True)
    tr["p1"] = M.cv_fit(tr, M.S1_FEATS, "s1b")
    M.importance("s1b")
    tr[["s1", "rid", "p1"]].to_parquet(os.path.join(WORK_DIR, "p1b_train.parquet"))
    del tr; gc.collect()
    ho = _load_feat("holdout", cols)
    ho["p1"] = M.cv_predict(ho, "s1b")
    ho[["s1", "rid", "p1"]].to_parquet(os.path.join(WORK_DIR, "p1b_holdout.parquet"))
    M.report(ho, ho.p1.to_numpy(), 0.7, "holdout", "stage1b")
    print(f"[stage1b] done ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ stage 2 features
def _stage2_frame(sub, p1):
    c = pd.read_parquet(os.path.join(WORK_DIR, f"feat_train_{sub}.parquet"))
    c = c.merge(p1, on=["s1", "rid"], how="left")
    c = M.t10_features(c, "train")
    keep = ["s1", "rid", "y"] + M.S2_FEATS
    c = c[keep].copy()
    for k in c.columns:
        if c[k].dtype == np.float64:
            c[k] = c[k].astype(np.float32)
    return c


def build_stage2_frames():
    p1 = pd.read_parquet(os.path.join(WORK_DIR, "p1b_train.parquet"))
    parts = []
    for s in SUBSETS:
        parts.append(_stage2_frame(s, p1))
        gc.collect()
    tr = pd.concat(parts, ignore_index=True)
    del parts, p1; gc.collect()
    ho = _stage2_frame("holdout", pd.read_parquet(os.path.join(WORK_DIR, "p1b_holdout.parquet")))
    return tr, ho


# ------------------------------------------------------------------ extra model families
def lgb_cv(tr, feats, tag):
    import lightgbm as lgb
    fold = _folds(tr.s1.to_numpy())
    X = tr[feats].to_numpy(np.float32); y = tr.y.to_numpy()
    oof = np.zeros(len(tr), np.float32)
    for k in range(NF):
        t0 = time.time()
        a, b = fold != k, fold == k
        m = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.08, num_leaves=127, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                               max_bin=255, n_jobs=os.cpu_count(), verbose=-1)
        m.fit(X[a], y[a], eval_set=[(X[b], y[b])], callbacks=[lgb.early_stopping(40, verbose=False)])
        oof[b] = m.predict_proba(X[b])[:, 1]
        m.booster_.save_model(os.path.join(MD, f"{tag}_fold{k}.txt"))
        print(f"  [{tag}] fold {k}: best_iter={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(feats, open(os.path.join(MD, f"{tag}_feats.json"), "w"))
    return oof


def lgb_predict(df, tag):
    import lightgbm as lgb
    feats = json.load(open(os.path.join(MD, f"{tag}_feats.json")))
    X = df[feats].to_numpy(np.float32)
    p = np.zeros(len(df), np.float32)
    for k in range(NF):
        p += lgb.Booster(model_file=os.path.join(MD, f"{tag}_fold{k}.txt")).predict(X) / NF
    return p


def cat_cv(tr, feats, tag):
    from catboost import CatBoostClassifier
    fold = _folds(tr.s1.to_numpy())
    X = tr[feats].to_numpy(np.float32); y = tr.y.to_numpy()
    oof = np.zeros(len(tr), np.float32)
    for k in range(NF):
        t0 = time.time()
        a, b = fold != k, fold == k
        m = CatBoostClassifier(iterations=2000, learning_rate=0.1, depth=8, loss_function="Logloss",
                               task_type="GPU" if _gpu() else "CPU", od_type="Iter", od_wait=50,
                               border_count=254, verbose=0, thread_count=os.cpu_count())
        m.fit(X[a], y[a], eval_set=(X[b], y[b]))
        oof[b] = m.predict_proba(X[b])[:, 1]
        m.save_model(os.path.join(MD, f"{tag}_fold{k}.cbm"))
        print(f"  [{tag}] fold {k}: best_iter={m.get_best_iteration()} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(feats, open(os.path.join(MD, f"{tag}_feats.json"), "w"))
    return oof


def cat_predict(df, tag):
    from catboost import CatBoostClassifier
    feats = json.load(open(os.path.join(MD, f"{tag}_feats.json")))
    X = df[feats].to_numpy(np.float32)
    p = np.zeros(len(df), np.float32)
    for k in range(NF):
        m = CatBoostClassifier(); m.load_model(os.path.join(MD, f"{tag}_fold{k}.cbm"))
        p += m.predict_proba(X)[:, 1] / NF
    return p


FAMILIES = {"xgb": (lambda tr: M.cv_fit(tr, M.S2_FEATS, "s2b"), lambda df: M.cv_predict(df, "s2b")),
            "lgb": (lambda tr: lgb_cv(tr, M.S2_FEATS, "s2b_lgb"), lambda df: lgb_predict(df, "s2b_lgb")),
            "cat": (lambda tr: cat_cv(tr, M.S2_FEATS, "s2b_cat"), lambda df: cat_predict(df, "s2b_cat"))}


def stage2(families=("xgb", "lgb", "cat")):
    t0 = time.time()
    tr, ho = build_stage2_frames()
    print(f"[stage2b] train {len(tr):,} pairs, holdout {len(ho):,} ({time.time()-t0:.0f}s)", flush=True)
    done = []
    for fam in families:
        try:
            tr[f"p2_{fam}"] = FAMILIES[fam][0](tr)
            ho[f"p2_{fam}"] = FAMILIES[fam][1](ho)
            M.report(ho, ho[f"p2_{fam}"].to_numpy(), 0.7, "holdout", f"stage2b {fam}")
            done.append(fam)
        except Exception as e:                     # a missing library must not kill the whole run
            print(f"[stage2b] {fam} skipped: {type(e).__name__}: {e}", flush=True)
    for d in (tr, ho):
        d["p2"] = d[[f"p2_{f}" for f in done]].mean(axis=1).astype(np.float32)
    M.report(ho, ho.p2.to_numpy(), 0.7, "holdout", f"stage2b mean({'+'.join(done)})")
    cols = ["s1", "rid", "y", "p1"] + [f"p2_{f}" for f in done] + ["p2"]
    tr[cols].to_parquet(os.path.join(WORK_DIR, "scores_train_oof_b.parquet"))
    ho[cols].to_parquet(os.path.join(WORK_DIR, "scores_train_holdout_b.parquet"))
    json.dump({"families": done, "subsets": SUBSETS}, open(os.path.join(MD, "stage2b.json"), "w"))
    print(f"[stage2b] done ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ test
def score_test(parts=range(10)):
    fams = json.load(open(os.path.join(MD, "stage2b.json")))["families"]
    out = []
    for k in parts:
        t0 = time.time()
        c = pd.read_parquet(os.path.join(WORK_DIR, f"cand_test_m{k}.parquet"))
        c = M.t7_features(c, "test")
        c["p1"] = M.cv_predict(c, "s1b")
        c = M.t10_features(c, "test")
        for f in fams:
            c[f"p2_{f}"] = FAMILIES[f][1](c)
        c["p2"] = c[[f"p2_{f}" for f in fams]].mean(axis=1).astype(np.float32)
        s = c[["s1", "rid", "p1"] + [f"p2_{f}" for f in fams] + ["p2"]]
        s.to_parquet(os.path.join(WORK_DIR, f"scores_test_b_m{k}.parquet"))
        out.append(s)
        print(f"[test m{k}] {len(c):,} pairs scored ({time.time()-t0:.0f}s)", flush=True)
        del c; gc.collect()
    s = pd.concat(out, ignore_index=True)
    s.to_parquet(os.path.join(WORK_DIR, "scores_test_b.parquet"))
    print(f"[test] {len(s):,} scored pairs, {s.s1.nunique():,} S1 -> work/scores_test_b.parquet", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage1":
        stage1()
    elif cmd == "stage2":
        stage2(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("xgb", "lgb", "cat"))
    elif cmd == "score_test":
        score_test([int(x) for x in sys.argv[2:]] or range(10))
