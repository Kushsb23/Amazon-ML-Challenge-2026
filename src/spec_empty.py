"""Specialist model for pairs whose record has NO address (80% of our missed true pairs).

Training: every empty-address candidate pair of train slices m1..m5 (feat_train_*.parquet: union + T7 matcher features),
+ name-focused features (name_feats.py) + our stage-2 OOF probability + cross-encoder logits where the pair was in the
band. LightGBM, 5-fold grouped by S1 -> out-of-fold scores for m1..m5 (the blender trains on m1-m2 OOF), fold-average
for holdout and test.

    set BER_V3=1 & python spec_empty.py     -> work/spec_{oof,holdout,test}.parquet  (s1, rid, p_spec)
"""
import os
import sys
import gc
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
import matcher as M
import name_feats as NF

TR_SUBS = ["m1", "m2", "m3", "m4", "m5"]
CE_DIR = os.path.join(WORK_DIR, "ce4")
TEST_CAND = os.environ.get("BER_SPEC_TEST_CAND", os.path.join(WORK_DIR, "ce4_base"))
NF_ = 5


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def extras(d, stage2_file, ce_files):
    s2 = pd.read_parquet(os.path.join(WORK_DIR, stage2_file))
    s2 = s2[["s1", "rid"] + [c for c in ("p1", "p2_lgb") if c in s2]].drop_duplicates(["s1", "rid"])
    d = d.merge(s2.rename(columns={"p1": "o_p1", "p2_lgb": "o_p2"}), on=["s1", "rid"], how="left")
    ce = pd.concat([pd.read_parquet(os.path.join(CE_DIR, f), columns=["s1", "rid", "ce", "cel"]) for f in ce_files])
    d = d.merge(ce.drop_duplicates(["s1", "rid"]).rename(columns={"ce": "o_ce", "cel": "o_cel"}), on=["s1", "rid"], how="left")
    g = d.groupby("s1", sort=False).o_p2
    d["o_p2_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    d["o_p2_gap"] = (g.transform("max") - d.o_p2).astype(np.float32)
    return d


def main():
    import lightgbm as lgb
    t0 = time.time()
    feats = [f for f in M.S1_FEATS] + NF.NFEATS + ["o_p1", "o_p2", "o_ce", "o_cel", "o_p2_rank", "o_p2_gap"]
    # ---------------- train (m1..m5)
    parts = []
    for sb in TR_SUBS:
        c = pd.read_parquet(os.path.join(WORK_DIR, f"feat_train_{sb}.parquet"))
        c = c[c.b_empty == 1].reset_index(drop=True)
        parts.append(c)
    tr = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    tr = extras(tr, "scores_train_oof_b.parquet", ["ce_stack.parquet"])
    tr["p_avg"] = tr.o_p2.fillna(0)
    f = NF.compute(tr, "train", "spec_tr", "p_avg")
    tr = tr.merge(f[["s1", "rid"] + NF.NFEATS], on=["s1", "rid"], how="left")
    feats = [x for x in feats if x in tr]
    log(f"train: {len(tr):,} empty-address pairs from {TR_SUBS} (pos {tr.y.mean():.3f}), {len(feats)} features")
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
    X = tr[feats].to_numpy(np.float32); y = tr.y.to_numpy()
    fold = (pd.util.hash_array(tr.s1.to_numpy()) % NF_).astype(np.int8)
    oof = np.zeros(len(tr)); models = []
    from sklearn.metrics import roc_auc_score
    for k in range(NF_):
        m = lgb.train(prm, lgb.Dataset(X[fold != k], y[fold != k]), 700)
        oof[fold == k] = m.predict(X[fold == k]); models.append(m)
        log(f"  fold {k}: AUC {roc_auc_score(y[fold == k], oof[fold == k]):.5f}")
    base = tr.o_p2.fillna(0).to_numpy()
    log(f"OOF AUC specialist {roc_auc_score(y, oof):.5f} vs our stage-2 {roc_auc_score(y, base):.5f}")
    pd.DataFrame({"s1": tr.s1.to_numpy(), "rid": tr.rid.to_numpy(), "p_spec": oof.astype(np.float32)}) \
        .to_parquet(os.path.join(WORK_DIR, "spec_oof.parquet"))
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda t: -t[1])
    log("gain: " + ", ".join(f"{a}={b/sum(v for _, v in imp):.3f}" for a, b in imp[:10]))
    del tr, X; gc.collect()

    def predict(d):
        Xd = d[feats].to_numpy(np.float32)
        return sum(m.predict(Xd) for m in models) / len(models)

    # ---------------- holdout
    ho = pd.read_parquet(os.path.join(WORK_DIR, "feat_train_holdout.parquet"))
    ho = ho[ho.b_empty == 1].reset_index(drop=True)
    ho = extras(ho, "scores_train_holdout_b.parquet", ["ce_holdout.parquet"])
    ho["p_avg"] = ho.o_p2.fillna(0)
    f = NF.compute(ho, "train", "spec_ho", "p_avg")
    ho = ho.merge(f[["s1", "rid"] + NF.NFEATS], on=["s1", "rid"], how="left")
    ho["p_spec"] = predict(ho)
    log(f"holdout AUC specialist {roc_auc_score(ho.y, ho.p_spec):.5f} vs stage-2 {roc_auc_score(ho.y, ho.o_p2.fillna(0)):.5f}")
    ho[["s1", "rid", "p_spec"]].to_parquet(os.path.join(WORK_DIR, "spec_holdout.parquet"))
    del ho; gc.collect()
    # ---------------- test (matcher features computed per slice, then filtered)
    parts = []
    for k in range(10):
        t1 = time.time()
        c = pd.read_parquet(os.path.join(TEST_CAND, f"cand_test_m{k}.parquet"))
        c = M.t7_features(c, "test")
        c = c[c.b_empty == 1].reset_index(drop=True)
        parts.append(c)
        log(f"  test m{k}: {len(c):,} empty-address pairs ({time.time()-t1:.0f}s)")
    te = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    te = extras(te, "scores_test_b.parquet", ["ce_test.parquet"])
    te["p_avg"] = te.o_p2.fillna(0)
    f = NF.compute(te, "test", "spec_te", "p_avg")
    te = te.merge(f[["s1", "rid"] + NF.NFEATS], on=["s1", "rid"], how="left")
    te["p_spec"] = predict(te)
    te[["s1", "rid", "p_spec"]].to_parquet(os.path.join(WORK_DIR, "spec_test.parquet"))
    log(f"specialist done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
