"""Architecture-v3 cheap pruning: LightGBM on retrieval-time signals only (no string features).

Trained on slice m1 (labels from the training ground truth), then applied to every slice:
keep pairs with c >= threshold. The threshold is the largest value whose holdout pair recall stays within
MAX_LOSS of the unpruned union (the architecture used c >= 0.005, ~6 candidates / S1).

    python prune.py fit        -> work/v3/prune.txt, prune.json (threshold, recall table)
    python prune.py apply      -> work/v3/cand_{split}_{subset}.parquet (signals + prune_c)
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
import union_trim as U

VDIR = os.path.join(WORK_DIR, "v3")
MAX_LOSS = float(os.environ.get("BER_PRUNE_MAX_LOSS", 0.0005))
SUBSETS = {"train": ["m1", "m2", "m3", "m4", "m5", "holdout"], "test": [f"m{k}" for k in range(10)]}
SIG = (["rank_fn", "rank_fa", "rank_ff", "rank_bn", "rank_ba", "rank_bf",
        "score_fn", "score_fa", "score_ff", "score_bn", "score_ba", "score_bf",
        "r_name", "r_addr", "r_rec", "h_emb", "h_key", "h_tf", "hits", "min_fr", "min_br",
        "cos_n", "cos_a", "cos_f", "csls_n", "csls_a", "csls_f", "t_has_addr", "s_has_addr", "src3",
        "s_n", "s_rank_f", "s_mg_f", "s_rank_n", "s_mg_n", "t_n", "t_rank_f", "t_mg_f", "t_rank_n", "t_mg_n"]
       + [f"k{f}" for f in U.KEY_FLAGS] + [f"t{f}" for f in U.TF_FLAGS])


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def frame(split, sb):
    c = pd.read_parquet(os.path.join(VDIR, f"sig_{split}_{sb}.parquet"))
    kf = c.kflags.to_numpy(); tf = c.tflags.to_numpy()
    for f in U.KEY_FLAGS:
        c[f"k{f}"] = ((kf & f) > 0).astype(np.int8)
    for f in U.TF_FLAGS:
        c[f"t{f}"] = ((tf & f) > 0).astype(np.int8)
    if split == "train":
        g = U.gt_subset(sb)[["s1", "rid"]].drop_duplicates().assign(y=np.int8(1))
        c = c.merge(g, on=["s1", "rid"], how="left")
        c["y"] = c.y.fillna(0).astype(np.int8)
    return c


def fit():
    import lightgbm as lgb
    t0 = time.time()
    tr = frame("train", "m1")
    m = lgb.train(dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
                       feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0),
                  lgb.Dataset(tr[SIG].to_numpy(np.float32), tr.y.to_numpy()), 600)
    m.save_model(os.path.join(VDIR, "prune.txt"))
    ho = frame("train", "holdout")
    ho["c"] = m.predict(ho[SIG].to_numpy(np.float32))
    g = U.gt_subset("holdout")[["s1", "rid"]].drop_duplicates()
    ns1 = U.n_s1("train", "holdout")
    base = ho.y.sum() / len(g)
    table, thr = [], None
    for t in (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05):
        k = ho.c >= t
        rec = ho.y[k].sum() / len(g)
        table.append({"thr": t, "per_s1": float(k.sum() / ns1), "recall": float(rec)})
        log(f"  c >= {t:<7} {k.sum()/ns1:6.2f} cands/S1  pair recall {rec:.4f}")
        if base - rec <= MAX_LOSS:
            thr = t
    json.dump({"threshold": thr, "union_recall": float(base), "table": table}, open(os.path.join(VDIR, "prune.json"), "w"), indent=1)
    log(f"prune model fitted ({time.time()-t0:.0f}s); union recall {base:.4f}; threshold {thr}")


def apply():
    import lightgbm as lgb
    m = lgb.Booster(model_file=os.path.join(VDIR, "prune.txt"))
    thr = json.load(open(os.path.join(VDIR, "prune.json")))["threshold"]
    for split, subs in SUBSETS.items():
        for sb in subs:
            t0 = time.time()
            c = frame(split, sb)
            c["prune_c"] = m.predict(c[SIG].to_numpy(np.float32)).astype(np.float32)
            n0 = len(c)
            c = c[c.prune_c >= thr].reset_index(drop=True)
            c.drop(columns=["y"], errors="ignore").to_parquet(os.path.join(VDIR, f"cand_{split}_{sb}.parquet"))
            log(f"{split}/{sb}: {n0:,} -> {len(c):,} pairs ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    {"fit": fit, "apply": apply}[sys.argv[1]]()
