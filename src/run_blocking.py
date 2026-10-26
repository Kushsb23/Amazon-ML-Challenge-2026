"""Key-blocking runner - key-based candidate generation.

    python run_blocking.py --split train --subset holdout   # dev: 220k holdout S1 -> recall report
    python run_blocking.py --split train --subset m1        # trim-model training subset (id % 10 == 1)
    python run_blocking.py --split train --subset all       # training candidates for the matcher 
    python run_blocking.py --split test  --subset all       # test candidates for the submission

Writes WORK_DIR/cand_keys_{split}_{subset}.parquet  (s1 int64, rid int64, flags uint16).
The first run per split builds and caches pool keys in WORK_DIR/poolkeys_{split}/ (~5 min);
delete that folder if you change normalize.py or the CAPS / key definitions.
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev
from config import WORK_DIR
from blocking_keys import generate, METHODS


def per_method(c, g):
    allhit = g.merge(c, on=["s1", "rid"], how="left")
    fl = allhit["flags"].fillna(0).astype(np.int64).to_numpy()
    rows = {}
    for flag, name in METHODS.items():
        m = (c["flags"].to_numpy() & flag) > 0
        rows[name] = {"pairs": int(m.sum()), "recall": ((fl & flag) > 0).mean(), "unique_recall": (fl == flag).mean()}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--subset", default="holdout", choices=["holdout", "m1", "all"])
    a = ap.parse_args()
    t0 = time.time()
    from tfidf_knn import subset_mask
    from blocking_keys import encode_id
    filt = None if a.subset == "all" else (lambda ids: subset_mask(encode_id(ids), a.subset))
    c, _ = generate(a.split, s1_filter=filt)
    out = os.path.join(WORK_DIR, f"cand_keys_{a.split}_{a.subset}.parquet")
    c.to_parquet(out)
    print(f"saved {len(c):,} pairs -> {out}")
    if a.split == "train":
        g = ev.gt_int(holdout_only=False)
        g = g[subset_mask(g.s1.to_numpy(), a.subset)]
        ids1 = encode_id(pd.read_parquet(os.path.join(WORK_DIR, "norm_train_s1.parquet"), columns=["entity_id"]).entity_id)
        n_s1 = int(subset_mask(ids1, a.subset).sum())
        pm = per_method(c, g)
        pm["avg_per_s1"] = pm.pairs / n_s1
        print(pm.round(4).to_string())
        print({k: round(v, 4) for k, v in ev.recall_report_int(c, g, n_s1).items()})
    print(f"done in {time.time()-t0:.0f}s")
