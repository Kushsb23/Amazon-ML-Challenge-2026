"""Sanity check: test scores vs holdout (coverage, score distribution, predicted-match rate)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from blocking_keys import encode_id

thr = json.load(open(os.path.join(WORK_DIR, "models", "thresholds_s2.json")))["thr2"]
ho = pd.read_parquet(os.path.join(WORK_DIR, "scores_train_holdout.parquet"))
te = pd.read_parquet(os.path.join(WORK_DIR, "scores_test.parquet"))
ids = encode_id(pd.read_parquet(os.path.join(WORK_DIR, "norm_test_s1.parquet"), columns=["entity_id"]).entity_id)
print(f"threshold {thr}")
print(f"test S1 total {len(ids):,}; with candidates {te.s1.nunique():,} ({te.s1.nunique()/len(ids):.4f}); pairs {len(te):,}")
print(f"duplicate (s1,rid) pairs in test scores: {te.duplicated(['s1','rid']).sum()}")
for name, d, n in (("holdout", ho, 220845), ("test", te, len(ids))):
    q = d.p2.quantile([.5, .9, .99]).round(4).tolist()
    sel = d[d.p2 >= thr].sort_values("p2", ascending=False).drop_duplicates("rid")
    per = sel.groupby("s1").size()
    print(f"{name:8s} p2 quantiles(50/90/99)={q}  predicted pairs/S1={len(sel)/n:.3f}  "
          f"S1 with >=1 match={len(per)/n:.4f}  mean matches|>0={per.mean():.2f}")
