"""Stage-2 scores with the empty-address specialist substituted: scores_*_bsp.parquet (p2_lgb = p_spec where present)."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR

for src, spec in (("scores_train_oof_b", "spec_oof"), ("scores_train_holdout_b", "spec_holdout"), ("scores_test_b", "spec_test")):
    d = pd.read_parquet(os.path.join(WORK_DIR, f"{src}.parquet"))
    sp = pd.read_parquet(os.path.join(WORK_DIR, f"{spec}.parquet")).drop_duplicates(["s1", "rid"])
    d = d.merge(sp, on=["s1", "rid"], how="left")
    m = d.p_spec.notna().to_numpy()
    d["p2_lgb"] = np.where(m, d.p_spec, d.p2_lgb).astype(np.float32)
    d.drop(columns="p_spec").to_parquet(os.path.join(WORK_DIR, src.replace("_b", "_bsp") + ".parquet"))
    print(f"{src}: {len(d):,} pairs, {int(m.sum()):,} replaced by the specialist", flush=True)
