"""Pick stage-2 blend weights (XGBoost / LightGBM / CatBoost) by macro F0.5 on the out-of-fold scores,
report on the holdout, save work/models/blend_b.json (postprocess.py then uses it for tuning and writing)."""
import os, sys, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BER_SCORES", "b")
import postprocess as PP
from config import WORK_DIR

fams = json.load(open(os.path.join(WORK_DIR, "models", "stage2b.json")))["families"]
oof = pd.read_parquet(os.path.join(WORK_DIR, PP.OOF_FILE))
ho = pd.read_parquet(os.path.join(WORK_DIR, PP.HO_FILE))
ids_o, c_o, nt_o = PP._universe(PP.TUNE_SUBSETS)
oof = oof[np.isin(oof.s1.to_numpy(), ids_o)].reset_index(drop=True)
ids_h, c_h, nt_h = PP._universe("holdout")

def score(df, w, ids, c, nt, thr):
    d = df[["s1", "rid", "y"]].copy()
    d["p2"] = sum(w[f] * df[f"p2_{f}"].to_numpy() for f in fams)
    return PP.macro_f05(d, PP.apply_rule(d, {"thr": thr, "fb": 0.5, "caps": True}, c), nt, ids)[0]

grid = [dict(zip(fams, w)) for w in itertools.product(np.arange(0, 1.01, 0.25), repeat=len(fams))
        if abs(sum(w) - 1) < 1e-9]
res = []
for w in grid:
    best = max((score(oof, w, ids_o, c_o, nt_o, t), t) for t in (0.65, 0.7, 0.75))
    res.append((best[0], best[1], w))
    print(f"  w={ {k: round(v, 2) for k, v in w.items()} } OOF F0.5 {best[0]:.5f} @thr {best[1]}", flush=True)
res.sort(key=lambda r: -r[0])
top, thr, w = res[0]
eq = {f: 1 / len(fams) for f in fams}
print(f"[blend] best OOF weights {w} ({top:.5f}); holdout {score(ho, w, ids_h, c_h, nt_h, thr):.5f} "
      f"vs equal-mean holdout {score(ho, eq, ids_h, c_h, nt_h, thr):.5f} @thr {thr}")
json.dump({k: float(v) for k, v in w.items()}, open(os.path.join(WORK_DIR, "models", "blend_b.json"), "w"))
