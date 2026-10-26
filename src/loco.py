"""Leave-one-country-out check: how does the stage-1 matcher behave on a country it never saw (proxy for France)?

Trains on one training country of the m1 slice, scores the other country on the holdout, and compares with a
model trained in-country. Reports holdout macro F0.5 over a threshold grid (with record exclusivity).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BER_VARIANT", "nc")
import matcher as M
import union_trim as U
import postprocess as PP

FEATS = [f for f in M.S1_FEATS if f != "country"]
CODE = {"US": 0, "India": 1}


def fit(df):
    import xgboost as xgb
    p = M._xgb_params(); p["n_estimators"] = 600
    m = xgb.XGBClassifier(**p)
    m.fit(df[FEATS].astype(np.float32).to_numpy(), df.y.to_numpy(), verbose=False)
    return m


def f05_grid(ho, p, ids, ctry, nt):
    d = ho[["s1", "rid", "y"]].copy(); d["p2"] = p
    out = {}
    for thr in (0.5, 0.6, 0.65, 0.7, 0.725, 0.75, 0.8, 0.85, 0.9, 0.95):
        k = PP.apply_rule(d, {"thr": float(thr), "fb": 1.1, "caps": True}, ctry)
        f, P, R, *_ = PP.macro_f05(d, k, nt, ids)
        out[thr] = (f, P, R)
    return out


if __name__ == "__main__":
    tr = pd.read_parquet(os.path.join(M.WORK_DIR, "feat_train_m1.parquet"))
    ho = pd.read_parquet(os.path.join(M.WORK_DIR, "feat_train_holdout.parquet"))
    ids_all, ctry, nt_all = PP._universe("holdout")
    res = {}
    for src in ("US", "India"):
        model_src = fit(tr[tr.country == CODE[src]])
        for tgt in ("US", "India"):
            h = ho[ho.country == CODE[tgt]].reset_index(drop=True)
            ids = ids_all[ctry.reindex(ids_all).to_numpy() == tgt]
            nt = nt_all[nt_all.index.isin(ids)]
            p = model_src.predict_proba(h[FEATS].astype(np.float32).to_numpy())[:, 1]
            g = f05_grid(h, p, ids, ctry, nt)
            best = max(g, key=lambda t: g[t][0])
            tag = f"train {src:5s} -> test {tgt:5s} ({'same' if src == tgt else 'UNSEEN'})"
            res[tag] = g
            print(f"{tag}: best thr {best} F0.5 {g[best][0]:.4f} | @0.725 {g[0.725][0]:.4f} "
                  f"(P {g[0.725][1]:.4f} R {g[0.725][2]:.4f})", flush=True)
            print("    " + "  ".join(f"{t}:{v[0]:.4f}" for t, v in g.items()), flush=True)
    json.dump({k: {str(t): v for t, v in g.items()} for k, g in res.items()},
              open(os.path.join(M.WORK_DIR, "models", "loco.json"), "w"), indent=1)
