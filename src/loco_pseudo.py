"""Self-training for an unseen country, validated leave-one-country-out.

source country: labelled (m1 slice); target country: its m1 rows are used WITHOUT labels.
Round r: model -> scores target -> confident pairs become pseudo-labels (positive: p >= hi and the pair is the
record's best S1; negative: p <= lo) -> retrain on source labels + target pseudo-labels.
Evaluated on the target country's holdout with its true labels (never used for training).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BER_VARIANT", "nc")
import loco as L
import postprocess as PP
import matcher as M


def pseudo(t, p, hi, lo):
    d = pd.DataFrame({"s1": t.s1.to_numpy(), "rid": t.rid.to_numpy(), "p": p})
    best_of_rid = d.groupby("rid").p.transform("max").to_numpy()
    pos = (p >= hi) & (p >= best_of_rid)
    neg = p <= lo
    keep = pos | neg
    out = t[keep].copy()
    out["y"] = pos[keep].astype(np.int8)
    return out, keep, pos


def run(src, tgt, hi, lo, rounds, tr, ho, ids_all, ctry, nt_all, weight=1.0):
    S = tr[tr.country == L.CODE[src]]
    Tu = tr[tr.country == L.CODE[tgt]].reset_index(drop=True)          # labels ignored
    H = ho[ho.country == L.CODE[tgt]].reset_index(drop=True)
    ids = ids_all[ctry.reindex(ids_all).to_numpy() == tgt]
    nt = nt_all[nt_all.index.isin(ids)]
    m = L.fit(S)
    X = lambda df: df[L.FEATS].astype(np.float32).to_numpy()
    g = L.f05_grid(H, m.predict_proba(X(H))[:, 1], ids, ctry, nt)
    print(f"[{src}->{tgt}] round 0 (source only): @0.725 {g[0.725][0]:.4f}  best {max(v[0] for v in g.values()):.4f}",
          flush=True)
    for r in range(1, rounds + 1):
        p = m.predict_proba(X(Tu))[:, 1]
        P_, keep, pos = pseudo(Tu, p, hi, lo)
        true = Tu.y.to_numpy()
        prec = true[pos].mean() if pos.any() else float("nan")
        print(f"   round {r}: pseudo pos {pos.sum():,} (true-label precision {prec:.4f}, "
              f"covers {true[pos].sum() / max(1, true.sum()):.3f} of target positives), "
              f"neg {(keep & ~pos).sum():,}, dropped {(~keep).sum():,}", flush=True)
        both = pd.concat([S, P_], ignore_index=True)
        import xgboost as xgb
        prm = M._xgb_params()
        m = xgb.XGBClassifier(**prm)
        w = np.r_[np.ones(len(S)), np.full(len(P_), weight)]
        m.fit(X(both), both.y.to_numpy(), sample_weight=w, verbose=False)
        g = L.f05_grid(H, m.predict_proba(X(H))[:, 1], ids, ctry, nt)
        bt = max(g, key=lambda t: g[t][0])
        print(f"[{src}->{tgt}] round {r} hi={hi} lo={lo}: @0.725 {g[0.725][0]:.4f} (P {g[0.725][1]:.4f} "
              f"R {g[0.725][2]:.4f})  best {g[bt][0]:.4f} @thr {bt}", flush=True)
    return g


if __name__ == "__main__":
    tr = pd.read_parquet(os.path.join(M.WORK_DIR, "feat_train_m1.parquet"))
    ho = pd.read_parquet(os.path.join(M.WORK_DIR, "feat_train_holdout.parquet"))
    ids_all, ctry, nt_all = PP._universe("holdout")
    for src, tgt in (("US", "India"), ("India", "US")):
        for hi, lo in ((0.9, 0.1), (0.97, 0.03)):
            run(src, tgt, hi, lo, 2, tr, ho, ids_all, ctry, nt_all)
