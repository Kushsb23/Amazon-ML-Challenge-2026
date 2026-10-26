"""Exact expected-F0.5 decision per S1 (independent Bernoulli candidates), optional isotonic calibration.

For one S1 with candidate probabilities q_1 >= q_2 >= ... (after record exclusivity), predicting the top k gives
F0.5 = 1.25 TP / (0.25 N + k), with TP ~ PoissonBinomial(q_1..q_k), N = TP + R + X, R ~ PoissonBinomial(rest)
and X ~ Poisson(extra) the true matches outside the candidates. Predicting nothing scores 1 iff N = 0.
The k with the highest exact expectation is chosen (the current rule uses a ratio of expectations instead).

    set BER_SCORES=ce3 & python decision_exact.py      -> compares on m1-m2 OOF, reports holdout, and if better
                                                          writes scores_*_ce3x.parquet + models/decision_ce3x.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR

MAXK = 10          # candidates per S1 considered (sorted by q); the rest only enter R
MAXX = 4           # truncation of the Poisson for true matches outside the candidates


def _pb(q):
    """Poisson-binomial pmf for rows of q (G, m) -> (G, m+1)."""
    G, m = q.shape
    f = np.zeros((G, m + 1)); f[:, 0] = 1.0
    for j in range(m):
        qj = q[:, j:j + 1]
        f[:, 1:j + 2] = f[:, 1:j + 2] * (1 - qj) + f[:, 0:j + 1] * qj
        f[:, 0:1] = f[:, 0:1] * (1 - qj)
    return f


def efx_keep(p, s1_codes, excl, gamma=1.0, extra=0.0, floor=0.0):
    q = np.where(excl, np.clip(p, 0.0, 1.0) ** gamma, 0.0)
    idx = np.flatnonzero(excl & (q > 0))
    keep = np.zeros(len(p), bool)
    if not len(idx):
        return keep
    o = idx[np.lexsort((-q[idx], s1_codes[idx]))]
    g = s1_codes[o]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    lens = np.diff(np.r_[start, len(o)])
    rank = np.arange(len(o)) - np.repeat(start, lens)
    grp = np.repeat(np.arange(len(start)), lens)
    n = np.minimum(lens, MAXK)
    Q = np.zeros((len(start), MAXK))
    sel = rank < MAXK
    Q[grp[sel], rank[sel]] = q[o][sel]
    tail = np.zeros(len(start)); np.add.at(tail, grp[~sel], q[o][~sel])     # mass beyond MAXK -> Poisson
    lam = tail + extra
    px = np.exp(-lam)[:, None] * np.cumprod(np.c_[np.ones(len(lam)), np.repeat(lam[:, None], MAXX, 1)], 1) / \
        np.cumprod(np.r_[1, np.arange(1, MAXX + 1)])[None, :]                # (G, MAXX+1)
    best = np.zeros(len(start)); kbest = np.zeros(len(start), int)
    # k = 0: P(N = 0)
    best[:] = np.prod(1 - Q, 1) * px[:, 0]
    for k in range(1, MAXK + 1):
        ok = n >= k
        if not ok.any():
            break
        a = _pb(Q[ok, :k])                                  # TP pmf (Gk, k+1)
        r = _pb(Q[ok, k:])                                  # R pmf (Gk, MAXK-k+1)
        rx = np.zeros((ok.sum(), r.shape[1] + MAXX))        # R + X
        for x in range(MAXX + 1):
            rx[:, x:x + r.shape[1]] += r * px[ok, x:x + 1]
        A = np.arange(k + 1)[:, None]; Rr = np.arange(rx.shape[1])[None, :]
        w = 1.25 * A / (0.25 * (A + Rr) + k)               # (k+1, R)
        e = np.einsum("ga,ar,gr->g", a, w, rx)
        e[Q[ok, k - 1] < floor] = -1
        better = e > best[ok]
        ii = np.flatnonzero(ok)[better]
        best[ii] = e[better]; kbest[ii] = k
    keep[o[rank < np.repeat(kbest, lens)]] = True
    return keep


def _iso_fit(p, y):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def main():
    import postprocess as P
    tag = P.TAG
    prm0 = json.load(open(P.DEC_PATH))
    tr = P._read_scores(P.OOF_FILE)
    ids_tr, ctry_tr, nt_tr = P._universe(P.TUNE_SUBSETS)
    tr = tr[np.isin(tr.s1.to_numpy(), ids_tr)].reset_index(drop=True)
    ho = P._read_scores(P.HO_FILE)
    ids_h, ctry_h, nt_h = P._universe("holdout")
    # cross-fitted isotonic on OOF (halves by S1), full fit for holdout / test
    half = (tr.s1.to_numpy() // 7) % 2
    tr["p_iso"] = 0.0
    for h in (0, 1):
        f = _iso_fit(tr.p2.to_numpy()[half != h], tr.y.to_numpy()[half != h])
        tr.loc[half == h, "p_iso"] = f.predict(tr.p2.to_numpy()[half == h])
    iso = _iso_fit(tr.p2.to_numpy(), tr.y.to_numpy())
    ho["p_iso"] = iso.predict(ho.p2.to_numpy())

    def ev(d, prm, pc, ctry, nt, ids):
        dd = d[["s1", "rid", "y"]].assign(p2=d[pc].to_numpy())
        if prm.get("mode") == "efx":
            p = dd.p2.to_numpy().astype(np.float64); rid = dd.rid.to_numpy()
            order = np.lexsort((-p, rid)); ro = rid[order]
            excl = np.zeros(len(dd), bool); excl[order] = np.r_[True, ro[1:] != ro[:-1]]
            codes, _ = pd.factorize(dd.s1.to_numpy())
            keep = efx_keep(p, codes, excl, prm["gamma"], prm["extra"], prm["floor"])
            keep = P._caps(keep, p, rid, codes)
        else:
            keep = P.apply_rule(dd, prm, ctry)
        return P.macro_f05(dd, keep, nt, ids)[0]

    base_tr = ev(tr, prm0, "p2", ctry_tr, nt_tr, ids_tr)
    base_h = ev(ho, prm0, "p2", ctry_h, nt_h, ids_h)
    print(f"[current rule {prm0}] OOF {base_tr:.5f}  holdout {base_h:.5f}", flush=True)
    res = []
    for pc in ("p2", "p_iso"):
        for gamma in (0.8, 1.0, 1.25):
            for extra in (0.0, 0.1, 0.3):
                for floor in (0.0, 0.3):
                    prm = {"mode": "efx", "gamma": gamma, "extra": extra, "floor": floor, "caps": True}
                    f = ev(tr, prm, pc, ctry_tr, nt_tr, ids_tr)
                    res.append((f, pc, prm))
                    print(f"  efx {pc:5s} gamma {gamma} extra {extra} floor {floor}: OOF {f:.5f}", flush=True)
    f, pc, prm = max(res, key=lambda t: t[0])
    fh = ev(ho, prm, pc, ctry_h, nt_h, ids_h)
    print(f"[best exact rule] {pc} {prm}: OOF {f:.5f} (current {base_tr:.5f})  holdout {fh:.5f} (current {base_h:.5f})")
    if f <= base_tr + 1e-5:
        print("[decision_exact] no gain on OOF -> keep the current rule"); return
    new = tag + "x"
    prm["iso"] = pc == "p_iso"
    if prm["iso"]:
        prm["iso_x"] = [float(v) for v in iso.X_thresholds_]; prm["iso_y"] = [float(v) for v in iso.y_thresholds_]
    json.dump(prm, open(os.path.join(WORK_DIR, "models", f"decision_{new}.json"), "w"), indent=1)
    te = P._read_scores(P.TEST_FILE)
    for d, fn in ((tr, P.OOF_FILE), (ho, P.HO_FILE), (te, P.TEST_FILE)):
        x = d.copy()
        if prm["iso"]:
            x["p2"] = iso.predict(x.p2.to_numpy()).astype(np.float32)
        x["p2_lgb"] = x["p2"]
        cols = [c for c in ("s1", "rid", "y", "p2", "p2_lgb") if c in x]
        x[cols].to_parquet(os.path.join(WORK_DIR, fn.replace(f"_{tag}.parquet", f"_{new}.parquet")))
    print(f"[decision_exact] wrote tag {new}: BER_SCORES={new} python postprocess.py write")


if __name__ == "__main__":
    main()
