"""Decision rule on top of the stage-2 scores, and the two submission files.

Decision rule (applied per Source-1 entity, all parameters tuned for macro F0.5 on train m1 OOF scores,
reported on the untouched holdout):
    1. exclusivity   - every S2/S3 record may belong to one S1 only: it is kept only for the S1 that gives it
                       the highest stage-2 score (applied over the whole split / all of test)
    2. threshold     - keep pairs with p2 >= thr[country]   (France / unseen countries use the global thr)
    3. relative      - ... and p2 >= rel * (best p2 of that S1)
    4. fallback      - an S1 with nothing kept gets its single best candidate if that p2 >= fb
    5. caps          - at most 5 S2 and 6 S3 records per S1 (max seen in train), best-scored first

    python postprocess.py tune        # -> work/models/decision.json + holdout report
    python postprocess.py write       # -> output/matching_results.tsv, output/candidate_pairs.tsv

candidate_pairs.tsv = every (S1, S2/S3) pair the matcher scored (top-15 trimmed union, the model's exact
inference set); matching_results.tsv is a subset of it by construction.
"""
import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, DATA_DIR
from blocking_keys import encode_id
import tfidf_knn as T

CAP = {2: 5, 3: 6}
# BER_SCORES=b -> use the multi-slice / ensemble scores from train_big.py (default: the original m1 models)
TAG = os.environ.get("BER_SCORES", "")
if TAG:
    OOF_FILE, HO_FILE, TEST_FILE = f"scores_train_oof_{TAG}.parquet", f"scores_train_holdout_{TAG}.parquet", \
        f"scores_test_{TAG}.parquet"
    TUNE_SUBSETS = os.environ.get("BER_TUNE_SUBSETS",
                                  os.environ.get("BER_TRAIN_SUBSETS", "m1,m2,m3,m4,m5")).split(",")
    DEC_PATH = os.path.join(WORK_DIR, "models", f"decision_{TAG}.json")
else:
    OOF_FILE, HO_FILE, TEST_FILE = "scores_train_m1_oof.parquet", "scores_train_holdout.parquet", "scores_test.parquet"
    TUNE_SUBSETS = ["m1"]
    DEC_PATH = os.path.join(WORK_DIR, "models", "decision.json")


# ------------------------------------------------------------------ decision rule
def apply_rule(d: pd.DataFrame, prm: dict, country: pd.Series) -> np.ndarray:
    """d: s1, rid, p2 (one row per candidate pair). country: Series s1 -> country label.
    Returns a boolean mask of predicted matches."""
    p = d.p2.to_numpy().astype(np.float64)
    s1 = d.s1.to_numpy()
    rid = d.rid.to_numpy()
    # 1. exclusivity: best S1 per rid
    order = np.lexsort((-p, rid))
    ro = rid[order]
    first_sorted = np.r_[True, ro[1:] != ro[:-1]]      # first row of each rid block = its best S1
    excl = np.zeros(len(d), bool)
    excl[order] = first_sorted
    s1_codes, s1_uni = pd.factorize(s1)
    if prm.get("mode") == "efx":                      # exact expected F0.5 (decision_exact.py)
        import decision_exact as DX
        keep = DX.efx_keep(p, s1_codes, excl, prm.get("gamma", 1.0), prm.get("extra", 0.0), prm.get("floor", 0.0))
        return _caps(keep, p, rid, s1_codes) if prm.get("caps", True) else keep
    if prm.get("mode") == "efo":
        keep = _efo_keep(p, s1_codes, len(s1_uni), excl, prm.get("gamma", 1.0), prm.get("floor", 0.0),
                         prm.get("extra", 0.0))
        return _caps(keep, p, rid, s1_codes) if prm.get("caps", True) else keep
    # 2. per-country threshold
    ctry = country.reindex(s1).fillna("").to_numpy()
    thr = np.full(len(d), prm["thr"], np.float64)
    for c, t in prm.get("thr_country", {}).items():
        thr[ctry == c] = t
    keep = excl & (p >= thr)
    # 3. relative to the S1's best score
    best = np.zeros(len(s1_uni), np.float64)
    np.maximum.at(best, s1_codes, np.where(excl, p, 0.0))
    if prm.get("rel", 0) > 0:
        keep &= p >= prm["rel"] * best[s1_codes]
    # 4. fallback: best exclusive candidate for S1s left empty
    if prm.get("fb", 1.1) <= 1.0:
        has = np.zeros(len(s1_uni), bool)
        np.logical_or.at(has, s1_codes, keep)
        is_best = excl & (p >= best[s1_codes]) & (p > 0)
        # one row per S1 even with ties
        cand = np.flatnonzero(is_best & ~has[s1_codes] & (p >= prm["fb"]))
        _, first_idx = np.unique(s1_codes[cand], return_index=True)
        keep[cand[first_idx]] = True
    # 5. caps per source
    if prm.get("caps", True):
        keep = _caps(keep, p, rid, s1_codes)
    return keep


def _caps(keep, p, rid, s1_codes):
    src = rid // 10 ** 12
    idx = np.flatnonzero(keep)
    o = idx[np.lexsort((-p[idx], src[idx], s1_codes[idx]))]
    grp = s1_codes[o] * 4 + src[o]
    start = np.r_[0, np.flatnonzero(grp[1:] != grp[:-1]) + 1]
    rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    cap = np.where(src[o] == 2, CAP[2], CAP[3])
    keep = keep.copy()
    keep[o[rank >= cap]] = False
    return keep


def _efo_keep(p, s1_codes, n_groups, excl, gamma, floor, extra):
    """Per-entity expected-F0.5-optimal selection. Candidates (after exclusivity) are sorted by
    q = p**gamma; for each k the expected F0.5 of predicting the top k is approximated by
    1.25 * sum(top-k q) / (0.25 * (sum(all q) + extra) + k), and predicting nothing by prod(1 - q)
    (the entity is a singleton). The best option is taken per entity; q < floor is never predicted."""
    q = np.where(excl, np.clip(p, 0.0, 1.0) ** gamma, 0.0)
    idx = np.flatnonzero(excl & (q > 0))
    keep = np.zeros(len(p), bool)
    if not len(idx):
        return keep
    o = idx[np.lexsort((-q[idx], s1_codes[idx]))]
    g = s1_codes[o]; qo = q[o]
    start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    lens = np.diff(np.r_[start, len(o)])
    rank = np.arange(len(o)) - np.repeat(start, lens)
    cs = np.cumsum(qo)
    before = np.repeat(np.r_[0.0, cs[start[1:] - 1]], lens)
    ck = cs - before                                           # sum of the top (rank+1) q within the entity
    tot = np.repeat(np.add.reduceat(qo, start), lens) + extra  # expected number of true matches
    ek = 1.25 * ck / (0.25 * tot + rank + 1)
    ek[qo < floor] = -1.0
    best_e = np.maximum.reduceat(ek, start)
    # k* = position of the best E_k (first occurrence) if it beats predicting nothing
    is_best = ek >= np.repeat(best_e, lens)
    first_best = np.full(len(start), -1)
    pos = np.flatnonzero(is_best)
    grp_of = np.repeat(np.arange(len(start)), lens)
    uniq, fi = np.unique(grp_of[pos], return_index=True)
    first_best[uniq] = rank[pos[fi]]
    kstar = np.where(best_e > np.exp(np.add.reduceat(np.log1p(-np.minimum(qo, 1 - 1e-9)), start)),
                     first_best + 1, 0)
    keep[o[rank < np.repeat(kstar, lens)]] = True
    return keep


def macro_f05(d, keep, n_true: pd.Series, s1_all: np.ndarray):
    """d has y; n_true: s1 -> number of true pairs (all S1 in the universe, 0 for singletons)."""
    idx = pd.Index(s1_all)
    codes = idx.get_indexer(d.s1.to_numpy())
    npred = np.bincount(codes[keep], minlength=len(idx))
    tp = np.bincount(codes[keep], weights=d.y.to_numpy()[keep], minlength=len(idx))
    nt = n_true.reindex(idx, fill_value=0).to_numpy()
    f = np.zeros(len(idx))
    sg = nt == 0
    f[sg] = npred[sg] == 0
    ok = ~sg & (tp > 0)
    pr, rc = tp[ok] / npred[ok], tp[ok] / nt[ok]
    f[ok] = 1.25 * pr * rc / (0.25 * pr + rc)
    P = tp.sum() / max(1, npred.sum()); R = tp.sum() / max(1, nt.sum())
    return float(f.mean()), P, R, f, nt, npred, tp


# ------------------------------------------------------------------ tuning
def _universe(subset):
    """subset: 'holdout', 'm1' or a list of subsets."""
    import union_trim as U
    subs = [subset] if isinstance(subset, str) else list(subset)
    s1 = pd.read_parquet(os.path.join(WORK_DIR, "norm_train_s1.parquet"), columns=["entity_id", "country"])
    s1["id"] = encode_id(s1.entity_id)
    m = np.zeros(len(s1), bool)
    for sb in subs:
        m |= T.subset_mask(s1.id.to_numpy(), sb)
    s1 = s1[m]
    g = pd.concat([U.gt_subset(sb) for sb in subs], ignore_index=True)
    n_true = g.groupby("s1").size()
    return s1.id.to_numpy(), pd.Series(s1.country.astype(str).to_numpy(), index=s1.id.to_numpy()), n_true


def _read_scores(fname):
    d = pd.read_parquet(os.path.join(WORK_DIR, fname))
    bp = os.path.join(WORK_DIR, "models", f"blend_{TAG}.json")
    if TAG and os.path.exists(bp):
        w = json.load(open(bp))
        d["p2"] = sum(v * d[f"p2_{k}"].to_numpy() for k, v in w.items()).astype(np.float32)
        print(f"  [{fname}] p2 = blend {w}", flush=True)
    return d


def tune():
    t0 = time.time()
    tr = _read_scores(OOF_FILE)
    ids_tr, ctry_tr, nt_tr = _universe(TUNE_SUBSETS)
    tr = tr[np.isin(tr.s1.to_numpy(), ids_tr)].reset_index(drop=True)   # tune on the chosen slices only
    ev = lambda prm: macro_f05(tr, apply_rule(tr, prm, ctry_tr), nt_tr, ids_tr)[0]
    best = {"thr": 0.7, "rel": 0.0, "fb": 1.1, "caps": True}
    base = ev(best)
    print(f"[tune] start (thr 0.70, excl + caps) m1 OOF F0.5 = {base:.5f}")
    # coordinate search, two sweeps
    grids = {"thr": np.round(np.arange(0.30, 0.95, 0.025), 3),
             "rel": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
             "fb": [1.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]}
    score = base
    for sweep in range(2):
        for k, grid in grids.items():
            for v in grid:
                prm = dict(best, **{k: float(v)})
                f = ev(prm)
                if f > score + 1e-6:
                    score, best = f, prm
            print(f"  sweep {sweep} {k}: {best[k]} -> {score:.5f}", flush=True)
    # per-country thresholds (only countries present in train; others keep the global thr)
    tc = {}
    for c in sorted(ctry_tr.unique()):
        bc, sc = best["thr"], score
        for v in grids["thr"]:
            prm = dict(best, thr_country=dict(tc, **{c: float(v)}))
            f = ev(prm)
            if f > sc + 1e-6:
                sc, bc = f, float(v)
        tc[c] = bc; score = sc
        print(f"  thr[{c}] = {bc} -> {score:.5f}", flush=True)
    best["thr_country"] = tc
    print(f"[tune] best threshold rule on OOF: {best}  F0.5={score:.5f} ({time.time()-t0:.0f}s)", flush=True)
    # expected-F0.5 per-entity rule
    e_best, e_score = None, -1
    for gamma in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        for floor in (0.0, 0.1, 0.2, 0.3, 0.4):
            prm = {"mode": "efo", "gamma": gamma, "floor": floor, "extra": 0.0, "caps": True}
            f = ev(prm)
            if f > e_score:
                e_score, e_best = f, prm
    for extra in (0.02, 0.05, 0.1, 0.2):
        prm = dict(e_best, extra=extra); f = ev(prm)
        if f > e_score:
            e_score, e_best = f, prm
    print(f"[tune] best expected-F0.5 rule on OOF: {e_best}  F0.5={e_score:.5f} ({time.time()-t0:.0f}s)", flush=True)
    report_holdout(best, label="threshold rule")
    report_holdout(e_best, label="expected-F0.5 rule")
    if e_score > score:
        best, score = e_best, e_score
    print(f"[tune] chosen: {best}  OOF F0.5={score:.5f}")
    os.makedirs(os.path.dirname(DEC_PATH), exist_ok=True)
    json.dump(best, open(DEC_PATH, "w"), indent=1)
    report_holdout(best, label="CHOSEN")


def report_holdout(prm=None, label="tuned rule"):
    prm = prm or json.load(open(DEC_PATH))
    ho = _read_scores(HO_FILE)
    ids, ctry, nt = _universe("holdout")
    rows = []
    for name, p in (("stage-2 p2 >= 0.70 + exclusivity (baseline rule)", {"thr": 0.7, "caps": False}),
                    ("+ caps", {"thr": 0.7, "caps": True}),
                    (label, prm)):
        f, P, R, *_ = macro_f05(ho, apply_rule(ho, p, ctry), nt, ids)
        rows.append((name, f, P, R))
    print("\n[holdout] rule                                         macroF0.5  pairP   pairR")
    for name, f, P, R in rows:
        print(f"  {name:50s} {f:.5f}  {P:.4f}  {R:.4f}")
    keep = apply_rule(ho, prm, ctry)
    f, P, R, fe, ntv, npred, tp = macro_f05(ho, keep, nt, ids)
    by_c = pd.DataFrame({"c": ctry.reindex(ids).to_numpy(), "f": fe}).groupby("c").f.agg(["size", "mean"])
    print("\n[holdout] per country:\n" + by_c.round(4).to_string())
    single = ntv == 0
    print(f"[holdout] singletons {single.mean():.3f} of S1, singleton F {fe[single].mean():.4f}; "
          f"non-singleton F {fe[~single].mean():.4f}")
    print(f"[holdout] FP pairs {int((npred - tp).sum()):,}  FN pairs {int((ntv - tp).sum()):,}  "
          f"(of which not in candidates {int(ntv.sum() - ho.y.sum()):,})")
    json.dump({"holdout_macro_f05": f, "pair_precision": P, "pair_recall": R},
              open(os.path.join(WORK_DIR, "models", f"holdout_report{'_' + TAG if TAG else ''}.json"), "w"), indent=1)


# ------------------------------------------------------------------ submission files
def _test_id_strings():
    """int id -> exact original string, straight from the test source files (no re-formatting)."""
    from io_utils import read_source
    parts = []
    for k in (1, 2, 3):
        e = read_source("test", k, ["entity_id"]).entity_id
        parts.append(pd.Series(e.astype(str).to_numpy(), index=encode_id(e)))
    m = pd.concat(parts)
    assert m.index.is_unique, "id encoding collision"
    return m, parts[0]


def _write_lists(pairs: pd.DataFrame, s1_order: pd.Series, id_str: pd.Series, path: str, col: str):
    pairs = pairs.sort_values(["s1", "rank"])
    ids = id_str.reindex(pairs.rid.to_numpy()).to_numpy()
    assert not pd.isna(ids).any(), "unknown rid"
    lists = pd.Series(ids).groupby(pairs.s1.to_numpy(), sort=False).agg(",".join)
    s1_int = encode_id(s1_order)
    body = lists.reindex(s1_int).fillna("").to_numpy()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in zip(s1_order.to_numpy(), body):
            f.write(f"{a}\t{b}\n")
    return int((body != "").sum())


def write():
    t0 = time.time()
    prm = json.load(open(DEC_PATH))
    d = _read_scores(TEST_FILE)
    d = d.drop_duplicates(["s1", "rid"]).reset_index(drop=True)
    s1c = pd.read_parquet(os.path.join(WORK_DIR, "norm_test_s1.parquet"), columns=["entity_id", "country"])
    ctry = pd.Series(s1c.country.astype(str).to_numpy(), index=encode_id(s1c.entity_id))
    keep = apply_rule(d, prm, ctry)
    id_str, s1_order = _test_id_strings()
    d["rank"] = -d.p2          # best first inside each list
    os.makedirs(OUT_DIR, exist_ok=True)
    n_c = _write_lists(d[["s1", "rid", "rank"]], s1_order, id_str,
                       os.path.join(OUT_DIR, "candidate_pairs.tsv"), "candidate_entity_ids")
    m = d.loc[keep, ["s1", "rid", "rank"]]
    n_m = _write_lists(m, s1_order, id_str, os.path.join(OUT_DIR, "matching_results.tsv"), "matched_entity_ids")
    by_c = pd.DataFrame({"c": ctry.reindex(m.s1.to_numpy()).to_numpy()}).value_counts()
    print(f"[write] S1 rows {len(s1_order):,}; candidate pairs {len(d):,} ({n_c:,} S1 non-empty); "
          f"matched pairs {len(m):,} ({n_m:,} S1 non-empty, {len(m)/len(s1_order):.3f}/S1) ({time.time()-t0:.0f}s)")
    print("[write] matched pairs by country:\n" + by_c.to_string())
    json.dump({"decision": prm, "test_s1": len(s1_order), "candidate_pairs": len(d), "matched_pairs": int(len(m)),
               "s1_with_match": n_m}, open(os.path.join(OUT_DIR, "run_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tune":
        tune()
    elif cmd == "report":
        report_holdout()
    elif cmd == "write":
        write()
