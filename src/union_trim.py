"""Union of all candidate generators + cheap score + top-N trim.

    python union_trim.py build train m1        # union + features for the trim-model training subset (id % 10 == 1)
    python union_trim.py build train holdout   # same for the shared holdout (id % 10 == 0)
    python union_trim.py fit                   # gradient-boosted trim model on train/m1, report on train/holdout
    python union_trim.py build test all        # (and train all) -> features for every S1
    python union_trim.py trim train holdout    # apply model, keep top-N per S1 -> work/cand_{split}_{subset}.parquet

Inputs:  work/cand_keys_{split}_{subset}.parquet  (run_blocking.py)
         work/cand_tfidf_{split}_{subset}.parquet (tfidf_knn.py)
         any extra generator file work/cand_extra_{split}_{subset}_*.parquet with columns s1_id|s1, rid (+ score)
Output:  work/union_{split}_{subset}.parquet   (all pairs + features)
         work/cand_{split}_{subset}.parquet    (trimmed: s1, rid, trim_score, trim_rank + features) -> matcher input
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
import tfidf_knn as T
from blocking_keys import METHODS, encode_id

TOP_N = 15          # default trim size (see the recall@N table printed by `fit`)
MIN_SCORE = 0.0     # optional probability floor applied after top-N
MODEL = os.path.join(WORK_DIR, "trim_hgb.pkl")   # sklearn HistGradientBoosting (LightGBM DLL is blocked on this laptop)
KEY_FLAGS = list(METHODS)                # 1..256
TF_FLAGS = list(T.FLAG.values())         # 512, 1024, 2048
EXTRA_FLAG = 4096
EMB = os.environ.get("BER_EMB", "") == "1"      # use embedding-retrieval candidates and features
V3 = os.environ.get("BER_V3", "") == "1"        # architecture v3: candidates = pruned union from v3_retrieval + prune
V3_FEATS = ["rank_fn", "rank_fa", "rank_ff", "rank_bn", "rank_ba", "rank_bf", "score_fn", "score_fa", "score_ff",
            "score_bn", "score_ba", "score_bf", "h_emb", "h_key", "h_tf", "hits", "min_fr", "min_br",
            "cos_n", "cos_a", "cos_f", "csls_n", "csls_a", "csls_f", "t_has_addr", "s_has_addr",
            "s_mg_f", "s_mg_n", "t_mg_f", "t_mg_n", "s_rank_f", "t_rank_f", "prune_c"]


# ------------------------------------------------------------------ loading
def _read_cands(kind, split, subset, columns=None):
    """cand_{kind}_{split}_{subset}.parquet, or the '_all' file filtered to the subset."""
    p = os.path.join(WORK_DIR, f"cand_{kind}_{split}_{subset}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p, columns=columns)
    p = os.path.join(WORK_DIR, f"cand_{kind}_{split}_all.parquet")
    if os.path.exists(p):
        d = pd.read_parquet(p, columns=columns)
        return d[T.subset_mask(d.s1.to_numpy(), subset)].reset_index(drop=True)
    return None


def load_union(split, subset):
    if V3:
        c = pd.read_parquet(os.path.join(WORK_DIR, "v3", f"cand_{split}_{subset}.parquet"))
        c["flags"] = (c.kflags.to_numpy() | c.tflags.to_numpy()
                      | np.where(c.h_emb.to_numpy() > 0, EXTRA_FLAG, 0)).astype(np.int64)
        for v in T.K:
            c[f"r_{v}"] = c[f"r_{v}"].fillna(255).clip(upper=255).astype(np.uint8)
        return c.drop(columns=["kflags", "tflags"])
    parts = []
    k = _read_cands("keys", split, subset, ["s1", "rid", "flags"])
    if k is not None:
        parts.append(k)
    tf = _read_cands("tfidf", split, subset)
    if tf is not None:
        parts.append(tf[["s1", "rid", "flags"]])
    sims = []
    for p in sorted(glob.glob(os.path.join(WORK_DIR, f"cand_extra_{split}_{subset}_*.parquet"))):
        if not EMB and p.endswith("_emb.parquet"):
            continue
        e = pd.read_parquet(p)
        if "s1" not in e:
            e = pd.DataFrame({"s1": encode_id(e.s1_id), "rid": encode_id(e.rid)})
        parts.append(pd.DataFrame({"s1": e.s1.to_numpy(), "rid": e.rid.to_numpy(), "flags": np.uint16(EXTRA_FLAG)}))
        if "e_f" in e:
            sims.append(e[["s1", "rid", "e_n", "e_f"]])
        print(f"  + extra candidates {os.path.basename(p)}: {len(e):,}")
    c = pd.concat(parts, ignore_index=True)
    c["flags"] = c["flags"].astype(np.int64)
    c = c.groupby(["s1", "rid"], sort=False)["flags"].sum().reset_index()   # sources use disjoint bits
    if tf is not None:
        c = c.merge(tf.drop(columns=["flags"]), on=["s1", "rid"], how="left")
    if EMB:
        if sims:
            es = pd.concat(sims, ignore_index=True).groupby(["s1", "rid"], sort=False).max().reset_index()
            c = c.merge(es, on=["s1", "rid"], how="left")
        for col in ("e_n", "e_f"):
            if col not in c:
                c[col] = np.nan
            c[col] = c[col].astype(np.float32)
    for v in T.K:
        if f"r_{v}" not in c:
            c[f"r_{v}"] = 255
        c[f"r_{v}"] = c[f"r_{v}"].fillna(255).astype(np.uint8)
    return c


# ------------------------------------------------------------------ features
NCOLS = ["entity_id", "country", "n_core", "a_num", "a_nums", "a_city", "a_street", "a_state", "a_empty", "legal"]


def _num_int(s: pd.Series) -> np.ndarray:
    x = s.fillna("").astype(str).str.lstrip("0").str.slice(0, 9)
    return pd.to_numeric(x.where(x.str.fullmatch(r"\d+"), None), errors="coerce").to_numpy(dtype=float)


def exact_cosines(c, split, subset):
    """c_name / c_addr / c_rec for every pair (the top-K search only has them for its own hits)."""
    qm, qn, qa = T.load_side(split, (1,), (lambda ids: T.subset_mask(ids, subset)))
    pm, pn, pa = T.load_side(split, (2, 3))
    qpos = pd.Series(np.arange(len(qm)), index=qm.id.to_numpy())
    ppos = pd.Series(np.arange(len(pm)), index=pm.id.to_numpy())
    cn = np.full(len(c), np.nan, np.float32); ca = cn.copy()
    qi_all = qpos.reindex(c.s1.to_numpy()).to_numpy()
    pi_all = ppos.reindex(c.rid.to_numpy()).to_numpy()
    qctry = qm.country.to_numpy()
    pair_ctry = np.where(np.isnan(qi_all), "", qctry[np.nan_to_num(qi_all).astype(np.int64)])
    for ctry in sorted(qm.country.unique()):
        qsel = np.flatnonzero((qm.country == ctry).to_numpy())
        psel = np.flatnonzero((pm.country == ctry).to_numpy())
        idx = np.flatnonzero((pair_ctry == ctry) & ~np.isnan(pi_all))
        if not len(idx) or not len(psel):
            continue
        views = T.country_views(qn[qsel], qa[qsel], pn[psel], pa[psel])
        # global row -> row inside this country's matrices
        qloc = np.full(len(qm), -1, np.int64); qloc[qsel] = np.arange(len(qsel))
        ploc = np.full(len(pm), -1, np.int64); ploc[psel] = np.arange(len(psel))
        a = qloc[qi_all[idx].astype(np.int64)]; b = ploc[pi_all[idx].astype(np.int64)]
        ok = (a >= 0) & (b >= 0)                  # cross-country pairs (should not exist) stay NaN
        idx, a, b = idx[ok], a[ok], b[ok]
        cn[idx] = T.rowdot(views["name"][0], a, views["name"][1], b)
        ca[idx] = T.rowdot(views["addr"][0], a, views["addr"][1], b)
    c["c_name"], c["c_addr"] = cn, ca
    c["c_rec"] = (T.W_NAME * np.nan_to_num(cn) + (1 - T.W_NAME) * np.nan_to_num(ca)).astype(np.float32)
    return c


def add_features(c, split, subset):
    t0 = time.time()
    c = exact_cosines(c, split, subset)
    print(f"  cosines ({time.time()-t0:.0f}s)", flush=True)
    s1 = pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s1.parquet"), columns=NCOLS)
    s1["id"] = encode_id(s1.entity_id)
    s1 = s1[T.subset_mask(s1.id.to_numpy(), subset)].set_index("id")
    pool = pd.concat([pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet"), columns=NCOLS)
                      for k in (2, 3)], ignore_index=True)
    pool["id"] = encode_id(pool.entity_id)
    pool = pool.set_index("id")
    A = s1.reindex(c.s1.to_numpy()); B = pool.reindex(c.rid.to_numpy())

    def eq(col):
        x, y = A[col].fillna("").astype(str).to_numpy(), B[col].fillna("").astype(str).to_numpy()
        return np.where((x == "") | (y == ""), -1, (x == y).astype(np.int8)).astype(np.int8)

    na, nb = _num_int(A.a_num), _num_int(B.a_num)
    d = np.abs(na - nb)
    c["num_eq"] = eq("a_num")
    c["num_diff"] = np.where(np.isnan(d), -1, np.minimum(d, 1e6)).astype(np.float32)
    # dropped / extra digit: one number string contained in the other
    xa, xb = A.a_num.fillna("").astype(str).str.lstrip("0"), B.a_num.fillna("").astype(str).str.lstrip("0")
    c["num_sub"] = np.array([(len(p) > 0 and len(q) > 0 and p != q and (p in q or q in p)) for p, q in zip(xa, xb)],
                            dtype=np.int8)
    # any shared number (India: many numbers per address)
    sa = A.a_nums.fillna("").astype(str).str.split(" "); sb = B.a_nums.fillna("").astype(str).str.split(" ")
    c["nums_shared"] = np.array([len(set(p) & set(q) - {""}) for p, q in zip(sa, sb)], dtype=np.int8)
    c["city_eq"] = eq("a_city"); c["street_eq"] = eq("a_street"); c["state_eq"] = eq("a_state")
    c["name_eq"] = eq("n_core"); c["legal_eq"] = eq("legal")
    c["b_empty"] = B.a_empty.fillna(True).astype(np.int8).to_numpy()
    c["src"] = (c.rid.to_numpy() // 10 ** 12).astype(np.int8)
    c["country"] = A.country.fillna("").astype(str).map({"US": 0, "India": 1, "France": 2}).fillna(3).astype(np.int8).to_numpy()
    fl = c["flags"].to_numpy()
    for f in KEY_FLAGS + TF_FLAGS + [EXTRA_FLAG]:
        c[f"f{f}"] = ((fl & f) > 0).astype(np.int8)
    c["n_keys"] = sum(c[f"f{f}"] for f in KEY_FLAGS).astype(np.int8)
    # context: how this pair ranks among the S1's candidates and among the rid's S1s (exclusivity)
    g = c.groupby("s1", sort=False)
    c["s1_n"] = g.rid.transform("size").astype(np.int32)
    c["s1_rank_rec"] = g.c_rec.rank(ascending=False, method="first").astype(np.int32)
    c["s1_gap_rec"] = (g.c_rec.transform("max") - c.c_rec).astype(np.float32)
    c["s1_rank_name"] = g.c_name.rank(ascending=False, method="first").fillna(999).astype(np.int32)
    g = c.groupby("rid", sort=False)
    c["rid_n"] = g.s1.transform("size").astype(np.int32)
    c["rid_rank_rec"] = g.c_rec.rank(ascending=False, method="first").astype(np.int32)
    c["rid_gap_rec"] = (g.c_rec.transform("max") - c.c_rec).astype(np.float32)
    if EMB and not V3:
        g = c.groupby("s1", sort=False)
        c["s1_rank_ef"] = g.e_f.rank(ascending=False, method="first").fillna(999).astype(np.int32)
        c["s1_gap_ef"] = (g.e_f.transform("max") - c.e_f).astype(np.float32)
        c["s1_rank_en"] = g.e_n.rank(ascending=False, method="first").fillna(999).astype(np.int32)
    print(f"  features done ({time.time()-t0:.0f}s)", flush=True)
    return c


FEATS = (["c_name", "c_addr", "c_rec", "r_name", "r_addr", "r_rec", "num_eq", "num_diff", "num_sub", "nums_shared",
          "city_eq", "street_eq", "state_eq", "name_eq", "legal_eq", "b_empty", "src", "country", "n_keys",
          "s1_n", "s1_rank_rec", "s1_gap_rec", "s1_rank_name", "rid_n", "rid_rank_rec", "rid_gap_rec"]
         + [f"f{f}" for f in KEY_FLAGS + TF_FLAGS + [EXTRA_FLAG]])
if EMB and not V3:     # supervised multilingual embedding retrieval (emb_retrieval.py): similarities + ranks
    FEATS = FEATS + ["e_n", "e_f", "s1_rank_ef", "s1_gap_ef", "s1_rank_en"]
if V3:
    FEATS = FEATS + V3_FEATS


def build(split, subset):
    t0 = time.time()
    c = load_union(split, subset)
    print(f"[union {split}/{subset}] {len(c):,} pairs", flush=True)
    c = add_features(c, split, subset)
    if split == "train":
        g = gt_subset(subset)
        c = c.merge(g.assign(y=np.int8(1)), on=["s1", "rid"], how="left")
        c["y"] = c.y.fillna(0).astype(np.int8)
        print(f"  union pair recall = {c.y.sum() / len(g):.4f} (true pairs {len(g):,})")
    c.to_parquet(os.path.join(WORK_DIR, f"union_{split}_{subset}.parquet"))
    if V3:                                  # the cheap pruning already selected the candidates: pass them through
        c = c.sort_values(["s1", "prune_c"], ascending=[True, False]).reset_index(drop=True)
        c["trim_score"] = c.prune_c.astype(np.float32)
        c["trim_rank"] = c.groupby("s1").cumcount()
        c.to_parquet(os.path.join(WORK_DIR, f"cand_{split}_{subset}.parquet"))
    print(f"[union] saved ({time.time()-t0:.0f}s)", flush=True)
    return c


def gt_subset(subset):
    import evaluation as ev
    g = ev.gt_int(holdout_only=False)
    return g[T.subset_mask(g.s1.to_numpy(), subset)]


def n_s1(split, subset):
    ids = encode_id(pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s1.parquet"), columns=["entity_id"]).entity_id)
    return int(T.subset_mask(ids, subset).sum())


# ------------------------------------------------------------------ model
def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=63, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0)


def fit():
    import pickle
    tr = pd.read_parquet(os.path.join(WORK_DIR, "union_train_m1.parquet"))
    t0 = time.time()
    m = _model().fit(tr[FEATS].astype(np.float32), tr.y)
    with open(MODEL, "wb") as f:
        pickle.dump(m, f)
    print(f"trim model: {m.n_iter_} iters on {len(tr):,} pairs ({time.time()-t0:.0f}s)", flush=True)
    ho = pd.read_parquet(os.path.join(WORK_DIR, "union_train_holdout.parquet"))
    ho["trim_score"] = m.predict_proba(ho[FEATS].astype(np.float32))[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"holdout AUC {roc_auc_score(ho.y, ho.trim_score):.4f}")
    recall_at_n(ho, gt_subset("holdout"), n_s1("train", "holdout"))


def recall_at_n(c, g, ns1):
    import evaluation as ev
    c = c.sort_values(["s1", "trim_score"], ascending=[True, False])
    c["trim_rank"] = c.groupby("s1").cumcount()
    print(f"{'N':>5} {'pairs/s1':>9} {'recall':>8} {'full_cov':>9} {'oracle_f05':>10}")
    for n in (5, 10, 15, 20, 25, 30, 40, 50, 75, 10 ** 9):
        cc = c[c.trim_rank < n]
        r = ev.recall_report_int(cc, g, ns1)
        print(f"{('all' if n > 1e6 else n):>5} {r['avg_cands_per_s1']:>9.2f} {r['pair_recall']:>8.4f} "
              f"{r['entities_fully_covered']:>9.4f} {r['oracle_macro_f05']:>10.4f}", flush=True)
    return c


def trim(split, subset, top_n=TOP_N):
    import pickle
    c = pd.read_parquet(os.path.join(WORK_DIR, f"union_{split}_{subset}.parquet"))
    with open(MODEL, "rb") as f:
        m = pickle.load(f)
    p = np.empty(len(c), np.float32)
    for s in range(0, len(c), 2_000_000):      # chunked: keeps peak memory low while other jobs run
        p[s:s + 2_000_000] = m.predict_proba(c[FEATS].iloc[s:s + 2_000_000].astype(np.float32))[:, 1]
    c["trim_score"] = p
    del p
    if split == "train" and subset == "holdout":
        c = recall_at_n(c, gt_subset(subset), n_s1(split, subset))
    else:
        c = c.sort_values(["s1", "trim_score"], ascending=[True, False])
        c["trim_rank"] = c.groupby("s1").cumcount()
    c = c[(c.trim_rank < top_n) & (c.trim_score >= MIN_SCORE)].reset_index(drop=True)
    out = os.path.join(WORK_DIR, f"cand_{split}_{subset}.parquet")
    c.to_parquet(out)
    print(f"[trim] kept {len(c):,} pairs (top {top_n}) -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3])
    elif cmd == "fit":
        fit()
    elif cmd == "trim":
        trim(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else TOP_N)
