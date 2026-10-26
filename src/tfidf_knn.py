"""TF-IDF nearest neighbours per country (GPU when available, multithreaded CPU otherwise).

Three views, each an L2-normalised TF-IDF vector (IDF fitted on the S2+S3 pool of the same country):
    name  : char_wb 3-grams + whole words of n_core                      (typos, transliteration, word order)
    addr  : words of a_full + '#<house no>' + '@<city>' tokens            (renamed / re-branded businesses)
    rec   : [sqrt(W) * name , sqrt(1-W) * addr]  -> cosine = W*name + (1-W)*addr   (the main generator)
For every S1 record the top-K pool records (S2+S3, same country) of each view become candidates.

    python tfidf_knn.py featurize train          # cache hashed TF matrices in work/tfidf/ (once per split)
    python tfidf_knn.py search train holdout     # -> work/cand_tfidf_train_holdout.parquet (+ recall report)
    python tfidf_knn.py search test all

Output columns: s1, rid (int64 ids), flags (512 name / 1024 addr / 2048 rec), r_name, r_addr, r_rec
(rank 0.. within that view, 255 = not in its top-K), c_name, c_addr, c_rec (cosine, NaN if not in top-K).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR

F = 2 ** 18                 # hashed feature space per view
W_NAME = 0.55               # weight of the name view inside 'rec'
K = {"name": 30, "addr": 20, "rec": 40}
FLAG = {"name": 512, "addr": 1024, "rec": 2048}
MIN_COS = 0.05
DF_MAX = 20_000             # query-side pruning: drop features whose pool df exceeds this ...
KEEP_RARE = 4               # ... but always keep each record's 4 rarest features (exact cosines are recomputed in union_trim.py)
TDIR = os.path.join(WORK_DIR, "tfidf")


# ------------------------------------------------------------------ subsets (shared with run_blocking)
def subset_mask(ids_int: np.ndarray, subset: str) -> np.ndarray:
    """ids_int: encoded int64 ids. 'all'; 'holdout' = id % 10 == 0; 'mK' = id % 10 == K
    ('m1' trains the trim model / matcher; test is processed as m0..m9 so each part looks like a train slice)."""
    if subset == "all":
        return np.ones(len(ids_int), bool)
    mod = 0 if subset == "holdout" else int(subset[1:])
    return (np.asarray(ids_int) % 10 ** 12) % 10 == mod


# ------------------------------------------------------------------ featurisation
def _vec(kind):
    from sklearn.feature_extraction.text import HashingVectorizer
    if kind == "char":
        return HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=F, alternate_sign=False,
                                 norm=None, lowercase=False, dtype=np.float32)
    return HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=F, alternate_sign=False,
                             norm=None, lowercase=False, dtype=np.float32)


def _feat_worker(args):
    names, addrs = args
    name = _vec("char").transform(names) + _vec("word").transform(names)
    addr = _vec("word").transform(addrs)
    # sublinear tf
    for m in (name, addr):
        m.data = np.log1p(m.data).astype(np.float32)
    return name.tocsr(), addr.tocsr()


def featurize(split, k, n_jobs=None):
    from concurrent.futures import ProcessPoolExecutor
    from blocking_keys import encode_id
    t0 = time.time()
    os.makedirs(TDIR, exist_ok=True)
    d = pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet"),
                        columns=["entity_id", "country", "n_core", "n_full", "a_full", "a_num", "a_city"])
    core = d.n_core.fillna("").astype(str)
    core = core.where(core.str.len() > 0, d.n_full.fillna("").astype(str))
    num = d.a_num.fillna("").astype(str).str.lstrip("0")
    city = d.a_city.fillna("").astype(str).str.replace(" ", "_")
    addr = (d.a_full.fillna("").astype(str) + np.where(num != "", " #" + num, "") +
            np.where(city != "", " @" + city, ""))
    names, addrs = core.tolist(), addr.tolist()
    ch = 100_000
    jobs = [(names[i:i + ch], addrs[i:i + ch]) for i in range(0, len(names), ch)]
    with ProcessPoolExecutor(n_jobs or max(1, (os.cpu_count() or 2) - 2)) as ex:
        parts = list(ex.map(_feat_worker, jobs))
    name = sp.vstack([p[0] for p in parts], format="csr")
    addr = sp.vstack([p[1] for p in parts], format="csr")
    sp.save_npz(os.path.join(TDIR, f"{split}_s{k}_name.npz"), name, compressed=False)
    sp.save_npz(os.path.join(TDIR, f"{split}_s{k}_addr.npz"), addr, compressed=False)
    pd.DataFrame({"id": encode_id(d.entity_id), "country": d.country.astype(str)}).to_parquet(
        os.path.join(TDIR, f"{split}_s{k}_meta.parquet"))
    print(f"featurized {split} S{k}: {name.shape[0]:,} rows, nnz name={name.nnz:,} addr={addr.nnz:,} "
          f"({time.time()-t0:.0f}s)", flush=True)


def load_feats(split, k):
    meta = pd.read_parquet(os.path.join(TDIR, f"{split}_s{k}_meta.parquet"))
    name = sp.load_npz(os.path.join(TDIR, f"{split}_s{k}_name.npz")).tocsr()
    addr = sp.load_npz(os.path.join(TDIR, f"{split}_s{k}_addr.npz")).tocsr()
    return meta, name, addr


# ------------------------------------------------------------------ tf-idf helpers
def idf_from(pool: sp.csr_matrix) -> np.ndarray:
    df = np.bincount(pool.indices, minlength=pool.shape[1]).astype(np.float64)
    return (np.log((1.0 + pool.shape[0]) / (1.0 + df)) + 1.0).astype(np.float32)


def weight(m: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    m = m.copy()
    m.data = m.data * idf[m.indices]
    norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags((1.0 / norms).astype(np.float32)) @ m


def rowdot(A: sp.csr_matrix, ai: np.ndarray, B: sp.csr_matrix, bi: np.ndarray, chunk=2_000_000) -> np.ndarray:
    """cosine of row pairs (A[ai[j]], B[bi[j]]) for already-normalised matrices."""
    out = np.empty(len(ai), np.float32)
    for s in range(0, len(ai), chunk):
        x, y = A[ai[s:s + chunk]], B[bi[s:s + chunk]]
        out[s:s + chunk] = np.asarray(x.multiply(y).sum(1)).ravel()
    return out


def prune_query(Q: sp.csr_matrix, P: sp.csr_matrix, df_max=DF_MAX, keep=KEEP_RARE) -> sp.csr_matrix:
    """Bound the cost of the sparse product: common features (long posting lists, low idf) are dropped
    from the query side, except each row's `keep` rarest ones."""
    df = np.bincount(P.indices, minlength=P.shape[1])
    Q = Q.tocsr(copy=True)
    Q.sort_indices()
    dfv = df[Q.indices]
    row = np.repeat(np.arange(Q.shape[0]), np.diff(Q.indptr))
    order = np.lexsort((dfv, row))                       # by row, then rarest first
    pos = np.empty(len(order), np.int64)
    pos[order] = np.arange(len(order)) - Q.indptr[row[order]]
    Q.data = np.where((dfv <= df_max) | (pos < keep), Q.data, 0).astype(Q.data.dtype)
    Q.eliminate_zeros()
    return Q


# ------------------------------------------------------------------ top-K backends
def _backend():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def topk_cuda(Q: sp.csr_matrix, P: sp.csr_matrix, k: int, mem_bytes=2.5e9, dev="cuda"):
    """exact top-k of Q @ P.T: P stays sparse on the GPU, query chunks are densified (F x b)."""
    import torch
    Pt = torch.sparse_csr_tensor(torch.from_numpy(P.indptr.astype(np.int64)),
                                 torch.from_numpy(P.indices.astype(np.int64)),
                                 torch.from_numpy(P.data.astype(np.float32)), size=P.shape, device=dev)
    n_p, f = P.shape
    k = min(k, n_p)
    b = int(max(8, min(1024, mem_bytes / (4 * n_p), 1.5e9 / (4 * f))))
    rows, cols, vals, ranks = [], [], [], []
    rk = torch.arange(k, device=dev).repeat(1, 1)
    for s in range(0, Q.shape[0], b):
        q = Q[s:s + b].tocoo()
        nb = Q[s:s + b].shape[0]
        D = torch.zeros((f, nb), device=dev, dtype=torch.float32)
        D[torch.from_numpy(q.col.astype(np.int64)).to(dev), torch.from_numpy(q.row.astype(np.int64)).to(dev)] = \
            torch.from_numpy(q.data.astype(np.float32)).to(dev)
        S = Pt @ D                                  # (n_p, nb)
        if (s // b) % 200 == 0:
            print(f"      {s + nb:>9,}/{Q.shape[0]:,} queries (chunk {b})", flush=True)
        v, i = S.topk(k, dim=0)                     # (k, nb)
        v, i = v.T.contiguous(), i.T.contiguous()   # (nb, k)
        keep = v > MIN_COS
        r = torch.arange(nb, device=dev).unsqueeze(1).expand(nb, k)
        rr = rk.expand(nb, k)
        rows.append((r[keep] + s).cpu().numpy()); cols.append(i[keep].cpu().numpy())
        vals.append(v[keep].cpu().numpy()); ranks.append(rr[keep].cpu().numpy())
        del D, S, v, i
    if dev == "cuda":
        torch.cuda.empty_cache()
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32),
            np.concatenate(ranks).astype(np.uint8))


def topk_cpu(Q: sp.csr_matrix, P: sp.csr_matrix, k: int, chunk=20_000):
    from sparse_dot_topn import sp_matmul_topn
    PT = P.T.tocsr()
    nt = max(1, (os.cpu_count() or 2) - 1)
    rows, cols, vals, ranks = [], [], [], []
    t0 = time.time()
    for s in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[s:s + chunk], PT, top_n=k, threshold=MIN_COS, sort=True, n_threads=nt).tocsr()
        if (s // chunk) % 5 == 0:
            print(f"      {s + chunk:>9,}/{Q.shape[0]:,} queries ({time.time()-t0:.0f}s)", flush=True)
        nnz_row = np.diff(C.indptr)
        rows.append(np.repeat(np.arange(C.shape[0]), nnz_row) + s)
        cols.append(C.indices.copy()); vals.append(C.data.astype(np.float32))
        ranks.append((np.arange(C.nnz) - np.repeat(C.indptr[:-1], nnz_row)).astype(np.uint8))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(ranks)


# ------------------------------------------------------------------ search
def load_side(split, ks, mask_fn=None):
    metas, names, addrs = [], [], []
    for k in ks:
        m, n, a = load_feats(split, k)
        if mask_fn is not None:
            keep = mask_fn(m.id.to_numpy())
            m, n, a = m[keep].reset_index(drop=True), n[keep], a[keep]
        metas.append(m); names.append(n); addrs.append(a)
    return (pd.concat(metas, ignore_index=True), sp.vstack(names, format="csr"), sp.vstack(addrs, format="csr"))


def country_views(qn, qa, pn, pa):
    """IDF from the pool, weight + normalise both sides, build the 'rec' concatenation."""
    idf_n, idf_a = idf_from(pn), idf_from(pa)
    qn, pn = weight(qn, idf_n), weight(pn, idf_n)
    qa, pa = weight(qa, idf_a), weight(pa, idf_a)
    a, b = np.float32(np.sqrt(W_NAME)), np.float32(np.sqrt(1 - W_NAME))
    qr = sp.hstack([qn * a, qa * b], format="csr")
    pr = sp.hstack([pn * a, pa * b], format="csr")
    return {"name": (qn, pn), "addr": (qa, pa), "rec": (qr, pr)}


def search(split, subset, backend=None):
    t0 = time.time()
    backend = backend or _backend()
    topk = {"cuda": topk_cuda, "cpu": topk_cpu,
            "torchcpu": lambda Q, P, k: topk_cuda(Q, P, k, mem_bytes=2e8, dev="cpu")}[backend]
    print(f"[tfidf] backend={backend}", flush=True)
    qm, qn, qa = load_side(split, (1,), (lambda ids: subset_mask(ids, subset)))
    pm, pn, pa = load_side(split, (2, 3))
    out = []
    for ctry in sorted(qm.country.unique()):
        qi = np.flatnonzero((qm.country == ctry).to_numpy())
        pi = np.flatnonzero((pm.country == ctry).to_numpy())
        if not len(qi) or not len(pi):
            continue
        views = country_views(qn[qi], qa[qi], pn[pi], pa[pi])
        parts = []
        for v, (Q, P) in views.items():
            t1 = time.time()
            if backend == "cpu":
                Q = prune_query(Q, P)
            r, c, val, rk = topk(Q, P, K[v])
            parts.append(pd.DataFrame({"q": r.astype(np.int64), "p": c.astype(np.int64), "view": v,
                                       "cos": val, "rank": rk}))
            print(f"  {ctry:8s} {v:5s} q={len(qi):>9,} pool={len(pi):>9,} pairs={len(r):>10,} "
                  f"({time.time()-t1:.0f}s)", flush=True)
        wide = None
        for part in parts:
            v = part.view.iat[0] if len(part) else None
            if v is None:
                continue
            part = part.drop(columns="view").rename(columns={"cos": f"c_{v}", "rank": f"r_{v}"})
            wide = part if wide is None else wide.merge(part, on=["q", "p"], how="outer")
        for v in K:
            if f"r_{v}" not in wide:
                wide[f"r_{v}"] = np.nan; wide[f"c_{v}"] = np.nan
        flags = np.zeros(len(wide), np.uint16)
        for v, f in FLAG.items():
            flags |= np.where(wide[f"r_{v}"].notna(), f, 0).astype(np.uint16)
        res = pd.DataFrame({"s1": qm.id.to_numpy()[qi][wide.q.to_numpy()],
                            "rid": pm.id.to_numpy()[pi][wide.p.to_numpy()], "flags": flags})
        for v in K:
            res[f"r_{v}"] = wide[f"r_{v}"].fillna(255).astype(np.uint8).to_numpy()
            res[f"c_{v}"] = wide[f"c_{v}"].astype(np.float32).to_numpy()
        out.append(res)
    c = pd.concat(out, ignore_index=True)
    path = os.path.join(WORK_DIR, f"cand_tfidf_{split}_{subset}.parquet")
    c.to_parquet(path)
    print(f"[tfidf] {len(c):,} pairs -> {path} ({time.time()-t0:.0f}s)", flush=True)
    return c, qm


def report(c, split, subset, n_s1):
    import evaluation as ev
    g = ev.gt_int(holdout_only=False)
    g = g[subset_mask(g.s1.to_numpy(), subset)]
    m = g.merge(c, on=["s1", "rid"], how="left")
    print(f"[tfidf] union recall={m['flags'].notna().mean():.4f}  avg/s1={len(c)/n_s1:.1f}")
    for v in K:
        r = m[f"r_{v}"].fillna(255).to_numpy()
        pts = [k for k in (1, 3, 5, 10, 20, 30, 40) if k <= K[v]]
        print(f"  {v:5s} recall@k " + "  ".join(f"@{k}={(r < k).mean():.4f}" for k in pts))


if __name__ == "__main__":
    cmd, split = sys.argv[1], sys.argv[2]
    if cmd == "featurize":
        for k in ((1, 2, 3) if len(sys.argv) < 4 else (int(sys.argv[3]),)):
            featurize(split, k)
    elif cmd == "search":
        subset = sys.argv[3] if len(sys.argv) > 3 else "holdout"
        backend = sys.argv[4] if len(sys.argv) > 4 else None
        c, qm = search(split, subset, backend)
        if split == "train":
            report(c, split, subset, len(qm))
