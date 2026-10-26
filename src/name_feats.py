"""Name-focused pair features (aimed at records with no address, where the name is the only evidence).

For pairs (s1, rid):
  idf_jacc     IDF-weighted token overlap of the core names (rare shared words count more)
  idf_shared   max IDF among shared tokens          idf_miss   max IDF among tokens only one side has
  n_tok_a/b    token counts                          name_freq  log #pool records with the same core name (chains)
  acro         one side is the acronym of the other  jw / tsort / psr   Jaro-Winkler, token-sort, partial ratio
  skel_eq      consonant-skeleton equality
  top_sim      token-set similarity of this record's name to the S1's best-scored OTHER candidate record
               (records of one business resemble each other)
Cached per split in work/namefeat_{tag}.parquet.
"""
import os
import sys
import math
import time
import collections
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from blocking_keys import encode_id

NFEATS = ["idf_jacc", "idf_shared", "idf_miss", "n_tok_a", "n_tok_b", "name_freq", "acro", "jw", "tsort", "psr",
          "skel_eq", "top_sim"]
COLS = ["entity_id", "n_core", "n_skel"]


def _load(split):
    s1 = pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s1.parquet"), columns=COLS)
    pool = pd.concat([pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet"), columns=COLS)
                      for k in (2, 3)], ignore_index=True)
    for d in (s1, pool):
        d["id"] = encode_id(d.entity_id)
        d.drop(columns="entity_id", inplace=True)
        d["n_core"] = d.n_core.fillna("").astype(str)
        d["n_skel"] = d.n_skel.fillna("").astype(str)
    return s1.drop_duplicates("id").set_index("id"), pool.drop_duplicates("id").set_index("id")


def _chunk(args):
    a, b, top, idf = args
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    out = np.zeros((len(a), 8), np.float32)
    for i, (x, y, t) in enumerate(zip(a, b, top)):
        ta, tb = set(x.split()), set(y.split())
        sh, un = ta & tb, ta | tb
        w = lambda s: sum(idf.get(k, 12.0) for k in s)
        out[i, 0] = w(sh) / w(un) if un else 0.0
        out[i, 1] = max((idf.get(k, 12.0) for k in sh), default=0.0)
        out[i, 2] = max((idf.get(k, 12.0) for k in (un - sh)), default=0.0)
        xa, yb = x.replace(" ", ""), y.replace(" ", "")
        ia = "".join(k[0] for k in x.split() if k); ib = "".join(k[0] for k in y.split() if k)
        out[i, 3] = 1.0 if (len(ia) > 1 and ia == yb) or (len(ib) > 1 and ib == xa) else 0.0
        out[i, 4] = JaroWinkler.normalized_similarity(x, y)
        out[i, 5] = fuzz.token_sort_ratio(x, y) / 100.0
        out[i, 6] = fuzz.partial_ratio(xa, yb) / 100.0 if xa and yb else 0.0
        out[i, 7] = fuzz.token_set_ratio(y, t) / 100.0 if t else -1.0
    return out


def compute(pairs, split, tag, score_col):
    """pairs: s1, rid, score_col (used to find each S1's best other candidate)."""
    path = os.path.join(WORK_DIR, f"namefeat_{tag}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    s1, pool = _load(split)
    docs = collections.Counter()
    for s in pool.n_core.to_numpy():
        docs.update(set(s.split()))
    N = len(pool)
    idf = {k: math.log(N / (1 + v)) for k, v in docs.items()}
    freq = pool.n_core.value_counts()
    d = pairs[["s1", "rid", score_col]].drop_duplicates(["s1", "rid"]).reset_index(drop=True)
    a = s1.n_core.reindex(d.s1.to_numpy()).fillna("").to_numpy()
    b = pool.n_core.reindex(d.rid.to_numpy()).fillna("").to_numpy()
    # best OTHER candidate of the S1 (by score_col): its name
    o = d.sort_values(["s1", score_col], ascending=[True, False])
    first = o.drop_duplicates("s1")
    second = o[~o.index.isin(first.index)].drop_duplicates("s1")
    top1 = pd.Series(first.rid.to_numpy(), index=first.s1.to_numpy())
    top2 = pd.Series(second.rid.to_numpy(), index=second.s1.to_numpy())
    t_rid = top1.reindex(d.s1.to_numpy()).to_numpy()
    t_alt = top2.reindex(d.s1.to_numpy()).to_numpy()
    t_rid = np.where(t_rid == d.rid.to_numpy(), t_alt, t_rid)
    ok = ~pd.isna(t_rid)
    top = np.full(len(d), "", dtype=object)
    top[ok] = pool.n_core.reindex(t_rid[ok].astype(np.int64)).fillna("").to_numpy()
    log_ = lambda m: print(time.strftime("[%H:%M:%S]"), m, flush=True)
    log_(f"name features for {len(d):,} {tag} pairs (idf over {N:,} records)")
    from joblib import Parallel, delayed
    ch = np.array_split(np.arange(len(d)), max(1, (os.cpu_count() or 2) * 2))

    def sub(c):                                   # only the IDF entries this chunk needs (keeps pickling small)
        toks = set()
        for x in a[c]:
            toks.update(x.split())
        for x in b[c]:
            toks.update(x.split())
        return {k: idf[k] for k in toks if k in idf}
    res = Parallel(n_jobs=-1, batch_size=1)(delayed(_chunk)((a[c], b[c], top[c], sub(c))) for c in ch)
    R = np.concatenate(res)
    f = pd.DataFrame(R, columns=["idf_jacc", "idf_shared", "idf_miss", "acro", "jw", "tsort", "psr", "top_sim"])
    f["name_freq"] = np.log1p(freq.reindex(b).fillna(0).to_numpy()).astype(np.float32)
    f.insert(0, "rid", d.rid.to_numpy()); f.insert(0, "s1", d.s1.to_numpy())
    f["n_tok_a"] = [len(x.split()) for x in a]; f["n_tok_b"] = [len(x.split()) for x in b]
    sa = s1.n_skel.reindex(d.s1.to_numpy()).fillna("").to_numpy(); sb = pool.n_skel.reindex(d.rid.to_numpy()).fillna("").to_numpy()
    f["skel_eq"] = ((sa == sb) & (sa != "")).astype(np.int8)
    f.to_parquet(path)
    log_(f"name features {tag} done ({time.time()-t0:.0f}s)")
    return f
