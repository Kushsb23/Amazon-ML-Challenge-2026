"""Architecture-v3 candidate generation on top of our pipeline.

Spaces (fine-tuned multilingual-e5, emb_retrieval.py): name (n), address (a), fused [n | a] (f).
Per 10% slice and country:
  forward  S1 -> records, top-20 per space
  backward record -> S1 of the slice, top-10 per space (kept rows: rank <= 3)
  + our generators: 9 blocking keys, 3 TF-IDF views
Union filter: keep a pair if >= 3 generator hits, or forward rank <= 2 in any space, or backward rank 1.
Every kept pair gets ~40 retrieval-time signals (ranks / scores per space and direction, key and TF-IDF hits,
exact cosines, CSLS hubness-corrected cosines, S1-side and record-side margins, counts, ranks) for prune.py.

    python v3_retrieval.py search        -> work/v3/sig_{split}_{subset}.parquet
    python v3_retrieval.py recall        -> holdout pair recall / candidates per S1 of the union
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
import tfidf_knn as T
import union_trim as U
from emb_retrieval import EDIR, _topk, TRAIN_S1_SUBSETS

VDIR = os.path.join(WORK_DIR, "v3")
os.makedirs(VDIR, exist_ok=True)
KF, KB, KB_KEEP = 20, 10, 3
BACK_DELTA = float(os.environ.get("BER_V3_BACK_DELTA", 0.03))
SPACES = ("n", "a", "f")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def _load(split, k):
    meta = pd.read_parquet(os.path.join(EDIR, f"{split}_s{k}_meta.parquet"))
    n = np.load(os.path.join(EDIR, f"{split}_s{k}_n.npy"))
    a = np.load(os.path.join(EDIR, f"{split}_s{k}_a.npy"))
    return meta, n, a


def _has_addr(split, ids):
    parts = []
    for k in (1, 2, 3):
        d = pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet"), columns=["entity_id", "a_empty"])
        parts.append(pd.Series(~d.a_empty.astype(bool).to_numpy(), index=U.encode_id(d.entity_id)))
    s = pd.concat(parts)
    s = s[~s.index.duplicated()]
    return s.reindex(ids).fillna(False).to_numpy()


def topk_chunked(Q, P, k, rows=2_000_000):
    """_topk over pool chunks (keeps GPU memory bounded for the 768-d fused space), merged exactly."""
    if len(P) <= rows:
        return _topk(Q, P, k)
    Is, Ss = [], []
    for s in range(0, len(P), rows):
        I, S = _topk(Q, P[s:s + rows], k)
        Is.append(I + s); Ss.append(S)
    I = np.concatenate(Is, 1); S = np.concatenate(Ss, 1)
    o = np.argsort(-S, 1)[:, :k]
    return np.take_along_axis(I, o, 1), np.take_along_axis(S, o, 1)


def topk3(Qn, Qa, Pn, Pa, k, rows=1_500_000):
    """Top-k in the name, address and fused spaces from ONE pair of matmuls per block (fused = (cos_n + cos_a) / 2).
    Q*, P* int8 (unit x127). Pool processed in chunks of `rows`, merged exactly. Returns {v: (I, S)}."""
    import torch
    out = {v: ([], []) for v in SPACES}
    for s in range(0, len(Pn), rows):
        pn = torch.from_numpy(np.ascontiguousarray(Pn[s:s + rows])).cuda().half().div_(127)
        pa = torch.from_numpy(np.ascontiguousarray(Pa[s:s + rows])).cuda().half().div_(127)
        kk = min(k, len(pn))
        chunk = int(max(64, min(8192, 6e8 / len(pn))))
        buf = {v: (np.zeros((len(Qn), kk), np.int64), np.zeros((len(Qn), kk), np.float32)) for v in SPACES}
        for i in range(0, len(Qn), chunk):
            qn = torch.from_numpy(np.ascontiguousarray(Qn[i:i + chunk])).cuda().half().div_(127)
            qa = torch.from_numpy(np.ascontiguousarray(Qa[i:i + chunk])).cuda().half().div_(127)
            sn_ = qn @ pn.T
            sv, ix = torch.topk(sn_, kk, dim=1); buf["n"][0][i:i + chunk] = ix.cpu().numpy(); buf["n"][1][i:i + chunk] = sv.float().cpu().numpy()
            sa_ = qa @ pa.T
            sv, ix = torch.topk(sa_, kk, dim=1); buf["a"][0][i:i + chunk] = ix.cpu().numpy(); buf["a"][1][i:i + chunk] = sv.float().cpu().numpy()
            sa_.add_(sn_).mul_(0.5); del sn_
            sv, ix = torch.topk(sa_, kk, dim=1); buf["f"][0][i:i + chunk] = ix.cpu().numpy(); buf["f"][1][i:i + chunk] = sv.float().cpu().numpy()
            del sa_, qn, qa
        for v in SPACES:
            out[v][0].append(buf[v][0] + s); out[v][1].append(buf[v][1])
        del pn, pa
        torch.cuda.empty_cache()
    res = {}
    for v in SPACES:
        I = np.concatenate(out[v][0], 1); S = np.concatenate(out[v][1], 1)
        if I.shape[1] > k:
            o = np.argsort(-S, 1)[:, :k]
            I, S = np.take_along_axis(I, o, 1), np.take_along_axis(S, o, 1)
        res[v] = (I, S)
    return res


def _cos(X, i, Y, j, chunk=2_000_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        a = X[i[s:s + chunk]].astype(np.float32); b = Y[j[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = (a * b).sum(1) / (127.0 * 127.0)
    return out


def search_split(split, subsets):
    t0 = time.time()
    sm, sn, sa = _load(split, 1)
    # pool vectors stay memory-mapped (10M x 384 x 2 views would not fit next to everything else in 31 GB)
    pm2 = pd.read_parquet(os.path.join(EDIR, f"{split}_s2_meta.parquet"))
    pm3 = pd.read_parquet(os.path.join(EDIR, f"{split}_s3_meta.parquet"))
    pm = pd.concat([pm2, pm3], ignore_index=True); n2 = len(pm2)
    PM = {v: [np.load(os.path.join(EDIR, f"{split}_s{k}_{v}.npy"), mmap_mode="r") for k in (2, 3)] for v in ("n", "a")}
    s_has = _has_addr(split, sm.id.to_numpy()); p_has = _has_addr(split, pm.id.to_numpy())
    sa[~s_has] = 0                                               # empty address -> zero vector
    div = {"n": 1.0, "a": 1.0, "f": 2.0}

    def prow(v, idx):
        """pool rows for sorted global indices idx"""
        i2 = idx[idx < n2]; i3 = idx[idx >= n2] - n2
        x = np.concatenate([np.asarray(PM[v][0][i2]), np.asarray(PM[v][1][i3])])
        if v == "a":
            x[~p_has[idx]] = 0
        return x

    def pgather(v, j):
        u, inv = np.unique(j, return_inverse=True)
        return prow(v, u)[inv]
    PCACHE = {}
    log(f"{split}: embeddings loaded ({time.time()-t0:.0f}s)")
    for sb in subsets:
        out = os.path.join(VDIR, f"sig_{split}_{sb}.parquet")
        if os.path.exists(out):
            log(f"{split}/{sb}: exists"); continue
        t1 = time.time()
        qm = T.subset_mask(sm.id.to_numpy(), sb)
        rows = []; r_s1 = {}; r_t = {}
        for c in sorted(set(sm.country[qm])):
            qi = np.flatnonzero(qm & (sm.country.to_numpy() == c))
            pi = np.flatnonzero(pm.country.to_numpy() == c)
            if not len(qi) or not len(pi):
                continue
            if (c, "n") not in PCACHE:                 # pool of this country: loaded once per split
                PCACHE[(c, "n")] = prow("n", pi); PCACHE[(c, "a")] = prow("a", pi)
            Pn, Pa = PCACHE[(c, "n")], PCACHE[(c, "a")]
            Qn, Qa = sn[qi], sa[qi]
            t2 = time.time()
            fwd = topk3(Qn, Qa, Pn, Pa, KF)
            bwd = topk3(Pn, Pa, Qn, Qa, KB)
            log(f"    {sb}/{c}: {len(qi):,} S1 x {len(pi):,} records searched ({time.time()-t2:.0f}s)")
            for v in SPACES:
                I, S = fwd[v]
                thr = S[:, -1].copy()                  # the S1's 20th forward score in this space
                rk = np.tile(np.arange(1, I.shape[1] + 1), len(qi))
                rows.append(pd.DataFrame({"s1": np.repeat(sm.id.to_numpy()[qi], I.shape[1]),
                                          "rid": pm.id.to_numpy()[pi][I.ravel()], "g": f"f{v}",
                                          "rank": rk.astype(np.int16), "score": S.ravel()}))
                r_s1.setdefault(v, []).append(pd.Series(S[:, :10].mean(1), index=sm.id.to_numpy()[qi]))
                I, S = bwd[v]
                r_t.setdefault(v, []).append(pd.Series(S.mean(1), index=pm.id.to_numpy()[pi]))
                I, S = I[:, :KB_KEEP], S[:, :KB_KEEP]
                rk = np.tile(np.arange(1, KB_KEEP + 1), len(pi))
                # a backward hit only matters near the S1's own forward frontier: pairs at or above the S1's 20th
                # forward score (features for pairs already found) and rank-1 hits just below it (new pairs).
                # Without this every record of the pool nominates an S1 of the slice (-> ~170 pairs / S1).
                t = thr[I.ravel()]; sc = S.ravel()
                kb = (sc >= t) | ((rk == 1) & (sc >= t - BACK_DELTA))
                rows.append(pd.DataFrame({"s1": sm.id.to_numpy()[qi][I.ravel()[kb]],
                                          "rid": np.repeat(pm.id.to_numpy()[pi], KB_KEEP)[kb], "g": f"b{v}",
                                          "rank": rk[kb].astype(np.int16), "score": sc[kb]}))
            del fwd, bwd
        emb = pd.concat(rows, ignore_index=True); del rows
        emb["g"] = emb.g.astype("category")
        wide = emb.groupby(["s1", "rid", "g"], sort=False, observed=True)[["rank", "score"]].min().unstack("g")
        wide.columns = [f"{a[0]}_{a[1]}" for a in wide.columns]
        wide = wide.reset_index(); del emb
        keys = U._read_cands("keys", split, sb, ["s1", "rid", "flags"])
        fl = keys["flags"].to_numpy().astype(np.int64)
        for b in U.KEY_FLAGS:                          # OR of flags per pair, vectorised (one column per bit)
            keys[f"b{b}"] = ((fl & b) > 0).astype(np.int64) * b
        keys = keys.drop(columns="flags").groupby(["s1", "rid"], sort=False).max()
        keys = keys.sum(axis=1).rename("kflags").reset_index()
        tf = U._read_cands("tfidf", split, sb)[["s1", "rid", "flags", "r_name", "r_addr", "r_rec"]]
        tf = tf.rename(columns={"flags": "tflags"})
        c = wide.merge(keys, on=["s1", "rid"], how="outer").merge(tf, on=["s1", "rid"], how="outer")
        del wide, keys, tf
        for g in [f"{d}{v}" for d in "fb" for v in SPACES]:
            for p in ("rank", "score"):
                if f"{p}_{g}" not in c:
                    c[f"{p}_{g}"] = np.nan
        c["kflags"] = c.kflags.fillna(0).astype(np.int64); c["tflags"] = c.tflags.fillna(0).astype(np.int64)
        for col in ("r_name", "r_addr", "r_rec"):
            c[col] = c[col].fillna(255).astype(np.float32)
        emb_hits = sum(c[f"rank_{d}{v}"].notna().astype(np.int8) for d in "fb" for v in SPACES)
        key_hits = sum(((c.kflags.to_numpy() & f) > 0).astype(np.int8) for f in U.KEY_FLAGS)
        tf_hits = sum(((c.tflags.to_numpy() & f) > 0).astype(np.int8) for f in U.TF_FLAGS)
        c["h_emb"] = emb_hits.astype(np.int8); c["h_key"] = key_hits.astype(np.int8); c["h_tf"] = tf_hits.astype(np.int8)
        c["hits"] = (c.h_emb + c.h_key + c.h_tf).astype(np.int8)
        c["min_fr"] = c[[f"rank_f{v}" for v in SPACES]].min(axis=1).fillna(99).astype(np.float32)
        c["min_br"] = c[[f"rank_b{v}" for v in SPACES]].min(axis=1).fillna(99).astype(np.float32)
        keep = (c.hits >= 3) | (c.min_fr <= 2) | (c.min_br <= 1) | (c.r_rec <= 1) | (c.r_name <= 1)
        n0 = len(c)
        c = c[keep].reset_index(drop=True)
        # exact cosines for every kept pair, CSLS hubness correction, margins
        spos = pd.Series(np.arange(len(sm)), index=sm.id.to_numpy()); ppos = pd.Series(np.arange(len(pm)), index=pm.id.to_numpy())
        i = spos.reindex(c.s1.to_numpy()).to_numpy(); j = ppos.reindex(c.rid.to_numpy()).to_numpy()
        c["cos_n"] = _cos(sn, i, pgather("n", j), np.arange(len(j)))
        c["cos_a"] = _cos(sa, i, pgather("a", j), np.arange(len(j)))
        c["cos_f"] = ((c.cos_n + c.cos_a) / 2).astype(np.float32)
        for v in SPACES:
            rs = pd.concat(r_s1[v]); rt = pd.concat(r_t[v])
            c[f"csls_{v}"] = (2 * c[f"cos_{v}"] - rs.reindex(c.s1.to_numpy()).fillna(0).to_numpy()
                              - rt.reindex(c.rid.to_numpy()).fillna(0).to_numpy()).astype(np.float32)
        c["cos_mean"] = ((c.cos_n + c.cos_a) / 2).astype(np.float32)
        c["t_has_addr"] = p_has[j].astype(np.int8); c["s_has_addr"] = s_has[i].astype(np.int8)
        c["src3"] = (c.rid.to_numpy() // 10 ** 12 == 3).astype(np.int8)
        for side, key in (("s", "s1"), ("t", "rid")):
            g = c.groupby(key, sort=False)
            c[f"{side}_n"] = g.rid.transform("size").astype(np.int16) if side == "s" else g.s1.transform("size").astype(np.int16)
            for v in ("f", "n"):
                col = f"cos_{v}"
                c[f"{side}_rank_{v}"] = g[col].rank(ascending=False, method="first").astype(np.int16)
                # margin over the best OTHER candidate on this side
                srt = c[[key, col]].sort_values([key, col], ascending=[True, False])
                k = srt[key].to_numpy(); vv = srt[col].to_numpy()
                st = np.r_[True, k[1:] != k[:-1]]; blk = np.cumsum(st) - 1; idx = np.flatnonzero(st)
                has2 = np.r_[k[1:] == k[:-1], False][idx]
                sec = np.where(has2, vv[np.minimum(idx + 1, len(vv) - 1)], -1.0)[blk]
                other = np.where(st, sec, vv[idx][blk])
                m = np.empty(len(c), np.float32); m[srt.index.to_numpy()] = vv - other
                c[f"{side}_mg_{v}"] = m
        c.to_parquet(out)
        log(f"{split}/{sb}: union {n0:,} -> kept {len(c):,} ({len(c)/max(1, qm.sum()):.1f}/S1) ({time.time()-t1:.0f}s)")


def stage_search():
    search_split("train", TRAIN_S1_SUBSETS)
    search_split("test", [f"m{k}" for k in range(10)])


def stage_recall():
    import evaluation as ev
    g = U.gt_subset("holdout")[["s1", "rid"]].drop_duplicates()
    c = pd.read_parquet(os.path.join(VDIR, "sig_train_holdout.parquet"), columns=["s1", "rid"])
    r = ev.recall_report_int(c, g, U.n_s1("train", "holdout"))
    log(f"v3 union holdout: {r['avg_cands_per_s1']:.1f}/S1, pair recall {r['pair_recall']:.4f}, "
        f"oracle F0.5 {r.get('oracle_macro_f05', float('nan')):.4f}")


if __name__ == "__main__":
    {"search": stage_search, "recall": stage_recall}[sys.argv[1]]()
