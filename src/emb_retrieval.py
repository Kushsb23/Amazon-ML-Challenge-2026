"""Supervised multilingual embedding retrieval: a third, complementary candidate generator.

Keys and TF-IDF miss 2.6% of true pairs. Most of them are name-only records (no address) whose name is shared by many
look-alikes, typo'd / reworded names, and native-script names. A fine-tuned multilingual sentence encoder searched in
BOTH directions (S1 -> record and record -> S1 of the same slice) finds many of them.

    python emb_retrieval.py train      # fine-tune intfloat/multilingual-e5-small (MIT) on true pairs of train slices m6..m9
    python emb_retrieval.py encode     # name view (N) and name+address view (F) for train S1 (m1..m5, holdout), pools, test
    python emb_retrieval.py search     # per slice and country, forward top-K + reverse top-R, both views
                                       # -> work/cand_extra_{split}_{subset}_emb.parquet (s1, rid, e_n, e_f)
    python emb_retrieval.py recall     # holdout recall of the old union vs old union + embeddings

The retriever never sees the slices the matcher trains on (m1..m5) or the holdout, so their candidates are unbiased.
Embeddings are stored int8 (unit vectors x 127) to keep disk use ~20 GB.
"""
import os
import sys
import time
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from blocking_keys import encode_id
from io_utils import read_source
import tfidf_knn as T

EDIR = os.path.join(WORK_DIR, "emb")
os.makedirs(EDIR, exist_ok=True)
BASE = os.environ.get("BER_EMB_MODEL", "intfloat/multilingual-e5-small")
MDIR = os.path.join(EDIR, "model")
RETR_SUBSETS = ["m6", "m7", "m8", "m9"]
TRAIN_S1_SUBSETS = ["m1", "m2", "m3", "m4", "m5", "holdout"]
LEN = {"n": 32, "f": 96, "a": 64}
VIEWS = os.environ.get("BER_EMB_VIEWS", "n,f").split(",")   # "n" = name only (fast), "f" = name + address
K_FWD = {"n": 20, "f": 20}
K_REV = {"n": 2, "f": 2}
REV_MARGIN = 0.05     # reverse hit kept if its similarity is within this of the S1's forward top-K cut-off
REV_CAP = 10          # at most this many reverse additions per S1 and view


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def text(view, name, addr):
    name = "" if name is None else str(name)
    if view == "n":
        return "query: " + name
    addr = "" if addr is None else str(addr)
    if view == "a":                       # address-only view (v3 retrieval)
        return "query: " + (addr if addr else "none")
    return "query: " + name + (" | " + addr if addr else "")


def read(split, k, subsets=None):
    r = read_source(split, k, ["entity_id", "business_name", "business_address", "country"])
    ids = encode_id(r.entity_id)
    d = pd.DataFrame({"id": ids, "name": r.business_name.to_numpy(dtype=object),
                      "addr": r.business_address.to_numpy(dtype=object), "country": r.country.astype(str).to_numpy()})
    if subsets:
        m = np.zeros(len(d), bool)
        for s in subsets:
            m |= T.subset_mask(ids, s)
        d = d[m].reset_index(drop=True)
    return d


# ------------------------------------------------------------------ model
def _pool(out, mask):
    import torch
    x = out.last_hidden_state * mask.unsqueeze(-1)
    x = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
    return torch.nn.functional.normalize(x.float(), dim=-1)


def stage_train():
    import torch
    from transformers import AutoTokenizer, AutoModel
    if os.path.exists(os.path.join(MDIR, "config.json")):
        log("retriever exists, skip"); return
    import union_trim as U
    g = pd.concat([U.gt_subset(s) for s in RETR_SUBSETS], ignore_index=True)
    g = g.sample(n=min(len(g), int(os.environ.get("BER_EMB_PAIRS", 1000000))), random_state=0).reset_index(drop=True)
    s1 = read("train", 1, RETR_SUBSETS).set_index("id")
    pool = pd.concat([read("train", k) for k in (2, 3)]).drop_duplicates("id").set_index("id")
    pool = pool.reindex(np.unique(g.rid.to_numpy()))
    A = s1.reindex(g.s1.to_numpy()); B = pool.reindex(g.rid.to_numpy())
    g["ctry"] = A.country.to_numpy()
    na, aa, nb, ab = A.name.to_numpy(), A.addr.to_numpy(), B.name.to_numpy(), B.addr.to_numpy()
    log(f"retriever training pairs {len(g):,}")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE)
    model.embeddings.word_embeddings.weight.requires_grad_(False)
    model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01)
    bs = 256
    # batches drawn within one country (harder in-batch negatives)
    rng = random.Random(0)
    batches = []
    for c, idx in g.groupby("ctry").groups.items():
        idx = list(idx); rng.shuffle(idx)
        batches += [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]
    rng.shuffle(batches)
    steps = len(batches)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    for step, ix in enumerate(batches):
        view = "n" if rng.random() < 0.5 else "f"
        qa = [text(view, na[j], aa[j]) for j in ix]
        qb = [text(view, nb[j], ab[j]) for j in ix]
        ea = tok(qa, truncation=True, max_length=LEN[view], padding=True, return_tensors="pt").to("cuda")
        eb = tok(qb, truncation=True, max_length=LEN[view], padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            za = _pool(model(**ea), ea["attention_mask"]); zb = _pool(model(**eb), eb["attention_mask"])
        sim = za @ zb.T / 0.05
        # S1 with several true records in one batch: mask the other positives out of the negatives
        s1v = g.s1.to_numpy()[ix]
        same = torch.tensor(s1v[:, None] == s1v[None, :], device="cuda")
        sim = sim.masked_fill(same & ~torch.eye(len(ix), dtype=torch.bool, device="cuda"), -1e4)
        lab = torch.arange(len(ix), device="cuda")
        loss = (torch.nn.functional.cross_entropy(sim, lab) + torch.nn.functional.cross_entropy(sim.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % 200 == 0:
            el = time.time() - t0
            log(f"  step {step+1}/{steps} loss {loss.item():.4f}  {el:.0f}s  eta {el/(step+1)*(steps-step-1)/60:.1f} min")
    model.save_pretrained(MDIR); tok.save_pretrained(MDIR)
    log(f"retriever trained ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ encode
def _encode(model, tok, texts, max_len, bs=1024):
    import torch
    L = np.array([len(t) for t in texts]); order = np.argsort(L)
    out = np.zeros((len(texts), model.config.hidden_size), np.int8)
    with torch.no_grad():
        for i in range(0, len(order), bs):
            ix = order[i:i + bs]
            e = tok([texts[j] for j in ix], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                z = _pool(model(**e), e["attention_mask"])
            out[ix] = torch.clamp(torch.round(z * 127), -127, 127).to(torch.int8).cpu().numpy()
    return out


def stage_encode():
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MDIR)
    model = AutoModel.from_pretrained(MDIR).cuda().half().eval()
    jobs = [("train", 1, TRAIN_S1_SUBSETS), ("train", 2, None), ("train", 3, None),
            ("test", 1, None), ("test", 2, None), ("test", 3, None)]
    for split, k, subs in jobs:
        meta = os.path.join(EDIR, f"{split}_s{k}_meta.parquet")
        todo = [v for v in VIEWS if not os.path.exists(os.path.join(EDIR, f"{split}_s{k}_{v}.npy"))
                or not os.path.exists(meta)]
        if not todo:
            log(f"{split} s{k}: exists"); continue
        t0 = time.time()
        d = read(split, k, subs)
        if os.path.exists(meta):                  # keep the row order of views already encoded
            m0 = pd.read_parquet(meta)
            assert len(m0) == len(d) and (m0.id.to_numpy() == d.id.to_numpy()).all()
        for v in todo:
            x = _encode(model, tok, [text(v, n, a) for n, a in zip(d.name, d.addr)], LEN[v])
            np.save(os.path.join(EDIR, f"{split}_s{k}_{v}.npy"), x)
        d[["id", "country"]].to_parquet(meta)
        log(f"{split} s{k}: {len(d):,} records, views {todo} ({time.time()-t0:.0f}s, {len(d)*len(todo)/(time.time()-t0):.0f} texts/s)")


# ------------------------------------------------------------------ search
def load_emb(split, k, v, mmap=False):
    meta = pd.read_parquet(os.path.join(EDIR, f"{split}_s{k}_meta.parquet"))
    x = np.load(os.path.join(EDIR, f"{split}_s{k}_{v}.npy"), mmap_mode="r" if mmap else None)
    return meta, x


def _topk(Q, P, k):
    """Q, P int8 numpy (unit x127). Returns (idx [nq,k], sim [nq,k]) by cosine, on GPU in fp16."""
    import torch
    Pt = torch.from_numpy(np.ascontiguousarray(P)).cuda().half().div_(127)
    k = min(k, len(P))
    chunk = int(max(64, min(8192, 1.5e9 / (2 * len(P)))))       # keep the score block ~1.5 GB
    I = np.zeros((len(Q), k), np.int64); S = np.zeros((len(Q), k), np.float32)
    for i in range(0, len(Q), chunk):
        q = torch.from_numpy(np.ascontiguousarray(Q[i:i + chunk])).cuda().half().div_(127)
        s, ix = torch.topk(q @ Pt.T, k, dim=1)
        I[i:i + chunk] = ix.cpu().numpy(); S[i:i + chunk] = s.float().cpu().numpy()
    del Pt
    torch.cuda.empty_cache()
    return I, S


def search_split(split, subsets):
    pm = []; px = {"n": [], "f": []}
    for k in (2, 3):
        pm.append(pd.read_parquet(os.path.join(EDIR, f"{split}_s{k}_meta.parquet")))
        for v in VIEWS:
            px[v].append(np.load(os.path.join(EDIR, f"{split}_s{k}_{v}.npy")))
    pm = pd.concat(pm, ignore_index=True); px = {v: np.concatenate(px[v]) for v in VIEWS}
    sm = pd.read_parquet(os.path.join(EDIR, f"{split}_s1_meta.parquet"))
    sx = {v: np.load(os.path.join(EDIR, f"{split}_s1_{v}.npy")) for v in VIEWS}
    for sb in subsets:
        out = os.path.join(WORK_DIR, f"cand_extra_{split}_{sb}_emb.parquet")
        if os.path.exists(out):
            log(f"{split}/{sb}: exists"); continue
        t0 = time.time()
        qm = T.subset_mask(sm.id.to_numpy(), sb)
        parts = []
        for c in sorted(set(sm.country[qm])):
            qi = np.flatnonzero(qm & (sm.country.to_numpy() == c))
            pi = np.flatnonzero(pm.country.to_numpy() == c)
            if not len(qi) or not len(pi):
                continue
            for v in VIEWS:
                I, S = _topk(sx[v][qi], px[v][pi], K_FWD[v])          # S1 -> records
                parts.append(pd.DataFrame({"s1": np.repeat(sm.id.to_numpy()[qi], I.shape[1]),
                                           "rid": pm.id.to_numpy()[pi][I.ravel()], "v": v, "sim": S.ravel()}))
                thr = S[:, -1] - REV_MARGIN                           # S1's own forward cut-off, relaxed
                I, S = _topk(px[v][pi], sx[v][qi], K_REV[v])          # records -> S1 of this slice
                qloc = I.ravel(); sim = S.ravel()
                keep = sim >= thr[qloc]                               # near-ties the forward list crowded out
                r = pd.DataFrame({"q": qloc[keep], "rid": np.repeat(pm.id.to_numpy()[pi], I.shape[1])[keep],
                                  "sim": sim[keep]})
                r = r.sort_values(["q", "sim"], ascending=[True, False])
                r = r[r.groupby("q").cumcount() < REV_CAP]
                parts.append(pd.DataFrame({"s1": sm.id.to_numpy()[qi][r.q.to_numpy()], "rid": r.rid.to_numpy(),
                                           "v": v, "sim": r.sim.to_numpy()}))
        c = pd.concat(parts, ignore_index=True)
        c = c.groupby(["s1", "rid", "v"], sort=False).sim.max().unstack("v").reset_index()
        c = c.rename(columns={"n": "e_n", "f": "e_f"}); c.columns.name = None
        for col in ("e_n", "e_f"):
            if col not in c:
                c[col] = np.nan
            c[col] = c[col].astype(np.float32)
        c.to_parquet(out)
        log(f"{split}/{sb}: {len(c):,} embedding candidates ({len(c)/max(1, qm.sum()):.1f}/S1, {time.time()-t0:.0f}s)")


def stage_search():
    search_split("train", TRAIN_S1_SUBSETS)
    search_split("test", [f"m{k}" for k in range(10)])


# ------------------------------------------------------------------ recall check
def stage_recall():
    import union_trim as U
    import evaluation as ev
    g = U.gt_subset("holdout")[["s1", "rid"]].drop_duplicates()
    n = U.n_s1("train", "holdout")
    old = pd.read_parquet(os.path.join(WORK_DIR, "union_train_holdout.parquet"), columns=["s1", "rid"])
    e = pd.read_parquet(os.path.join(WORK_DIR, "cand_extra_train_holdout_emb.parquet"), columns=["s1", "rid"])
    new = pd.concat([old, e]).drop_duplicates(["s1", "rid"])
    for name, c in (("old union", old), ("embeddings alone", e), ("old union + embeddings", new)):
        r = ev.recall_report_int(c, g, n)
        log(f"{name:25s} pairs/S1 {r['avg_cands_per_s1']:6.1f}  pair recall {r['pair_recall']:.4f}  "
            f"fully covered {r['entities_fully_covered']:.4f}  oracle F0.5 {r.get('oracle_macro_f05', float('nan')):.4f}")


if __name__ == "__main__":
    {"train": stage_train, "encode": stage_encode, "search": stage_search, "recall": stage_recall}[sys.argv[1]]()
