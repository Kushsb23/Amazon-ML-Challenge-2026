"""Cross-encoder re-scoring of the uncertain band of the stage-2 LightGBM scores.

The GBM sees engineered similarity features; a cross-encoder reads both raw records jointly
("name | address" </s></s> "name | address") and catches what features miss (native script vs
transliteration, heavy typos, re-ordered tokens, padded numbers). It is only run on the band where
the GBM is unsure, then a small stacker merges both opinions into one calibrated probability.

    band   : p2 in [LO, HI] and among the top-3 S1 of its S2/S3 record              -> work/ce/band_*.parquet
    train  : xlm-roberta-base (MIT) fine-tuned on band pairs of training slices m3..m5
             (+ a few easy pairs outside the band), noise augmentation               -> work/ce/model/
    score  : cross-encoder logits for band pairs of m1,m2 (stacker data), holdout, test (resumable chunks)
    stack  : LightGBM stacker on [logit p2, ce, rival p / ce, S1 context]; fitted on m1,m2 with 2-fold
             cross-fitting (OOF for the decision-rule tuning), full fit applied to holdout + test
             -> work/scores_{train_oof,train_holdout,test}_ce.parquet  (column p2 = stacked probability)

Then: BER_SCORES=ce BER_TUNE_SUBSETS=m1,m2 python postprocess.py tune   (holdout report)
      BER_SCORES=ce BER_OUT_DIR=../output_ce python postprocess.py write
The cross-encoder never sees m1, m2 or the holdout, so the stacker and the holdout report stay clean.
"""
import os
import sys
import json
import time
import random
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from blocking_keys import encode_id
from io_utils import read_source
import tfidf_knn as T

VER = os.environ.get("BER_CE_VER", "")          # "" = first version (work/ce), "2" = v2 (work/ce2)
CE_DIR = os.path.join(WORK_DIR, os.environ.get("BER_CE_DIRNAME", "ce" + VER))
BSUF = os.environ.get("BER_CE_BSUF", "b")            # stage-2 score files scores_*_{BSUF}.parquet
OUT_TAG = os.environ.get("BER_CE_OUT_TAG", "ce" + VER)
RIVAL_SLICE = os.environ.get("BER_CE_RIVAL_SLICE", "") == "1"   # rid competitors counted within the S1's 10% slice
STACK_NAME = "stacker" if OUT_TAG == "ce" + VER else f"stacker_{OUT_TAG}"
os.makedirs(CE_DIR, exist_ok=True)
MODEL = os.environ.get("BER_CE_MODEL", "FacebookAI/xlm-roberta-base")
LO, HI, TOPR = 0.02, 0.995, 3
CE_TRAIN_SUBSETS = ["m3", "m4", "m5"]
STACK_SUBSETS = ["m1", "m2"]
MAX_TRAIN = int(os.environ.get("BER_CE_MAX_TRAIN", 180000 if not VER else 600000))
PER_SLICE = bool(VER)       # v2: rank / rival context computed within each 10% slice, for train, holdout and test alike
FIELDS = bool(VER)          # v2: "name: .. | address: .." and field-aware, number-preserving noise
EPOCHS = float(os.environ.get("BER_CE_EPOCHS", 2))
MAX_LEN = 128


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def _in(ids, subsets):
    m = np.zeros(len(ids), bool)
    for s in subsets:
        m |= T.subset_mask(ids, s)
    return m


def _lgb_scores(fname):
    import pyarrow.parquet as pq
    path = os.path.join(WORK_DIR, fname)
    cols = [c for c in ("s1", "rid", "y", "p2_lgb") if c in pq.read_schema(path).names]
    d = pd.read_parquet(path, columns=cols)
    d["p"] = d.pop("p2_lgb").astype(np.float64)   # final stage-2 model (blend weight lgb = 1)
    return d.drop_duplicates(["s1", "rid"]).reset_index(drop=True)


def _context(d):
    """Per-pair context from the full candidate frame: rank / rival within the record, S1-level stats."""
    d = d.sort_values(["rid", "p"], ascending=[True, False]).reset_index(drop=True)
    d["r_rid"] = d.groupby("rid").cumcount() + 1
    first = d.groupby("rid").p.transform("max")
    second = _second_max(d, "rid")
    d["rival_p"] = np.where(d.r_rid == 1, second, first)
    d = d.sort_values(["s1", "p"], ascending=[True, False]).reset_index(drop=True)
    d["r_s1"] = d.groupby("s1").cumcount() + 1
    d["s1_best"] = d.groupby("s1").p.transform("max")
    d["s1_n50"] = (d.p >= 0.5).astype(np.float32).groupby(d.s1).transform("sum")
    d["s1_sum"] = d.groupby("s1").p.transform("sum")
    return d


def _second_max(d, key):
    # d sorted by key, p desc: second value of each block (0 if block has one row)
    k = d[key].to_numpy(); p = d.p.to_numpy()
    start = np.r_[True, k[1:] != k[:-1]]
    idx = np.flatnonzero(start)
    nxt = idx + 1
    has2 = np.r_[k[1:] == k[:-1], False][idx]
    val = np.where(has2, p[np.minimum(nxt, len(p) - 1)], 0.0)
    blk = np.cumsum(start) - 1
    return val[blk]


def context(d, subsets):
    """v1: one context over the whole frame. v2: context within each slice, so train, holdout and test see the
    same number of competing S1 per record (test is scored slice by slice upstream for the same reason)."""
    if not PER_SLICE:
        return _context(d)
    parts = []
    for sb in subsets:
        x = d[T.subset_mask(d.s1.to_numpy(), sb)]
        if len(x):
            parts.append(_context(x))
    return pd.concat(parts, ignore_index=True)


def band_of(d):
    return (d.p >= LO) & (d.p <= HI) & (d.r_rid <= TOPR)


# ------------------------------------------------------------------ raw text
def load_texts(split, ids):
    ids = np.unique(np.asarray(ids, np.int64))
    out = []
    for k in (1, 2, 3):
        r = read_source(split, k, ["entity_id", "business_name", "business_address"])
        e = encode_id(r.entity_id)
        m = np.isin(e, ids)
        out.append(pd.DataFrame({"id": e[m], "name": r.business_name.to_numpy(dtype=object)[m],
                                 "addr": r.business_address.to_numpy(dtype=object)[m]}))
        del r
    t = pd.concat(out, ignore_index=True).drop_duplicates("id").set_index("id")
    if FIELDS:
        return t
    t["txt"] = [f"{n} | {a}" if a else str(n) for n, a in zip(t.name, t.addr)]
    return t.txt


def side_ascii(n, a):
    """Architecture-v3 cross-encoder input: raw name, its transliteration, raw address."""
    from unidecode import unidecode
    n = "" if n is None else str(n); a = "" if a is None else str(a)
    return f"name: {n} | ascii: {unidecode(n).lower()} | address: {a if a else 'none'}"


LEGAL_GROUPS = [("pvt ltd", "private limited", "ltd", "limited"), ("llc",), ("inc", "corp", "corporation"), ("llp",),
                ("sarl", "sas", "sa", "eurl"), ("gmbh",), ("plc",), ("pc", "pllc"), ("lp",), ("co",)]
LEGAL = ["pvt ltd", "private limited", "ltd", "limited", "llc", "inc", "corp", "corporation", "llp", "co",
         "sarl", "sas", "sa", "eurl", "gmbh", "plc", "pc", "pllc", "lp"]


def synth_negative(n, a, rng):
    """A true record mutated into a near-copy decoy: other house number, or a non-equivalent legal form."""
    a = "" if a is None else str(a); n = "" if n is None else str(n)
    m = re.search(r"\d+", a)
    if m and rng.random() < 0.6:
        v = int(m.group()); w = v + rng.choice([1, 2, 3, 5, 10, 11, 20, 100]) * rng.choice([-1, 1])
        w = abs(w) if w != v else v + 1
        return n, a[:m.start()] + str(w) + a[m.end():]
    low = n.lower()
    for lf in LEGAL:
        if low.endswith(" " + lf):
            grp = next(g for g in LEGAL_GROUPS if lf in g)
            alt = rng.choice([x for g in LEGAL_GROUPS if g is not grp for x in g])   # a NON-equivalent form
            return n[: len(n) - len(lf)] + alt.title(), a
    return n + " " + rng.choice(["Holdings", "Group", "Trust", "Foundation", "Partners"]), a


def side(n, a):
    return f"name: {n} | address: {a}" if a else f"name: {n} | address: none"


def attach_text(split, d):
    txt = load_texts(split, np.r_[d.s1.to_numpy(), d.rid.to_numpy()])
    d = d.copy()
    if FIELDS:
        for col, key in (("a", d.s1.to_numpy()), ("b", d.rid.to_numpy())):
            d[f"n{col}"] = txt.name.reindex(key).fillna("").to_numpy()
            d[f"ad{col}"] = txt.addr.reindex(key).fillna("").to_numpy()
            d[f"t{col}"] = [side(n, a) for n, a in zip(d[f"n{col}"], d[f"ad{col}"])]
        return d
    d["ta"] = txt.reindex(d.s1.to_numpy()).fillna("").to_numpy()
    d["tb"] = txt.reindex(d.rid.to_numpy()).fillna("").to_numpy()
    return d


# ------------------------------------------------------------------ stage: band
def stage_band():
    t0 = time.time()
    oof = context(_lgb_scores("scores_train_oof_" + BSUF + ".parquet"), ["m1", "m2", "m3", "m4", "m5"])
    b = oof[band_of(oof)]
    tr_m = _in(oof.s1.to_numpy(), CE_TRAIN_SUBSETS)
    trb = b[_in(b.s1.to_numpy(), CE_TRAIN_SUBSETS)]
    outside = oof[tr_m & ~band_of(oof)].sample(n=min(int(0.08 * len(trb)), int((tr_m & ~band_of(oof)).sum())),
                                              random_state=0)
    tr = pd.concat([trb, outside])
    if VER in ("3", "4"):
        tr = tr.iloc[:0]
    if len(tr) > MAX_TRAIN:
        tr = tr.sample(n=MAX_TRAIN, random_state=1)
    log(f"train band pairs {len(trb):,} + outside {len(outside):,} -> {len(tr):,} (pos {tr.y.mean():.3f})")
    if len(tr):
        attach_text("train", tr[["s1", "rid", "y"]]).to_parquet(os.path.join(CE_DIR, "train_pairs.parquet"))
    st = oof[_in(oof.s1.to_numpy(), STACK_SUBSETS)]
    st.to_parquet(os.path.join(CE_DIR, "ctx_stack.parquet"))
    sb = st[band_of(st)]
    log(f"stack slices {STACK_SUBSETS}: {len(st):,} pairs, band {len(sb):,} ({len(sb)/len(st):.1%}, pos {sb.y.mean():.3f})")
    attach_text("train", sb[["s1", "rid"]]).to_parquet(os.path.join(CE_DIR, "band_stack.parquet"))
    del oof
    ho = context(_lgb_scores("scores_train_holdout_" + BSUF + ".parquet"), ["holdout"])
    ho.to_parquet(os.path.join(CE_DIR, "ctx_holdout.parquet"))
    hb = ho[band_of(ho)]
    log(f"holdout: {len(ho):,} pairs, band {len(hb):,} ({len(hb)/len(ho):.1%}, pos {hb.y.mean():.3f}; "
        f"positives in band {hb.y.sum():,} of {ho.y.sum():,})")
    attach_text("train", hb[["s1", "rid"]]).to_parquet(os.path.join(CE_DIR, "band_holdout.parquet"))
    del ho
    te = context(_lgb_scores("scores_test_" + BSUF + ".parquet"), [f"m{k}" for k in range(10)])
    te.to_parquet(os.path.join(CE_DIR, "ctx_test.parquet"))
    tb = te[band_of(te)]
    log(f"test: {len(te):,} pairs, band {len(tb):,} ({len(tb)/len(te):.1%})")
    attach_text("test", tb[["s1", "rid"]]).to_parquet(os.path.join(CE_DIR, "band_test.parquet"))
    log(f"band done ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ cross-encoder
LEET = {"o": "0", "l": "1", "e": "3", "a": "4", "s": "5", "i": "1"}


def noise(s, rng, p):
    if not s or rng.random() > p:
        return s
    op = rng.randrange(6)
    if op == 0 and len(s) > 4:                       # char typo
        i = rng.randrange(1, len(s) - 2)
        k = rng.randrange(3)
        s = s[:i] + s[i + 1:] if k == 0 else (s[:i] + s[i] + s[i:] if k == 1 else s[:i] + s[i + 1] + s[i] + s[i + 2:])
    elif op == 1:
        s = s.upper() if rng.random() < 0.6 else s.lower()
    elif op == 2:
        c = [i for i, ch in enumerate(s) if ch.lower() in LEET]
        if c:
            i = rng.choice(c); s = s[:i] + LEET[s[i].lower()] + s[i + 1:]
    elif op == 3:
        w = s.split(" ")
        if len(w) > 3:
            w.pop(rng.randrange(len(w))); s = " ".join(w)
    elif op == 4:
        w = s.split(" ")
        if len(w) > 2:
            i = rng.randrange(len(w) - 1); w[i], w[i + 1] = w[i + 1], w[i]; s = " ".join(w)
    else:
        s = re.sub(r"\s+", " ", s.replace(",", " ").replace(".", " ")).strip()
    return s


def noise_addr(s, rng, p):
    """Address noise that never touches digits: identity evidence (house / unit numbers, postcodes) stays intact."""
    if not s or rng.random() > p:
        return s
    op = rng.randrange(4)
    w = s.split(" ")
    if op == 0:
        c = [i for i in range(1, len(s) - 2) if not (s[i].isdigit() or s[i + 1].isdigit())]
        if c and len(s) > 4:
            i = rng.choice(c); s = s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i + 1] + s[i] + s[i + 2:]
    elif op == 1:
        s = s.upper() if rng.random() < 0.6 else s.lower()
    elif op == 2:
        c = [i for i, t in enumerate(w) if t and not any(ch.isdigit() for ch in t)]
        if len(w) > 3 and c:
            w.pop(rng.choice(c)); s = " ".join(w)
    else:
        s = re.sub(r"\s+", " ", s.replace(",", " ").replace(".", " ")).strip()
    return s


def stage_train():
    import torch
    from torch import nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    out = os.environ.get("BER_CE_OUT", os.path.join(CE_DIR, "model"))
    if os.path.exists(os.path.join(out, "config.json")):
        log("model exists, skip"); return
    d = pd.read_parquet(os.environ.get("BER_CE_PAIRS", os.path.join(CE_DIR, "train_pairs.parquet")))
    if os.environ.get("BER_CE_MAX_PAIRS"):
        d = d.sample(n=min(len(d), int(os.environ["BER_CE_MAX_PAIRS"])), random_state=0).reset_index(drop=True)
    log(f"training on {len(d):,} pairs, {EPOCHS} epochs, device {torch.cuda.get_device_name(0)}")
    rng = random.Random(0); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    model.roberta.embeddings.word_embeddings.weight.requires_grad_(False)   # 70% of params, keeps optimiser small
    model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(os.environ.get("BER_CE_LR", 2e-5)), weight_decay=0.01)
    bs = int(os.environ.get("BER_CE_BS", 32))
    n = len(d); steps = int(EPOCHS * ((n + bs - 1) // bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss()
    SIDE = side_ascii if os.environ.get("BER_CE_ASCII") == "1" else side
    synth = float(os.environ.get("BER_CE_SYNTH", 0))
    if synth > 0 and FIELDS:
        pos = d[d.y == 1].sample(frac=min(1.0, synth * len(d) / max(1, (d.y == 1).sum())), random_state=2)
        mut = [synth_negative(nb, ab, rng) for nb, ab in zip(pos.nb, pos.adb)]
        neg = pos.assign(nb=[m[0] for m in mut], adb=[m[1] for m in mut], y=0)
        d = pd.concat([d, neg], ignore_index=True).sample(frac=1.0, random_state=3).reset_index(drop=True)
        n = len(d); steps = int(EPOCHS * ((n + bs - 1) // bs))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
        log(f"  + {len(neg):,} synthetic decoy negatives -> {n:,} pairs, {steps} steps")
    Y = d.y.to_numpy().astype(np.float32)
    if FIELDS:
        NA, AA, NB, AB = d.na.tolist(), d.ada.tolist(), d.nb.tolist(), d.adb.tolist()
    else:
        A, B = d.ta.tolist(), d.tb.tolist()
    step, t0 = 0, time.time()
    while step < steps:
        order = list(range(n)); rng.shuffle(order)
        for i in range(0, n, bs):
            if step >= steps:
                break
            ix = order[i:i + bs]
            if FIELDS:
                a = [SIDE(noise(NA[j], rng, 0.25), noise_addr(AA[j], rng, 0.15)) for j in ix]
                b = [SIDE(noise(NB[j], rng, 0.25), noise_addr(AB[j], rng, 0.15)) for j in ix]
            else:
                a = [noise(A[j], rng, 0.2) for j in ix]; b = [noise(B[j], rng, 0.2) for j in ix]
            if rng.random() < 0.5:
                a, b = b, a                           # symmetric
            enc = tok(a, b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor(Y[ix], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(**enc).logits.squeeze(-1)
            loss = lossf(lo.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            ema = loss.item() if step == 1 else 0.99 * ema + 0.01 * loss.item()
            if step == int(os.environ.get("BER_CE_GUARD_STEP", 2000)) and ema > 0.6:
                log(f"  training collapsed (loss EMA {ema:.3f} at step {step}) -> abort")
                sys.exit(3)
            if step % 500 == 0:
                el = time.time() - t0
                log(f"  step {step}/{steps} loss {loss.item():.4f}  {el:.0f}s  eta {el/step*(steps-step)/60:.1f} min")
    model.save_pretrained(out); tok.save_pretrained(out)
    if SIDE is side_ascii:
        open(os.path.join(out, "ascii.flag"), "w").write("x")
    log(f"train done ({time.time()-t0:.0f}s)")


LARGE_DIR = os.environ.get("BER_CE_SECOND", os.path.join(WORK_DIR, "ce_large"))   # second cross-encoder
B_DIR = os.path.join(WORK_DIR, "ce2")


def score_texts(model_dir, A, B, bs=256, label=""):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().half().eval()
    L = np.array([len(x) + len(y) for x, y in zip(A, B)]); order = np.argsort(L)
    res = np.zeros(len(A), np.float32); t0 = time.time()
    with torch.no_grad():
        for n, i in enumerate(range(0, len(order), bs)):
            ix = order[i:i + bs]
            enc = tok([A[j] for j in ix], [B[j] for j in ix], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to("cuda")
            res[ix] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if n % 1000 == 999:
                log(f"  {label}: {i + bs:,}/{len(A):,}  {(i + bs) / (time.time() - t0):.0f} pairs/s")
    del model
    torch.cuda.empty_cache()
    return res


def cached_scores(keys, A, B, model_dir, cache_path, label):
    """Scores for (s1, rid) keys; pairs already in the cache parquet (s1, rid, score) are reused, the rest scored
    and appended to the cache."""
    cache = pd.read_parquet(cache_path) if os.path.exists(cache_path) else pd.DataFrame(
        {"s1": np.array([], np.int64), "rid": np.array([], np.int64), "score": np.array([], np.float32)})
    m = keys[["s1", "rid"]].merge(cache.drop_duplicates(["s1", "rid"]), on=["s1", "rid"], how="left")
    miss = m.score.isna().to_numpy()
    log(f"{label}: {len(m):,} pairs, {int(miss.sum()):,} to score ({miss.mean():.1%})")
    if miss.any():
        ix = np.flatnonzero(miss)
        sc = score_texts(model_dir, [A[i] for i in ix], [B[i] for i in ix], label=label)
        m.loc[miss, "score"] = sc
        new = pd.DataFrame({"s1": m.s1.to_numpy()[ix], "rid": m.rid.to_numpy()[ix], "score": sc})
        pd.concat([cache, new], ignore_index=True).to_parquet(cache_path)
    return m.score.to_numpy(np.float32)


def stage_prescore_large():
    """Large model: score the existing (v2) band pairs into its cache while the candidate rebuild runs on CPU."""
    os.makedirs(LARGE_DIR, exist_ok=True)
    for split, bands in (("train", ["stack", "holdout"]), ("test", ["test"])):
        for w in bands:
            d = pd.read_parquet(os.path.join(B_DIR, f"band_{w}.parquet"), columns=["s1", "rid", "ta", "tb"])
            cached_scores(d, d.ta.tolist(), d.tb.tolist(), os.path.join(LARGE_DIR, "model"),
                          os.path.join(LARGE_DIR, f"cache_{split}.parquet"), f"large/{w}")


def _b_cache(split):
    """Seed a cache for model B from its v2 score files."""
    p = os.path.join(CE_DIR, f"cacheB_{split}.parquet")
    if not os.path.exists(p):
        ws = ["stack", "holdout"] if split == "train" else ["test"]
        pd.concat([pd.read_parquet(os.path.join(B_DIR, f"ce_{w}.parquet")).rename(columns={"ce": "score"})
                   for w in ws], ignore_index=True).drop_duplicates(["s1", "rid"]).to_parquet(p)
    return p


def stage_score_v3(which=("stack", "holdout", "test")):
    for w in which:
        dst = os.path.join(CE_DIR, f"ce_{w}.parquet")
        if os.path.exists(dst):
            log(f"{w}: exists, skip"); continue
        split = "test" if w == "test" else "train"
        d = pd.read_parquet(os.path.join(CE_DIR, f"band_{w}.parquet"),
                            columns=["s1", "rid", "ta", "tb", "na", "ada", "nb", "adb"])
        A, B = d.ta.tolist(), d.tb.tolist()
        ce = cached_scores(d, A, B, os.path.join(B_DIR, "model"), _b_cache(split), f"B/{w}")
        if os.path.exists(os.path.join(LARGE_DIR, "model", "ascii.flag")):
            A = [side_ascii(n, a) for n, a in zip(d.na, d.ada)]; B = [side_ascii(n, a) for n, a in zip(d.nb, d.adb)]
        if os.path.exists(LARGE_OFF) or (w == "test" and not large_helps()):
            cel = ce.copy()                     # second model not used: skip its (slow) test scoring
        else:
            cel = cached_scores(d, A, B, os.path.join(LARGE_DIR, "model"),
                                os.path.join(LARGE_DIR, f"cache_{split}.parquet"), f"large/{w}")
        pd.DataFrame({"s1": d.s1.to_numpy(), "rid": d.rid.to_numpy(), "ce": ce, "cel": cel}).to_parquet(dst)


LARGE_OFF = os.path.join(CE_DIR, "large_off")


def large_helps():
    """Cross-fitted band AUC on m1-m2 with and without the second cross-encoder (decided once, before test scoring)."""
    if os.path.exists(LARGE_OFF):
        return False
    flag_on = os.path.join(CE_DIR, "large_on")
    if os.path.exists(flag_on):
        return True
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    ctx = pd.read_parquet(os.path.join(CE_DIR, "ctx_stack.parquet"))
    st = _ens_large(_extras(_stack_frame(ctx, pd.read_parquet(os.path.join(CE_DIR, "ce_stack.parquet"))),
                            "train", STACK_SUBSETS, "stack"))
    y = st.y.to_numpy(); fold = (st.s1.to_numpy() // 7) % 2
    auc = {}
    for name, feats in (("base", SFEATS), ("large", SFEATS + LARGE_FEATS)):
        X = st[feats].to_numpy(np.float32); q = np.zeros(len(st))
        for h in (0, 1):
            m = lgb.train(_lgb_params(), lgb.Dataset(X[fold != h], y[fold != h]), 400)
            q[fold == h] = m.predict(X[fold == h])
        auc[name] = roc_auc_score(y, q)
    on = auc["large"] > auc["base"] + 2e-4
    log(f"second cross-encoder on m1-m2: AUC alone {roc_auc_score(y, st.cel):.4f} (B {roc_auc_score(y, st.ce):.4f}); "
        f"stacked base {auc['base']:.5f} vs with it {auc['large']:.5f} -> {'use it' if on else 'drop it'}")
    open(flag_on if on else LARGE_OFF, "w").write("x")
    return on


def stage_score(which=("stack", "holdout", "test")):
    if VER in ("3", "4"):
        return stage_score_v3(which)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    md = os.path.join(CE_DIR, "model")
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForSequenceClassification.from_pretrained(md).cuda().half().eval()
    CH = 200_000
    for w in which:
        dst = os.path.join(CE_DIR, f"ce_{w}.parquet")
        if os.path.exists(dst):
            log(f"{w}: exists, skip"); continue
        d = pd.read_parquet(os.path.join(CE_DIR, f"band_{w}.parquet"))
        t0 = time.time(); parts = []
        for c0 in range(0, len(d), CH):
            cp = os.path.join(CE_DIR, f"ce_{w}_{c0 // CH:04d}.npy")
            if os.path.exists(cp):
                parts.append(np.load(cp)); continue
            a = d.ta.to_numpy()[c0:c0 + CH]; b = d.tb.to_numpy()[c0:c0 + CH]
            L = np.array([len(x) + len(y) for x, y in zip(a, b)]); order = np.argsort(L)
            res = np.zeros(len(a), np.float32)
            with torch.no_grad():
                for i in range(0, len(a), 256):
                    ix = order[i:i + 256]
                    enc = tok(list(a[ix]), list(b[ix]), truncation=True, max_length=MAX_LEN, padding=True,
                              return_tensors="pt").to("cuda")
                    res[ix] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            np.save(cp, res); parts.append(res)
            el = time.time() - t0; done = min(c0 + CH, len(d))
            log(f"  {w}: {done:,}/{len(d):,}  {done/el:.0f} pairs/s")
        pd.DataFrame({"s1": d.s1.to_numpy(), "rid": d.rid.to_numpy(), "ce": np.concatenate(parts)}).to_parquet(dst)
        log(f"{w}: scored {len(d):,} ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ stacker
SFEATS = ["lp", "ce", "ce_sig_minus_p", "rival_p", "rival_ce", "ce_margin_rid", "r_rid", "r_s1", "s1_best",
          "p_rel", "s1_n50", "s1_sum", "ce_rank_s1", "ce_best_s1", "ce_gap_s1", "n_band_s1"]
# v2: structured evidence from the candidate stage (address availability, numbers, name ambiguity)
XFEATS = ["b_empty", "a_empty", "num_eq", "num_diff", "num_sub", "nums_shared", "name_eq", "city_eq", "street_eq",
          "c_name", "c_addr", "c_rec", "s1_n", "rid_n"]
RAW_FEATS = ["r_ratio_case", "r_ratio_ascii", "r_jac3", "r_monge", "r_legal_only", "r_script_mm", "r_leet_mm",
             "r_a_empty_any", "r_digit_diff", "r_zero_pad", "r_hn_gap", "r_a_ratio", "r_a_tset", "r_len_ratio"]
if VER:
    SFEATS = SFEATS + XFEATS
if VER == "4":
    SFEATS = SFEATS + RAW_FEATS


def raw_features(na, ada, nb, adb):
    """Raw-string evidence of a pair (architecture-v3 stacker inputs)."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    from unidecode import unidecode
    legal = set(w for lf in LEGAL for w in lf.split())
    out = {k: np.zeros(len(na), np.float32) for k in RAW_FEATS}

    def core(x):
        return " ".join(w for w in re.sub(r"[^a-z0-9 ]", " ", x.lower()).split() if w not in legal)

    for i, (n1, a1, n2, a2) in enumerate(zip(na, ada, nb, adb)):
        n1 = n1 or ""; n2 = n2 or ""; a1 = a1 or ""; a2 = a2 or ""
        u1, u2 = unidecode(n1).lower(), unidecode(n2).lower()
        out["r_ratio_case"][i] = fuzz.ratio(n1, n2)
        out["r_ratio_ascii"][i] = fuzz.ratio(u1, u2)
        g1 = {u1[j:j + 3] for j in range(max(1, len(u1) - 2))}; g2 = {u2[j:j + 3] for j in range(max(1, len(u2) - 2))}
        out["r_jac3"][i] = len(g1 & g2) / max(1, len(g1 | g2))
        t1, t2 = u1.split(), u2.split()
        out["r_monge"][i] = np.mean([max(JaroWinkler.similarity(x, y) for y in t2) for x in t1]) if t1 and t2 else 0
        c1, c2 = core(u1), core(u2)
        out["r_legal_only"][i] = float(c1 == c2 and u1.strip() != u2.strip())
        out["r_script_mm"][i] = float(any(ord(ch) > 127 for ch in n1) != any(ord(ch) > 127 for ch in n2))
        out["r_leet_mm"][i] = float(any(ch.isdigit() for ch in n1) != any(ch.isdigit() for ch in n2))
        out["r_a_empty_any"][i] = float(not a1 or not a2)
        d1, d2 = re.findall(r"\d+", a1), re.findall(r"\d+", a2)
        s1, s2 = set(d1), set(d2)
        out["r_digit_diff"][i] = len(s1 ^ s2)
        out["r_zero_pad"][i] = float(s1 != s2 and {x.lstrip("0") for x in s1} == {x.lstrip("0") for x in s2})
        out["r_hn_gap"][i] = min(abs(int(d1[0]) - int(d2[0])), 1e4) if d1 and d2 and len(d1[0]) < 9 and len(d2[0]) < 9 else -1
        out["r_a_ratio"][i] = fuzz.ratio(unidecode(a1).lower(), unidecode(a2).lower()) if a1 and a2 else -1
        out["r_a_tset"][i] = fuzz.token_set_ratio(unidecode(a1).lower(), unidecode(a2).lower()) if a1 and a2 else -1
        out["r_len_ratio"][i] = min(len(u1), len(u2)) / max(1, len(u1), len(u2))
    return pd.DataFrame(out)


def _extras(d, split, subsets, band_name):
    """Join candidate-stage features (cand_{split}_{subset}.parquet) onto band pairs d (s1, rid)."""
    cols = ["s1", "rid"] + [c for c in XFEATS if c != "a_empty"]
    keys = d[["s1", "rid"]]
    parts = []
    for sb in subsets:
        k = keys[T.subset_mask(keys.s1.to_numpy(), sb)]
        if not len(k):
            continue
        c = pd.read_parquet(os.path.join(WORK_DIR, f"cand_{split}_{sb}.parquet"), columns=cols)
        parts.append(k.merge(c.drop_duplicates(["s1", "rid"]), on=["s1", "rid"], how="left"))
    x = pd.concat(parts, ignore_index=True)
    d = d.merge(x, on=["s1", "rid"], how="left")
    ad = pd.read_parquet(os.path.join(CE_DIR, f"band_{band_name}.parquet"), columns=["s1", "rid", "ada"])
    d = d.merge(ad.assign(a_empty=(ad.ada.fillna("").str.strip() == "").astype(np.int8)).drop(columns="ada"),
                on=["s1", "rid"], how="left")
    if VER == "4":
        cache = os.path.join(CE_DIR, f"raw_{band_name}.parquet")
        if os.path.exists(cache):
            rf = pd.read_parquet(cache)
        else:
            t0 = time.time()
            b = pd.read_parquet(os.path.join(CE_DIR, f"band_{band_name}.parquet"),
                                columns=["s1", "rid", "na", "ada", "nb", "adb"])
            from joblib import Parallel, delayed
            ch = np.array_split(np.arange(len(b)), max(1, os.cpu_count() or 1))
            parts = Parallel(n_jobs=-1)(delayed(raw_features)(b.na.to_numpy()[c], b.ada.to_numpy()[c],
                                                              b.nb.to_numpy()[c], b.adb.to_numpy()[c]) for c in ch)
            rf = pd.concat(parts, ignore_index=True)
            rf.insert(0, "rid", b.rid.to_numpy()); rf.insert(0, "s1", b.s1.to_numpy())
            rf.to_parquet(cache)
            log(f"  raw-string features for {len(b):,} {band_name} pairs ({time.time()-t0:.0f}s)")
        d = d.merge(rf, on=["s1", "rid"], how="left")
    return d


def _stack_frame(ctx, ce):
    d = ctx.merge(ce, on=["s1", "rid"], how="inner")
    pc = np.clip(d.p.to_numpy(), 1e-5, 1 - 1e-5)
    d["lp"] = np.log(pc / (1 - pc))
    d["ce_sig_minus_p"] = 1 / (1 + np.exp(-d.ce.to_numpy())) - d.p.to_numpy()
    d["_rk"] = _rkey(d)
    d = d.sort_values(["_rk", "ce"], ascending=[True, False]).reset_index(drop=True)
    first = d.groupby("_rk").cumcount() == 0
    k = d._rk.to_numpy(); cev = d.ce.to_numpy()
    start = np.r_[True, k[1:] != k[:-1]]; idx = np.flatnonzero(start)
    has2 = np.r_[k[1:] == k[:-1], False][idx]
    sec = np.where(has2, cev[np.minimum(idx + 1, len(cev) - 1)], -10.0)[np.cumsum(start) - 1]
    mx = d.groupby("_rk").ce.transform("max").to_numpy()
    d["rival_ce"] = np.where(first, sec, mx)
    d["ce_margin_rid"] = d.ce - d.rival_ce
    d["p_rel"] = d.p / d.s1_best.clip(lower=1e-6)
    g = d.groupby("s1").ce
    d["ce_rank_s1"] = g.rank(ascending=False, method="first")
    d["ce_best_s1"] = g.transform("max")
    d["ce_gap_s1"] = d.ce - d.ce_best_s1
    d["n_band_s1"] = g.transform("size")
    return d.drop(columns="_rk")


def _rkey(d):
    """Competition key for a record: the record itself, or (record, S1 slice) with BER_CE_RIVAL_SLICE=1 so that
    stack (m1+m2), holdout (one slice) and test (ten slices) all see competitors from one 10% slice only."""
    r = d.rid.to_numpy().astype(np.int64)
    if not RIVAL_SLICE:
        return r
    g = int(os.environ.get("BER_CE_RIVAL_GROUP", 1))       # slices pooled per competition group (test density match)
    return r * 10 + ((d.s1.to_numpy().astype(np.int64) % 10 ** 12) % 10) // g


def _lgb_params():
    return dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


V1_DIR = os.path.join(WORK_DIR, "ce")
ENS_FEATS = ["ce1", "ce1_margin_rid", "cem", "cem_margin_rid", "ce_agree"]


def _rival(d, col):
    """Best other value of `col` among the S1 competing for the same record (-10 if none)."""
    rk = _rkey(d)
    o = np.lexsort((-d[col].to_numpy(), rk))
    k = rk[o]; v = d[col].to_numpy()[o]
    start = np.r_[True, k[1:] != k[:-1]]; blk = np.cumsum(start) - 1; idx = np.flatnonzero(start)
    has2 = np.r_[k[1:] == k[:-1], False][idx]
    sec = np.where(has2, v[np.minimum(idx + 1, len(v) - 1)], -10.0)[blk]
    r = np.where(start, sec, v[idx][blk])
    out = np.empty(len(d)); out[o] = r
    return out


def _score_v1(texts_a, texts_b):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    md = os.path.join(V1_DIR, "model")
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForSequenceClassification.from_pretrained(md).cuda().half().eval()
    L = np.array([len(x) + len(y) for x, y in zip(texts_a, texts_b)]); order = np.argsort(L)
    res = np.zeros(len(texts_a), np.float32)
    with torch.no_grad():
        for i in range(0, len(order), 256):
            ix = order[i:i + 256]
            enc = tok([texts_a[j] for j in ix], [texts_b[j] for j in ix], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to("cuda")
            res[ix] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return res


def _ce1(w):
    """First cross-encoder's logits for this version's band of `w` (cached). Pairs the first version never
    scored (the band moved with the per-slice context) are scored now with the first model."""
    dst = os.path.join(CE_DIR, f"ce1_{w}.parquet")
    if os.path.exists(dst):
        return pd.read_parquet(dst)
    b = pd.read_parquet(os.path.join(CE_DIR, f"band_{w}.parquet"), columns=["s1", "rid", "na", "ada", "nb", "adb"])
    old = pd.read_parquet(os.path.join(V1_DIR, f"ce_{w}.parquet")).rename(columns={"ce": "ce1"})
    m = b.merge(old.drop_duplicates(["s1", "rid"]), on=["s1", "rid"], how="left")
    miss = m.ce1.isna().to_numpy()
    if miss.any():
        t0 = time.time()
        mm = m[miss]
        v1 = lambda n, a: f"{n} | {a}" if a else str(n)
        m.loc[miss, "ce1"] = _score_v1([v1(n, a) for n, a in zip(mm.na, mm.ada)], [v1(n, a) for n, a in zip(mm.nb, mm.adb)])
        log(f"  {w}: scored {int(miss.sum()):,} pairs with the first cross-encoder ({time.time()-t0:.0f}s)")
    m = m[["s1", "rid", "ce1"]]
    m.to_parquet(dst)
    return m


def _ens(d, w):
    d = d.merge(_ce1(w), on=["s1", "rid"], how="left")
    d["ce1_margin_rid"] = d.ce1 - _rival(d, "ce1")
    d["cem"] = (d.ce + d.ce1) / 2
    d["cem_margin_rid"] = d.cem - _rival(d, "cem")
    d["ce_agree"] = d.ce - d.ce1
    return d


LARGE_FEATS = ["cel", "cel_margin_rid", "cem2", "cem2_margin_rid", "cel_minus_ce", "cel_rank_s1", "cel_gap_s1"]


def _ens_large(d):
    d = d.copy()
    d["cel_margin_rid"] = d.cel - _rival(d, "cel")
    d["cem2"] = (d.ce + d.cel) / 2
    d["cem2_margin_rid"] = d.cem2 - _rival(d, "cem2")
    d["cel_minus_ce"] = d.cel - d.ce
    g = d.groupby("s1").cel
    d["cel_rank_s1"] = g.rank(ascending=False, method="first")
    d["cel_gap_s1"] = d.cel - g.transform("max")
    return d


def stage_stack():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    ctx = pd.read_parquet(os.path.join(CE_DIR, "ctx_stack.parquet"))
    st = _stack_frame(ctx, pd.read_parquet(os.path.join(CE_DIR, "ce_stack.parquet")))
    variants = {"base": SFEATS}
    if VER:
        st = _extras(st, "train", STACK_SUBSETS, "stack")
        if VER == "2" and os.path.exists(os.path.join(V1_DIR, "model", "config.json")):
            st = _ens(st, "stack")
            variants["ens"] = SFEATS + ENS_FEATS
        if VER in ("3", "4"):
            st = _ens_large(st)
            if not os.path.exists(LARGE_OFF):
                variants["large"] = SFEATS + LARGE_FEATS
    y = st.y.to_numpy()
    fold = (st.s1.to_numpy() // 7) % 2
    best, best_auc, qs = None, -1, {}
    for name, feats in variants.items():
        X = st[feats].to_numpy(np.float32)
        q = np.zeros(len(st))
        for h in (0, 1):
            m = lgb.train(_lgb_params(), lgb.Dataset(X[fold != h], y[fold != h]), 400)
            q[fold == h] = m.predict(X[fold == h])
        auc = roc_auc_score(y, q); qs[name] = q
        log(f"stack [{name}] band AUC: p {roc_auc_score(y, st.p):.4f}  ce {roc_auc_score(y, st.ce):.4f}  "
            f"stacker(cross-fit) {auc:.5f}  ({len(st):,} pairs, {len(feats)} inputs)")
        if auc > best_auc:
            best, best_auc = name, auc
    feats = variants[best]; q = qs[best]
    log(f"stacker variant chosen on m1-m2 cross-fit AUC: {best}")
    full = lgb.train(_lgb_params(), lgb.Dataset(st[feats].to_numpy(np.float32), y), 400)
    full.save_model(os.path.join(CE_DIR, f"{STACK_NAME}.txt"))
    json.dump({"variant": best, "feats": feats}, open(os.path.join(CE_DIR, f"{STACK_NAME}.json"), "w"))
    imp = sorted(zip(feats, full.feature_importance("gain")), key=lambda t: -t[1])
    log("stacker gain: " + ", ".join(f"{a}={b/sum(v for _, v in imp):.3f}" for a, b in imp[:10]))

    def merged(ctx, keys, qv, out, cols):
        s = pd.DataFrame({"s1": keys.s1.to_numpy(), "rid": keys.rid.to_numpy(), "q": qv})
        d = ctx.merge(s, on=["s1", "rid"], how="left")
        d["p2"] = np.where(d.q.notna(), d.q, d.p).astype(np.float32)
        d["p2_lgb"] = d["p2"]
        d[cols + ["p2", "p2_lgb"]].to_parquet(os.path.join(WORK_DIR, out))
        log(f"wrote {out}: {len(d):,} pairs, {int(d.q.notna().sum()):,} re-scored")

    merged(ctx, st, q, f"scores_train_oof_{OUT_TAG}.parquet", ["s1", "rid", "y"])
    del ctx, st
    for w, out, cols in (("holdout", f"scores_train_holdout_{OUT_TAG}.parquet", ["s1", "rid", "y"]),
                         ("test", f"scores_test_{OUT_TAG}.parquet", ["s1", "rid"])):
        ctx = pd.read_parquet(os.path.join(CE_DIR, f"ctx_{w}.parquet"))
        s = _stack_frame(ctx, pd.read_parquet(os.path.join(CE_DIR, f"ce_{w}.parquet")))
        if VER:
            s = _extras(s, "train" if w == "holdout" else "test", ["holdout"] if w == "holdout" else
                        [f"m{k}" for k in range(10)], w)
            if best == "ens":
                s = _ens(s, w)
            if VER in ("3", "4"):
                s = _ens_large(s)
        qv = full.predict(s[feats].to_numpy(np.float32))
        if w == "holdout":
            log(f"holdout band AUC: p {roc_auc_score(s.y, s.p):.4f}  ce {roc_auc_score(s.y, s.ce):.4f}  "
                f"stacker {roc_auc_score(s.y, qv):.5f}")
        merged(ctx, s, qv, out, cols)
        del ctx, s
    log(f"stack done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    st = sys.argv[1]
    {"band": stage_band, "train": stage_train, "score": stage_score, "stack": stage_stack,
     "prescore_large": stage_prescore_large}[st]()
