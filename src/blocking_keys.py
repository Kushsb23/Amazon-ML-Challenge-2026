"""Key-based candidate generation (address keys + rare name tokens), all within country.

Every record gets a handful of 64-bit hash keys; S1 records are joined to S2/S3 records that share
a key. Keys whose block in the S2/S3 pool is larger than a cap are dropped (generic values such as
'house no 1 + Mumbai' or the token 'services'), so the candidate set stays small.

Methods (bit flags, so one pair can be found by several):
    1   num_street   house number + first street word                (US / France)
    2   num_city     house number + city
    4   street_city  full street name + city (catches missing / altered house numbers)
    8   rare_tok     rare name token + city   (renamed / DBA / generic names)
   16   rare_tok_ctry very rare name token alone (city missing or misspelt)
   32   nospace      whole core name without spaces (domains, hashtags: silverportfolio.com)
   64   num_loc      India: any address number + any locality segment
  128   name_city    exact core name + city
  256   alt_name     DBA 'other side' name / tokens  (both directions)

Usage:
    cands = generate(split="train", s1_filter=ev.is_holdout)     # DataFrame s1 (int64), rid (int64), flags (uint16)
"""
import os
import gc
import time
import numpy as np
import pandas as pd

from config import WORK_DIR

METHODS = {1: "num_street", 2: "num_city", 4: "street_city", 8: "rare_tok", 16: "rare_tok_ctry",
           32: "nospace", 64: "num_loc", 128: "name_city", 256: "alt_name"}

CAPS = {  # max pool records per key (larger blocks are skipped as uninformative)
    "num_street": 60, "num_city": 60, "street_city": 60, "rare_tok": 60, "rare_tok_ctry": 30,
    "nospace": 60, "num_loc": 60, "name_city": 100, "alt_name": 30,
}
RARE_DF = 3000          # a name token is 'rare' if it appears in <= RARE_DF records of the country pool
N_RARE = 3              # use up to 3 rarest tokens per record
MIN_TOK_LEN = 3
STOP = {"pvt", "ltd", "llc", "llp", "inc", "corp", "co", "plc", "pc", "pa", "sarl", "sas", "sasu", "sa", "eurl",
        "sci", "ei", "ets", "and", "the", "of", "shri", "services", "service", "group", "india", "enterprises",
        "solutions", "center", "company", "trading", "global", "international", "associates", "holdings"}

NORM_COLS = ["entity_id", "country", "n_core", "n_alt", "n_nospace", "a_num", "a_nums", "a_street",
             "a_city", "a_locs"]


# ------------------------------------------------------------------ id / hash helpers
def encode_id(ids: pd.Series) -> np.ndarray:
    """'S2-123' -> 2_000_000_000_123 so joins run on int64 instead of strings."""
    s = pd.Series(ids).astype(str)
    return s.str.slice(1, 2).astype("int64").to_numpy() * 10 ** 12 + s.str.slice(3).astype("int64").to_numpy()


def decode_id(x) -> pd.Series:
    x = np.asarray(x, dtype=np.int64)
    src, num = x // 10 ** 12, x % 10 ** 12
    return pd.Series(np.char.add(np.char.add("S", src.astype(str)), np.char.add("-", num.astype(str))),
                     dtype="string[pyarrow]")


def hkey(*cols) -> np.ndarray:
    s = cols[0].astype(str)
    for c in cols[1:]:
        s = s + "\x1f" + c.astype(str)
    return pd.util.hash_pandas_object(s, index=False).to_numpy()


def load_norm(split, k, cols=NORM_COLS):
    return pd.read_parquet(os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet"), columns=cols)


# ------------------------------------------------------------------ key extraction
def _explode_tokens(ids, country, text, city=None, rare_df=None, n_keep=N_RARE):
    """One row per (record, rare token). rare_df: Series token->df for this pool."""
    t = pd.DataFrame({"id": ids, "country": country, "tok": text.str.split(" ")})
    if city is not None:
        t["city"] = city
    t = t.explode("tok")
    t = t[t.tok.notna() & (t.tok.str.len() >= MIN_TOK_LEN) & ~t.tok.isin(STOP)]
    if rare_df is not None:
        t["df"] = rare_df.reindex(pd.MultiIndex.from_arrays([t.country, t.tok])).to_numpy()
        t = t[t.df.notna() & (t.df <= RARE_DF)]
        t = t.sort_values(["id", "df"]).groupby("id", sort=False).head(n_keep)
    return t


def make_keys(d: pd.DataFrame, tok_df: pd.Series) -> dict:
    """d: normalised records (one country or many). Returns {method: DataFrame(id, key)}."""
    ids = encode_id(d.entity_id)
    ctry = d.country.astype(str)
    num = d.a_num.astype(str).str.lstrip("0")
    street = d.a_street.astype(str)
    city = d.a_city.astype(str)
    street_first = street.str.split(" ").str[0].fillna("")
    has_num, has_city, has_street = (num != "").to_numpy(), (city != "").to_numpy(), (street != "").to_numpy()
    out = {}

    m = has_num & has_street
    out["num_street"] = pd.DataFrame({"id": ids[m], "key": hkey(ctry[m], num[m], street_first[m])})
    m = has_num & has_city
    out["num_city"] = pd.DataFrame({"id": ids[m], "key": hkey(ctry[m], num[m], city[m])})
    m = has_street & has_city
    out["street_city"] = pd.DataFrame({"id": ids[m], "key": hkey(ctry[m], street[m], city[m])})
    m = has_city
    out["name_city"] = pd.DataFrame({"id": ids[m], "key": hkey(ctry[m], d.n_core.astype(str)[m], city[m])})
    ns = d.n_nospace.astype(str)
    m = (ns.str.len() >= 5).to_numpy()
    out["nospace"] = pd.DataFrame({"id": ids[m], "key": hkey(ctry[m], ns[m])})

    # rare name tokens (+ city, and alone when very rare)
    t = _explode_tokens(ids, ctry.to_numpy(), d.n_core.astype(str), city.to_numpy(), tok_df)
    tc = t[t.city != ""]
    out["rare_tok"] = pd.DataFrame({"id": tc.id.to_numpy(), "key": hkey(tc.country, tc.tok, tc.city)})
    tv = t[t.df <= 200]
    out["rare_tok_ctry"] = pd.DataFrame({"id": tv.id.to_numpy(), "key": hkey(tv.country, tv.tok)})

    # DBA other side: exact alt core name, and its tokens vs the other side's core tokens
    alt = d.n_alt.astype(str)
    m = (alt != "").to_numpy()
    if m.any():
        out["alt_name"] = pd.DataFrame({"id": ids[m], "key": hkey(ctry[m], alt[m])})

    # India: every (number, locality) combination
    ind = (ctry == "India").to_numpy()
    if ind.any():
        di = pd.DataFrame({"id": ids[ind], "country": ctry[ind].to_numpy(),
                           "nums": d.a_nums.astype(str)[ind].str.split(" ").str[:3].to_numpy(),
                           "locs": d.a_locs.astype(str)[ind].str.split("|").to_numpy()})
        di = di.explode("nums").explode("locs")
        di = di[(di.nums.fillna("") != "") & (di.locs.fillna("").str.len() >= 3)]
        out["num_loc"] = pd.DataFrame({"id": di.id.to_numpy(), "key": hkey(di.country, di.nums, di.locs)})
    return out


# ------------------------------------------------------------------ candidate generation
ALT_NAME_FLAG = 256
CHUNK = 500_000


def iter_norm(split, k, cols=NORM_COLS, chunk=CHUNK):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet"))
    for b in f.iter_batches(batch_size=chunk, columns=cols):
        yield b.to_pandas()


def pool_token_df(split) -> pd.Series:
    """(country, token) -> number of S2/S3 pool records containing it (streamed)."""
    acc = None
    for k in (2, 3):
        for d in iter_norm(split, k, ["country", "n_core"]):
            t = pd.DataFrame({"country": d.country.astype(str).to_numpy(),
                              "tok": d.n_core.astype(str).str.split(" ").to_numpy()}).explode("tok")
            vc = t.groupby(["country", "tok"]).size()
            acc = vc if acc is None else acc.add(vc, fill_value=0)
    return acc


def build_keys(split, k_list, tdf, s1_filter=None):
    """Stream records of the given sources and return {method: DataFrame(id int64, key uint64)}."""
    parts = {}
    for k in k_list:
        for d in iter_norm(split, k):
            if s1_filter is not None:
                d = d[s1_filter(d.entity_id)]
                if not len(d):
                    continue
            for meth, df in make_keys(d, tdf).items():
                parts.setdefault(meth, []).append(df)
            if k == 1:   # S1 exact core name, to be matched against pool DBA 'other side'
                parts.setdefault("_s1name", []).append(
                    pd.DataFrame({"id": encode_id(d.entity_id), "key": hkey(d.country.astype(str), d.n_core.astype(str))}))
            del d; gc.collect()
    return {m: pd.concat(v, ignore_index=True) for m, v in parts.items()}


def _cache_dir(split):
    return os.path.join(WORK_DIR, f"poolkeys_{split}")


def save_pool_cache(split, pkeys, tdf):
    d = _cache_dir(split); os.makedirs(d, exist_ok=True)
    for m, v in pkeys.items():
        v.to_parquet(os.path.join(d, f"{m}.parquet"))
    tdf.rename("df").reset_index().to_parquet(os.path.join(d, "_tokdf.parquet"))


def load_pool_cache(split):
    d = _cache_dir(split)
    if not os.path.exists(os.path.join(d, "_tokdf.parquet")):
        return None
    pkeys = {f[:-8]: pd.read_parquet(os.path.join(d, f)) for f in os.listdir(d)
             if f.endswith(".parquet") and not f.startswith("_")}
    t = pd.read_parquet(os.path.join(d, "_tokdf.parquet"))
    tdf = t.set_index(["country", "tok"])["df"]
    return pkeys, tdf


def generate(split="train", s1_filter=None, verbose=True, pool=None) -> pd.DataFrame:
    """Candidates DataFrame(s1:int64, rid:int64, flags:uint16) for S1 records of `split`
    (optionally only those where s1_filter(entity_id_series) is True).
    `pool` = (pkeys, tdf) can be passed to reuse pool keys across calls."""
    t0 = time.time()
    if pool is None:
        pool = load_pool_cache(split)
    if pool is None:
        tdf = pool_token_df(split)
        pkeys = build_keys(split, (2, 3), tdf)
        # cap block sizes once
        for meth, pk in pkeys.items():
            cap = CAPS.get(meth, 60)
            size = pk.key.map(pk.key.value_counts())
            pkeys[meth] = pk[(size <= cap).to_numpy()].reset_index(drop=True)
        if verbose:
            print(f"[{split}] pool keys built ({time.time()-t0:.0f}s): " +
                  ", ".join(f"{m}={len(v):,}" for m, v in pkeys.items()), flush=True)
        save_pool_cache(split, pkeys, tdf)
    else:
        pkeys, tdf = pool
    qkeys = build_keys(split, (1,), tdf, s1_filter)
    qkeys.pop("alt_name", None)
    qname = qkeys.pop("_s1name")

    flag_of = {v: kk for kk, v in METHODS.items()}
    parts = []
    for meth, qk in qkeys.items():
        j = qk.merge(pkeys[meth], on="key", suffixes=("", "_r"))[["id", "id_r"]]
        j["flags"] = np.uint16(flag_of[meth])
        parts.append(j)
        if verbose:
            print(f"  {meth:14s} q_keys={len(qk):>9,} pairs={len(j):>10,}", flush=True)
    if "alt_name" in pkeys:
        j = qname.merge(pkeys["alt_name"], on="key", suffixes=("", "_r"))[["id", "id_r"]]
        j["flags"] = np.uint16(ALT_NAME_FLAG)
        parts.append(j)
    del qkeys; gc.collect()
    c = pd.concat(parts, ignore_index=True).drop_duplicates()
    # each part carries a single bit, so OR of distinct bits == sum
    c = c.groupby(["id", "id_r"], sort=False)["flags"].sum().reset_index()
    c.columns = ["s1", "rid", "flags"]
    c["flags"] = c["flags"].astype(np.uint16)
    if verbose:
        print(f"[{split}] candidates: {len(c):,} unique pairs ({time.time()-t0:.0f}s)", flush=True)
    return c, (pkeys, tdf)


def to_string_pairs(c: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"s1_id": decode_id(c.s1.to_numpy()), "rid": decode_id(c.rid.to_numpy())})


def expand_methods(c: pd.DataFrame) -> pd.DataFrame:
    """long format (s1_id, rid, method) for evaluation.candidate_report."""
    rows = []
    for flag, name in METHODS.items():
        m = (c.flags.to_numpy() & flag) > 0
        if m.any():
            sub = c[m]
            rows.append(pd.DataFrame({"s1_id": decode_id(sub.s1.to_numpy()), "rid": decode_id(sub.rid.to_numpy()),
                                      "method": name}))
    return pd.concat(rows, ignore_index=True)
