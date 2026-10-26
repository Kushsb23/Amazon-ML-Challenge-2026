"""Diagnostics for an unseen country (default France): how well normalisation parses its records, and how
confident the matcher is on it compared with the training countries. Writes a plain-text report."""
import os, sys, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from io_utils import read_source
from blocking_keys import encode_id

CTRY = sys.argv[1] if len(sys.argv) > 1 else "France"
SFX = sys.argv[2] if len(sys.argv) > 2 else ""
rng = np.random.default_rng(0)
out = []
P = out.append
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)

raw = {k: read_source("test", k) for k in (1, 2, 3)}
norm = {k: pd.read_parquet(os.path.join(WORK_DIR, f"norm_test_s{k}.parquet")) for k in (1, 2, 3)}

# ---- field coverage per country (S1 + pool)
P("=== field coverage by country (share of records with a non-empty field) ===")
for k in (1, 2, 3):
    n = norm[k]
    cov = n.groupby(n.country.astype(str)).agg(
        rows=("entity_id", "size"),
        a_num=("a_num", lambda s: (s.astype(str) != "").mean()),
        a_street=("a_street", lambda s: (s.astype(str) != "").mean()),
        a_city=("a_city", lambda s: (s.astype(str) != "").mean()),
        a_state=("a_state", lambda s: (s.astype(str) != "").mean()),
        a_unit=("a_unit", lambda s: (s.astype(str) != "").mean()),
        num_5digit=("a_num", lambda s: s.astype(str).str.fullmatch(r"\d{5}").mean()),
        city_has_digit=("a_city", lambda s: s.astype(str).str.contains(r"\d").mean()),
        a_empty=("a_empty", "mean"))
    P(f"-- source {k}\n" + cov.round(3).to_string())

# ---- raw vs parsed samples
def sample(k, m):
    r = raw[k].merge(norm[k], on=["entity_id"], suffixes=("", "_n"))
    r = r[r.country.astype(str) == CTRY]
    return r.iloc[rng.choice(len(r), size=min(m, len(r)), replace=False)]

for k, m in ((1, 60), (2, 40), (3, 40)):
    s = sample(k, m)
    P(f"\n=== {CTRY} source {k}: raw -> parsed ({len(s)} random records) ===")
    for _, x in s.iterrows():
        P(f"{x.entity_id} | {x.business_name} | {x.business_address}")
        P(f"    core='{x.n_core}' legal='{x.legal}' | num='{x.a_num}' nums='{x.a_nums}' street='{x.a_street}' "
          f"city='{x.a_city}' locs='{x.a_locs}' state='{x.a_state}' unit='{x.a_unit}'")

# ---- matcher confidence by country
sc = pd.read_parquet(os.path.join(WORK_DIR, f"scores_test{SFX}.parquet"))
ctry = pd.Series(norm[1].country.astype(str).to_numpy(), index=encode_id(norm[1].entity_id))
sc["c"] = ctry.reindex(sc.s1.to_numpy()).to_numpy()
best = sc.groupby("s1").agg(c=("c", "first"), pmax=("p2", "max"),
                            n_hi=("p2", lambda x: (x >= 0.725).sum()),
                            n_mid=("p2", lambda x: ((x >= 0.2) & (x < 0.725)).sum()))
P("\n=== matcher confidence per country (test) ===")
P(best.groupby("c").agg(S1=("pmax", "size"), best_p_median=("pmax", "median"),
                        share_best_lt_0_5=("pmax", lambda x: (x < 0.5).mean()),
                        share_best_0_5_0_9=("pmax", lambda x: ((x >= 0.5) & (x < 0.9)).mean()),
                        mean_n_accepted=("n_hi", "mean"),
                        mean_n_uncertain_0_2_0_725=("n_mid", "mean")).round(4).to_string())

# ---- top candidates of random French S1 (raw text) to eyeball
idmap = pd.concat([pd.Series(raw[k].entity_id.to_numpy(), index=encode_id(raw[k].entity_id)) for k in (1, 2, 3)])
txt = pd.concat([raw[k].set_index("entity_id")[["business_name", "business_address"]] for k in (1, 2, 3)])
fr = best[best.c == CTRY]
for label, sel in (("uncertain (best p between 0.3 and 0.8)", fr[(fr.pmax >= 0.3) & (fr.pmax < 0.8)]),
                   ("random", fr)):
    ids = sel.index.to_numpy()
    ids = rng.choice(ids, size=min(15, len(ids)), replace=False)
    P(f"\n=== {CTRY} S1 with top-5 candidates: {label} ===")
    for s1 in ids:
        s1s = idmap[s1]
        P(f"\n{s1s} | {txt.loc[s1s].business_name} | {txt.loc[s1s].business_address}")
        c = sc[sc.s1 == s1].nlargest(5, "p2")
        for _, r in c.iterrows():
            rs = idmap[r.rid]
            P(f"   p={r.p2:.3f} {rs} | {txt.loc[rs].business_name} | {txt.loc[rs].business_address}")

path = os.path.join(WORK_DIR, f"audit_{CTRY}.txt")
open(path, "w", encoding="utf-8").write("\n".join(out))
print(f"wrote {path}")
