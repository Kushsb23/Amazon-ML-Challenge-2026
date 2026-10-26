"""Read-only: what do the holdout true pairs that no candidate source generates look like?

    python missed_pairs.py      -> counts by country / source file + 60 raw examples
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
import union_trim as U
from cross_encoder import load_texts
from rapidfuzz import fuzz

g = U.gt_subset("holdout")[["s1", "rid"]].drop_duplicates()
u = pd.read_parquet(os.path.join(WORK_DIR, "union_train_holdout.parquet"), columns=["s1", "rid"])
m = g.merge(u.assign(inu=1), on=["s1", "rid"], how="left")
miss = m[m.inu.isna()][["s1", "rid"]].reset_index(drop=True)
print(f"true pairs {len(g):,}; never generated {len(miss):,} ({len(miss)/len(g):.2%})")
s1c = pd.read_parquet(os.path.join(WORK_DIR, "norm_train_s1.parquet"), columns=["entity_id", "country"])
from blocking_keys import encode_id
ctry = pd.Series(s1c.country.astype(str).to_numpy(), index=encode_id(s1c.entity_id))
miss["country"] = ctry.reindex(miss.s1.to_numpy()).to_numpy()
miss["src"] = miss.rid // 10 ** 12
print(miss.groupby(["country", "src"]).size().to_string())
# S1s whose every true pair is missed vs partially missed
nt = g.groupby("s1").size(); nm = miss.groupby("s1").size()
print(f"S1 with all true pairs missed: {(nm == nt.reindex(nm.index)).sum():,} of {len(nm):,} affected S1")
smp = miss.sample(n=min(4000, len(miss)), random_state=0)
txt = load_texts("train", np.r_[smp.s1.to_numpy(), smp.rid.to_numpy()])
smp["a"] = txt.reindex(smp.s1.to_numpy()).to_numpy(); smp["b"] = txt.reindex(smp.rid.to_numpy()).to_numpy()
na = smp.a.str.split(" \\| ").str[0].fillna(""); nb = smp.b.str.split(" \\| ").str[0].fillna("")
smp["name_sim"] = [fuzz.token_set_ratio(x, y) for x, y in zip(na, nb)]
nonascii = lambda s: pd.Series(s).fillna("").map(lambda x: any(ord(ch) > 127 for ch in x))
smp["script_diff"] = nonascii(smp.a).to_numpy() != nonascii(smp.b).to_numpy()
print("\nname token_set similarity of missed pairs (sample):")
print(pd.cut(smp.name_sim, [-1, 30, 50, 70, 85, 100]).value_counts().sort_index().to_string())
print(f"one side non-ASCII, other not: {smp.script_diff.mean():.3f}")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print("\n60 examples:")
for _, r in smp.head(60).iterrows():
    print(f"[{r.country} S{r.src} sim {r.name_sim:>3}]  {r.a}\n{'':22}{r.b}")
