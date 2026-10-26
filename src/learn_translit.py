"""Learn Indic-script -> Latin token / segment maps from TRAINING pairs only (no external data).

Names:   Indic-script S2/S3 names are word-by-word transliterations of the S1 (Latin) name
         (e.g. 'मॉडर्न डिजिटल कंसल्टेंसी प्राइवेट लिमिटेड' <-> 'Modern Digital Consultancy Private Limited').
         When both sides have the same number of tokens we align them position-wise and count.
Address: Indic segments are state names; we count co-occurrence with the S1 address's comma segments.

Output: WORK_DIR/translit_map.json  {"token": {indic: latin}, "segment": {indic: latin}}
"""
import os, sys, re, json, time, collections
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from io_utils import read_source, read_gt_pairs

INDIC_RE = re.compile(r"[ऀ-෿]")
ZW_RE = re.compile(r"[​-‍﻿]")


def latin_tokens(s):
    s = s.lower().replace("&", " and ")
    return re.findall(r"[a-z0-9]+", s)


def indic_tokens(s):
    s = ZW_RE.sub("", s)
    return [t for t in re.split(r"[\s,.()\-]+", s) if t]


def main(min_count=3, min_share=0.6):
    t0 = time.time()
    pairs = read_gt_pairs()
    s1 = read_source("train", 1)
    s1 = s1[s1.country == "India"].set_index("entity_id")
    recs = []
    for k in (2, 3):
        r = read_source("train", k)
        r = r[(r.country == "India") & (r.business_name.str.contains(INDIC_RE.pattern) |
                                        r.business_address.str.contains(INDIC_RE.pattern))]
        recs.append(r)
    recs = pd.concat(recs).set_index("entity_id")
    p = pairs[pairs.rid.isin(recs.index)]
    p = p[p.s1_id.isin(s1.index)]
    n1 = s1.business_name.reindex(p.s1_id).to_numpy()
    a1 = s1.business_address.reindex(p.s1_id).to_numpy()
    nr = recs.business_name.reindex(p.rid).to_numpy()
    ar = recs.business_address.reindex(p.rid).to_numpy()
    print(f"indic-containing true pairs: {len(p):,}  ({time.time()-t0:.0f}s)")

    tok = collections.defaultdict(collections.Counter)
    seg = collections.defaultdict(collections.Counter)
    aligned = total = 0
    for a, b, aa, bb in zip(n1, nr, a1, ar):
        if INDIC_RE.search(b):
            total += 1
            it, lt = indic_tokens(b), latin_tokens(a)
            if len(it) == len(lt):
                aligned += 1
                for x, y in zip(it, lt):
                    if INDIC_RE.search(x):
                        tok[x][y] += 1
        if INDIC_RE.search(bb):
            lsegs = {s.strip().lower() for s in aa.split(",") if s.strip()}
            for sgm in bb.split(","):
                sgm = ZW_RE.sub("", sgm.strip())
                if INDIC_RE.search(sgm):
                    for l in lsegs:
                        seg[sgm][l] += 1
                    seg[sgm]["__n__"] += 1
    print(f"name pairs with indic script: {total:,}; token-count aligned: {aligned/total:.3f}")

    tmap = {}
    for k, c in tok.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= min_share:
            tmap[k] = y
    smap = {}
    for k, c in seg.items():
        n_all = c.pop("__n__")
        y, n = c.most_common(1)[0]
        if n / n_all >= 0.8:
            smap[k] = y
    print(f"token map: {len(tmap):,} entries (of {len(tok):,} indic tokens seen); segment map: {smap}")
    json.dump({"token": tmap, "segment": smap}, open(os.path.join(WORK_DIR, "translit_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    # coverage: share of indic token occurrences covered
    occ = sum(sum(c.values()) for c in tok.values())
    cov = sum(sum(tok[k].values()) for k in tmap)
    print(f"occurrence coverage of learned tokens (train): {cov/occ:.3f}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
