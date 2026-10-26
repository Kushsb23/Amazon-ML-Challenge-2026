"""Compare two matching_results.tsv files per country (how much each country's predictions changed)."""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import read_tsv, read_source

a, b = sys.argv[1], sys.argv[2]
ctry = read_source("test", 1, ["entity_id", "country"]).set_index("entity_id").country
def pairs(p):
    d = read_tsv(p)
    d = d[d.matched_entity_ids != ""]
    e = d.assign(rid=d.matched_entity_ids.str.split(",")).explode("rid")
    return set(zip(e.source1_entity_id, e.rid)), d.set_index("source1_entity_id").matched_entity_ids
pa, la = pairs(a); pb, lb = pairs(b)
rows = []
for c in sorted(ctry.unique()):
    ids = set(ctry.index[ctry == c])
    A = {x for x in pa if x[0] in ids}; B = {x for x in pb if x[0] in ids}
    changed = sum(1 for s in ids if la.get(s, "") != lb.get(s, ""))
    rows.append((c, len(ids), len(A), len(B), len(A & B), len(A - B), len(B - A), changed / len(ids)))
print(pd.DataFrame(rows, columns=["country", "S1", "pairs_old", "pairs_new", "same", "dropped", "added",
                                  "S1_changed_share"]).to_string(index=False))
