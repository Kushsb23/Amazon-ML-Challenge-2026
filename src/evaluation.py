"""Shared evaluation for the whole team.

Holdout: train S1 entities with int(id[3:]) % 10 == 0.

Candidate file format (parquet or tsv) every blocking method writes:
    s1_id | rid | method | score        (one row per candidate pair; score optional)

CLI:
    python evaluation.py candidates  <file> [<file> ...]      # blocking report on holdout (union of files)
    python evaluation.py predictions <matching_results.tsv>    # macro F0.5 on holdout
    python evaluation.py export                                  # write holdout id list + holdout GT for teammates
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HOLDOUT_MOD, WORK_DIR, DATA_DIR
from io_utils import read_source, read_gt_pairs, read_tsv


# ------------------------------------------------------------------ holdout
def is_holdout(ids) -> np.ndarray:
    s = pd.Series(ids, dtype="string[pyarrow]")
    return (s.str.slice(3).astype("int64") % HOLDOUT_MOD == 0).to_numpy()


def holdout_ids():
    s1 = read_source("train", 1, ["entity_id"]).entity_id
    return s1[is_holdout(s1)].tolist()


# ------------------------------------------------------------------ metric
def f05_per_entity(n_pred: np.ndarray, n_true: np.ndarray, n_tp: np.ndarray) -> np.ndarray:
    """Vectorised per-entity F0.5 with the challenge's singleton rule."""
    n_pred = np.asarray(n_pred, float); n_true = np.asarray(n_true, float); n_tp = np.asarray(n_tp, float)
    out = np.zeros(len(n_pred))
    single = n_true == 0
    out[single] = (n_pred[single] == 0).astype(float)
    ok = (~single) & (n_tp > 0)
    p = n_tp[ok] / n_pred[ok]
    r = n_tp[ok] / n_true[ok]
    out[ok] = 1.25 * p * r / (0.25 * p + r)
    return out


def macro_f05(pred_pairs: pd.DataFrame, true_pairs: pd.DataFrame, s1_ids) -> float:
    """pred_pairs/true_pairs: DataFrames with s1_id, rid. s1_ids: evaluation universe."""
    return float(per_entity_scores(pred_pairs, true_pairs, s1_ids).mean())


def per_entity_scores(pred_pairs, true_pairs, s1_ids) -> pd.Series:
    ids = pd.Index(pd.Series(list(s1_ids), dtype="string[pyarrow]").unique())
    pp = pred_pairs[["s1_id", "rid"]].drop_duplicates()
    pp = pp[pp.s1_id.isin(ids)]
    tp_ = true_pairs[true_pairs.s1_id.isin(ids)]
    n_pred = pp.groupby("s1_id").size().reindex(ids, fill_value=0)
    n_true = tp_.groupby("s1_id").size().reindex(ids, fill_value=0)
    hit = pp.merge(tp_, on=["s1_id", "rid"])
    n_tp = hit.groupby("s1_id").size().reindex(ids, fill_value=0)
    return pd.Series(f05_per_entity(n_pred.values, n_true.values, n_tp.values), index=ids)


# ------------------------------------------------------------------ candidate report
def candidate_report(cands: pd.DataFrame, true_pairs: pd.DataFrame, s1_ids, by_method=True) -> dict:
    """Recall ceiling + size of a candidate set, and the oracle F0.5 a perfect matcher would get."""
    ids = pd.Index(pd.Series(list(s1_ids), dtype="string[pyarrow]").unique())
    c = cands[cands.s1_id.isin(ids)]
    tp_ = true_pairs[true_pairs.s1_id.isin(ids)]
    cu = c[["s1_id", "rid"]].drop_duplicates()
    hit = cu.merge(tp_, on=["s1_id", "rid"])
    rep = {
        "n_s1": len(ids),
        "n_candidate_pairs": len(cu),
        "avg_cands_per_s1": len(cu) / len(ids),
        "pair_recall": len(hit) / max(1, len(tp_)),
        # perfect matcher on these candidates: predicts exactly the true pairs it can see
        "oracle_macro_f05": macro_f05(hit, tp_, ids),
        "s1_with_zero_cands": float((~ids.isin(cu.s1_id)).mean()),
    }
    if by_method and "method" in c.columns:
        per = {}
        for m, g in c.groupby("method"):
            gu = g[["s1_id", "rid"]].drop_duplicates()
            h = gu.merge(tp_, on=["s1_id", "rid"])
            per[m] = {"pairs": len(gu), "avg_per_s1": round(len(gu) / len(ids), 2),
                      "recall": round(len(h) / max(1, len(tp_)), 4)}
            # unique contribution: true pairs only this method finds
            others = c[c.method != m][["s1_id", "rid"]].drop_duplicates()
            only = h.merge(others, on=["s1_id", "rid"], how="left", indicator=True)
            per[m]["unique_recall"] = round((only._merge == "left_only").sum() / max(1, len(tp_)), 4)
        rep["by_method"] = per
    return rep


def _read_any(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    df = read_tsv(path)
    if "candidate_entity_ids" in df.columns or "matched_entity_ids" in df.columns:  # submission format
        col = "candidate_entity_ids" if "candidate_entity_ids" in df.columns else "matched_entity_ids"
        df = df[df[col] != ""]
        ex = df.assign(rid=df[col].str.split(",")).explode("rid")
        return pd.DataFrame({"s1_id": ex.source1_entity_id, "rid": ex.rid.str.strip(),
                             "method": os.path.basename(path)})
    return df


def main(argv):
    cmd = argv[1]
    tp = read_gt_pairs()
    ids = holdout_ids()
    tp = tp[tp.s1_id.isin(set(ids))]
    if cmd == "candidates":
        frames = []
        for p in argv[2:]:
            d = _read_any(p)
            if "method" not in d.columns:
                d["method"] = os.path.basename(p)
            frames.append(d)
        rep = candidate_report(pd.concat(frames, ignore_index=True), tp, ids)
        by = rep.pop("by_method", {})
        for k, v in rep.items():
            print(f"{k:>22}: {v:.4f}" if isinstance(v, float) else f"{k:>22}: {v:,}")
        for m, v in by.items():
            print(f"  [{m}] {v}")
    elif cmd == "predictions":
        pred = _read_any(argv[2])
        print(f"holdout macro F0.5 = {macro_f05(pred, tp, ids):.5f}  (n_s1={len(ids):,})")
    elif cmd == "export":
        os.makedirs(WORK_DIR, exist_ok=True)
        pd.Series(ids, name="source1_entity_id").to_csv(os.path.join(WORK_DIR, "holdout_s1_ids.tsv"), index=False, sep="\t")
        print(f"wrote {len(ids):,} holdout ids -> {WORK_DIR}/holdout_s1_ids.tsv")


if __name__ == "__main__":
    main(sys.argv)


# ------------------------------------------------------------------ fast integer-id evaluation (large candidate sets)
def gt_int(holdout_only=True) -> pd.DataFrame:
    """Ground-truth pairs as int64 ids (S2-123 -> 2e12+123), cached in WORK_DIR."""
    from blocking_keys import encode_id
    p = os.path.join(WORK_DIR, "gt_pairs_int.parquet")
    if not os.path.exists(p):
        tp = read_gt_pairs()
        pd.DataFrame({"s1": encode_id(tp.s1_id), "rid": encode_id(tp.rid)}).to_parquet(p)
    g = pd.read_parquet(p)
    if holdout_only:
        g = g[(g.s1 % 10 ** 12) % HOLDOUT_MOD == 0]
    return g


def recall_report_int(c: pd.DataFrame, g: pd.DataFrame, n_s1: int, flags=None) -> dict:
    """c: int candidates (s1, rid[, flags]); g: int gt pairs for the same S1 universe."""
    cu = c[["s1", "rid"]].drop_duplicates()
    hit = g.merge(cu, on=["s1", "rid"])
    per_true = g.groupby("s1").size()
    per_hit = hit.groupby("s1").size().reindex(per_true.index, fill_value=0)
    rep = {"pairs": len(cu), "avg_cands_per_s1": len(cu) / n_s1, "pair_recall": len(hit) / len(g),
           "entities_fully_covered": float((per_hit == per_true).mean())}
    # oracle F0.5 (perfect matcher on candidates): recall r_i per entity, precision 1 -> F = 1.25 r/(0.25+r)
    r = (per_hit / per_true).to_numpy()
    f = np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)
    n_single = n_s1 - len(per_true)                  # singletons: perfect matcher predicts empty -> 1.0
    rep["oracle_macro_f05"] = (f.sum() + n_single) / n_s1
    return rep
