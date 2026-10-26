"""Blend our final pair scores with a second, independent pipeline (pipeline B).

Inputs (pipeline B's export, one row per candidate pair of its pruned set):
    <BER_PIPE_B_DIR>/test_pairs_v4.parquet        source1_entity_id, target_entity_id, matcher_p, ce_logit, in_band,
                                                  final_score, linked
    <BER_PIPE_B_DIR>/heldout_C_pairs_v4.parquet   same + label, for its held-out split C (~20% of train S1,
                                                  never used to train its models)
Our side: scores_{train_oof,train_holdout,test}_{BASE}.parquet (s1, rid, y, p2), BASE = BER_BLEND_BASE (ce4).

No test labels and no training on test: the blender is fitted only on S1 in C and in our stacker slices m1-m2
(both sides out-of-fold there), cross-fitted by S1 for the decision-rule tuning, and reported on S1 in C and in
our holdout (both sides out-of-sample).

    set BER_BLEND_BASE=ce4 & python ensemble.py      -> scores_*_blend.parquet, models/decision_blend.json
    set BER_SCORES=blend & set BER_OUT_DIR=..\\output_blend & python postprocess.py write
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from blocking_keys import encode_id
import tfidf_knn as T
import union_trim as U

BASE = os.environ.get("BER_BLEND_BASE", "ce4")
TEST_BASE = os.environ.get("BER_BLEND_TEST_BASE", BASE)   # e.g. ce4fr: France-variant test scores, same blender
FDIR = os.environ.get("BER_PIPE_B_DIR", os.path.join(WORK_DIR, "pipeline_b"))
TAG = os.environ.get("BER_BLEND_TAG", "blend")
MD = os.path.join(WORK_DIR, "models")
FEATS = ["p_o", "has_o", "fm", "ff", "fce", "fband", "fl", "has_f", "p_avg", "p_max", "p_min",
         "o_rank", "f_rank", "o_best", "f_best", "o_gap", "f_gap", "o_rid_rank", "f_rid_rank", "n_o", "n_f", "src"]


RICH = os.environ.get("BER_BLEND_RICH", "") == "1"   # + our stage-2 / cross-encoder / pair features
RCOLS = ["c_name", "c_addr", "c_rec", "num_eq", "num_diff", "name_eq", "legal_eq", "street_eq", "city_eq", "b_empty",
         "hits", "prune_c", "cos_n", "cos_a", "cos_f", "min_fr", "min_br", "csls_f", "s_mg_f", "t_mg_f"]
TEST_CAND_DIR = os.environ.get("BER_BLEND_TEST_CAND", os.path.join(WORK_DIR, "ce4_base"))


def enrich(d, split, subsets, stage2_file, ce_file, cand_dir=WORK_DIR):
    """Join our stage-2 probabilities, cross-encoder logits and pair features (NaN where we have no pair)."""
    keys = d[["s1", "rid"]]
    parts = []
    for sb in subsets:
        p = os.path.join(cand_dir, f"cand_{split}_{sb}.parquet")
        import pyarrow.parquet as pq
        cols = ["s1", "rid"] + [c for c in RCOLS if c in pq.ParquetFile(p).schema_arrow.names]
        k = keys[T.subset_mask(keys.s1.to_numpy(), sb)]
        parts.append(k.merge(pd.read_parquet(p, columns=cols).drop_duplicates(["s1", "rid"]), on=["s1", "rid"], how="left"))
    x = pd.concat(parts, ignore_index=True)
    d = d.merge(x, on=["s1", "rid"], how="left")
    s2 = pd.read_parquet(os.path.join(WORK_DIR, stage2_file))
    s2 = s2[["s1", "rid"] + [c for c in ("p1", "p2_lgb") if c in s2]].drop_duplicates(["s1", "rid"])
    d = d.merge(s2.rename(columns={"p1": "o_p1", "p2_lgb": "o_p2"}), on=["s1", "rid"], how="left")
    ce = pd.read_parquet(os.path.join(WORK_DIR, "ce4", ce_file), columns=["s1", "rid", "ce", "cel"])
    d = d.merge(ce.drop_duplicates(["s1", "rid"]).rename(columns={"ce": "o_ce", "cel": "o_cel"}), on=["s1", "rid"], how="left")
    for c in ("o_p2", "o_ce"):
        g = d.groupby("rid", sort=False)[c]
        d[f"{c}_rid_gap"] = (g.transform("max") - d[c]).astype(np.float32)
    return d


RFEATS = [c for c in RCOLS] + ["o_p1", "o_p2", "o_ce", "o_cel", "o_p2_rid_gap", "o_ce_rid_gap"]
SPLIT = os.environ.get("BER_BLEND_SPLIT_EMPTY", "") == "1"
NAME = os.environ.get("BER_BLEND_NAME", "") == "1"          # + name-focused pair features (name_feats.py)
SPEC = os.environ.get("BER_BLEND_SPEC", "") == "1"          # + empty-address specialist score (spec_empty.py)


def add_spec(d, fname):
    sp = pd.read_parquet(os.path.join(WORK_DIR, fname)).drop_duplicates(["s1", "rid"])
    d = d.merge(sp, on=["s1", "rid"], how="left")
    d["has_spec"] = d.p_spec.notna().astype(np.int8)
    g = d.groupby("s1", sort=False).p_spec
    d["spec_gap"] = (g.transform("max") - d.p_spec).astype(np.float32)
    return d


def add_names(d, split, tag):
    import name_feats as NF
    f = NF.compute(d, split, f"{tag}_{BASE}", "p_avg")
    return d.merge(f[["s1", "rid"] + NF.NFEATS], on=["s1", "rid"], how="left")   # separate blenders + score shift for empty-address records
NORID = os.environ.get("BER_BLEND_NORID", "") == "1"   # drop record-side rank features: their competitor pool differs
if NORID:                                               # between the blender's training frame (split C only) and test
    FEATS[:] = [f for f in FEATS if f not in ("o_rid_rank", "f_rid_rank")]
    RFEATS = [f for f in RFEATS if not f.endswith("_rid_gap")]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def pipe_b(fname):
    p = os.path.join(FDIR, fname)
    if not os.path.exists(p):                   # shipped to the laptop in <20 MB parts
        import glob
        parts = sorted(glob.glob(p.replace(".parquet", ".part*.parquet")))
        assert parts, f"missing {p}"
        d = pd.concat([pd.read_parquet(q) for q in parts], ignore_index=True)
    else:
        d = pd.read_parquet(p)
    out = pd.DataFrame({"s1": encode_id(d.source1_entity_id), "rid": encode_id(d.target_entity_id),
                        "fm": d.matcher_p.astype(np.float32).to_numpy(),
                        "ff": d.final_score.astype(np.float32).to_numpy(),
                        "fce": d.ce_logit.fillna(-20).astype(np.float32).to_numpy(),
                        "fband": d.in_band.astype(np.int8).to_numpy(),
                        "fl": d.linked.astype(np.int8).to_numpy()})
    if "label" in d:
        out["fy"] = d.label.astype(np.int8).to_numpy()
    return out.drop_duplicates(["s1", "rid"])


def ours(fname):
    d = pd.read_parquet(os.path.join(WORK_DIR, fname))
    cols = [c for c in ("s1", "rid", "y", "p2") if c in d]
    return d[cols].rename(columns={"p2": "p_o"}).drop_duplicates(["s1", "rid"])


def join(o, f, s1_keep=None):
    if s1_keep is not None:
        o = o[np.isin(o.s1.to_numpy(), s1_keep)]; f = f[np.isin(f.s1.to_numpy(), s1_keep)]
    d = o.merge(f, on=["s1", "rid"], how="outer")
    d["has_o"] = d.p_o.notna().astype(np.int8); d["has_f"] = d.fm.notna().astype(np.int8)
    d["p_o"] = d.p_o.fillna(0).astype(np.float32)
    for c in ("fm", "ff"):
        d[c] = d[c].fillna(0).astype(np.float32)
    d["fce"] = d.fce.fillna(-20).astype(np.float32)
    d["fband"] = d.fband.fillna(0).astype(np.int8); d["fl"] = d.fl.fillna(0).astype(np.int8)
    d["p_avg"] = ((d.p_o + d.ff) / 2).astype(np.float32)
    d["p_max"] = np.maximum(d.p_o, d.ff); d["p_min"] = np.minimum(d.p_o, d.ff)
    for side, col in (("o", "p_o"), ("f", "ff")):
        g = d.groupby("s1", sort=False)[col]
        d[f"{side}_rank"] = g.rank(ascending=False, method="first").astype(np.int16)
        d[f"{side}_best"] = g.transform("max").astype(np.float32)
        d[f"{side}_gap"] = (d[f"{side}_best"] - d[col]).astype(np.float32)
        d[f"{side}_rid_rank"] = d.groupby("rid", sort=False)[col].rank(ascending=False, method="first").astype(np.int16)
    d["n_o"] = d.groupby("s1", sort=False).has_o.transform("sum").astype(np.int16)
    d["n_f"] = d.groupby("s1", sort=False).has_f.transform("sum").astype(np.int16)
    d["src"] = (d.rid.to_numpy() // 10 ** 12).astype(np.int8)
    return d.reset_index(drop=True)


def label(d, subsets):
    g = pd.concat([U.gt_subset(sb) for sb in subsets], ignore_index=True)[["s1", "rid"]].drop_duplicates()
    d = d.drop(columns=["y"], errors="ignore").merge(g.assign(y=np.int8(1)), on=["s1", "rid"], how="left")
    d["y"] = d.y.fillna(0).astype(np.int8)
    return d


def main():
    import lightgbm as lgb
    import postprocess as P
    t0 = time.time()
    fC = pipe_b("heldout_C_pairs_v4.parquet")
    C = np.unique(fC.s1.to_numpy())
    ids_tr, ctry_tr, nt_tr = P._universe(["m1", "m2"])
    ids_ho, ctry_ho, nt_ho = P._universe("holdout")
    s_tr = np.intersect1d(C, ids_tr); s_ho = np.intersect1d(C, ids_ho)
    log(f"split C: {len(C):,} S1; in our m1-m2: {len(s_tr):,}; in our holdout: {len(s_ho):,}")
    tr = label(join(ours(f"scores_train_oof_{BASE}.parquet"), fC, s_tr), ["m1", "m2"])
    ho = label(join(ours(f"scores_train_holdout_{BASE}.parquet"), fC, s_ho), ["holdout"])
    if RICH:
        tr = enrich(tr, "train", ["m1", "m2"], "scores_train_oof_b.parquet", "ce_stack.parquet")
        ho = enrich(ho, "train", ["holdout"], "scores_train_holdout_b.parquet", "ce_holdout.parquet")
        FEATS.extend(RFEATS)
    if NAME:
        import name_feats as NF
        tr = add_names(tr, "train", "trC"); ho = add_names(ho, "train", "hoC")
        FEATS.extend(NF.NFEATS)
    if SPEC:
        tr = add_spec(tr, "spec_oof.parquet"); ho = add_spec(ho, "spec_holdout.parquet")
        FEATS.extend(["p_spec", "has_spec", "spec_gap"])
    log(f"train pairs {len(tr):,} (ours {tr.has_o.mean():.3f}, B {tr.has_f.mean():.3f}, pos {tr.y.mean():.3f}); "
        f"holdout pairs {len(ho):,}")
    prm_lgb = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                   feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
    X = tr[FEATS].to_numpy(np.float32); y = tr.y.to_numpy()
    fold = (tr.s1.to_numpy() // 7) % 2
    tr["q"] = 0.0
    for h in (0, 1):
        m = lgb.train(prm_lgb, lgb.Dataset(X[fold != h], y[fold != h]), 500)
        tr.loc[fold == h, "q"] = m.predict(X[fold == h])
    full = lgb.train(prm_lgb, lgb.Dataset(X, y), 500)
    full.save_model(os.path.join(MD, "ensemble.txt"))
    ho["q"] = full.predict(ho[FEATS].to_numpy(np.float32))
    fulls = None
    if SPLIT:
        grp = lambda d: (d.b_empty.fillna(0).to_numpy() == 1).astype(int)
        gtr, gho = grp(tr), grp(ho)
        fulls = {}
        for g in (0, 1):
            mg = gtr == g
            Xg, yg, fg = X[mg], y[mg], fold[mg]
            qg = np.zeros(mg.sum())
            for h in (0, 1):
                m = lgb.train(prm_lgb, lgb.Dataset(Xg[fg != h], yg[fg != h]), 500)
                qg[fg == h] = m.predict(Xg[fg == h])
            tr.loc[mg, "q"] = qg
            fulls[g] = lgb.train(prm_lgb, lgb.Dataset(Xg, yg), 500)
            ho.loc[gho == g, "q"] = fulls[g].predict(ho.loc[gho == g, FEATS].to_numpy(np.float32))
            log(f"  specialist blender group empty={g}: {int(mg.sum()):,} train pairs (pos {yg.mean():.3f})")
    gain = dict(zip(FEATS, full.feature_importance("gain")))
    tot = sum(gain.values())
    log("blender gain: " + ", ".join(f"{k}={v/tot:.3f}" for k, v in sorted(gain.items(), key=lambda t: -t[1])[:8]))

    # restricted universes (S1 of C only), true-pair counts from the ground truth
    nt_tr_c = nt_tr[nt_tr.index.isin(s_tr)]; nt_ho_c = nt_ho[nt_ho.index.isin(s_ho)]

    def f05(d, keep, nt, ids):
        return P.macro_f05(d, keep, nt, ids)[0]

    def tune(d, col, ctry, nt, ids):
        x = d[["s1", "rid", "y"]].assign(p2=d[col].to_numpy())
        best, bs = None, -1
        for thr in np.round(np.arange(0.3, 0.96, 0.025), 3):
            for fb in (1.1, 0.2, 0.3, 0.4):
                prm = {"thr": float(thr), "rel": 0.0, "fb": fb, "caps": True}
                s = f05(x, P.apply_rule(x, prm, ctry), nt, ids)
                if s > bs:
                    bs, best = s, prm
        for gamma in (0.75, 1.0, 1.25, 1.5, 2.0):
            for floor in (0.0, 0.2, 0.3):
                prm = {"mode": "efo", "gamma": gamma, "floor": floor, "extra": 0.02, "caps": True}
                s = f05(x, P.apply_rule(x, prm, ctry), nt, ids)
                if s > bs:
                    bs, best = s, prm
        return best, bs

    res = {}
    dec_o = json.load(open(os.path.join(MD, f"decision_{BASE}.json")))
    xo = ho[ho.has_o == 1][["s1", "rid", "y"]].assign(p2=ho.p_o[ho.has_o == 1].to_numpy())
    res["ours (own tuned rule)"] = f05(xo, P.apply_rule(xo, dec_o, ctry_ho), nt_ho_c, s_ho)
    xf = ho[ho.has_f == 1]
    res["pipeline B (its link flag)"] = f05(xf, xf.fl.to_numpy().astype(bool), nt_ho_c, s_ho)
    b_avg, s_avg = tune(tr, "p_avg", ctry_tr, nt_tr_c, s_tr)
    xa = ho[["s1", "rid", "y"]].assign(p2=ho.p_avg.to_numpy())
    res["mean of final scores"] = f05(xa, P.apply_rule(xa, b_avg, ctry_ho), nt_ho_c, s_ho)
    delta = 0.0
    if SPLIT:                                   # score shift for empty-address pairs, chosen on out-of-fold data
        lg = lambda q: np.log(np.clip(q, 1e-7, 1 - 1e-7) / (1 - np.clip(q, 1e-7, 1 - 1e-7)))
        b0, _ = tune(tr, "q", ctry_tr, nt_tr_c, s_tr)
        best_d, best_s = 0.0, -1
        for dlt in (-1.5, -1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0, 1.5):
            x = tr[["s1", "rid", "y"]].assign(p2=1 / (1 + np.exp(-(lg(tr.q.to_numpy()) + dlt * gtr))))
            sc = f05(x, P.apply_rule(x, b0, ctry_tr), nt_tr_c, s_tr)
            log(f"  empty-address shift {dlt:+.1f}: OOF {sc:.5f}")
            if sc > best_s + 1e-6:
                best_s, best_d = sc, dlt
        delta = best_d
        for d, gg in ((tr, gtr), (ho, gho)):
            d["q"] = 1 / (1 + np.exp(-(lg(d.q.to_numpy()) + delta * gg)))
        log(f"  chosen empty-address shift {delta:+.1f}")
    b_q, s_q = tune(tr, "q", ctry_tr, nt_tr_c, s_tr)
    xq = ho[["s1", "rid", "y"]].assign(p2=ho.q.to_numpy())
    res["stacked blend (LightGBM)"] = f05(xq, P.apply_rule(xq, b_q, ctry_ho), nt_ho_c, s_ho)
    log(f"decision rules: mean {b_avg} (OOF {s_avg:.5f}); stacked {b_q} (OOF {s_q:.5f})")
    print(f"\n[blend] macro F0.5 on the {len(s_ho):,} S1 in both split C and our holdout:")
    for k, v in res.items():
        print(f"  {k:28s} {v:.5f}")
    use_q = res["stacked blend (LightGBM)"] >= res["mean of final scores"]
    col, dec = ("q", b_q) if use_q else ("p_avg", b_avg)
    json.dump(dict(dec), open(os.path.join(MD, f"decision_{TAG}.json"), "w"), indent=1)
    json.dump({k: float(v) for k, v in res.items()}, open(os.path.join(MD, "blend_report.json"), "w"), indent=1)

    # test: every S1, union of both candidate sets
    ft = pipe_b("test_pairs_v4.parquet")
    te = join(ours(f"scores_test_{TEST_BASE}.parquet"), ft)
    if RICH:
        te = enrich(te, "test", [f"m{k}" for k in range(10)], "scores_test_b.parquet", "ce_test.parquet", TEST_CAND_DIR)
    if NAME:
        te = add_names(te, "test", "te")
    if SPEC:
        te = add_spec(te, "spec_test.parquet")
    if use_q and SPLIT:
        gte = (te.b_empty.fillna(0).to_numpy() == 1).astype(int)
        qv = np.zeros(len(te))
        for g in (0, 1):
            qv[gte == g] = fulls[g].predict(te.loc[gte == g, FEATS].to_numpy(np.float32))
        qv = np.clip(qv, 1e-7, 1 - 1e-7)
        te["q"] = 1 / (1 + np.exp(-(np.log(qv / (1 - qv)) + delta * gte)))
    else:
        te["q"] = full.predict(te[FEATS].to_numpy(np.float32)) if use_q else te.p_avg.to_numpy()
    log(f"test: {len(te):,} pairs (ours {te.has_o.mean():.3f}, B {te.has_f.mean():.3f}), {te.s1.nunique():,} S1")
    for d, fn in ((tr, "scores_train_oof"), (ho, "scores_train_holdout"), (te, "scores_test")):
        x = d[["s1", "rid"] + (["y"] if "y" in d else [])].assign(p2=d[col].astype(np.float32).to_numpy())
        x["p2_lgb"] = x.p2
        x.to_parquet(os.path.join(WORK_DIR, f"{fn}_{TAG}.parquet"))
    log(f"blend written with {'stacked' if use_q else 'mean'} scores, rule {dec} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
