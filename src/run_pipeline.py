"""End-to-end driver: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python run_pipeline.py              # everything
    python run_pipeline.py --from 7     # resume from a step (see STEPS below)
    python run_pipeline.py --list

Set BER_DATA_DIR if `dataset/` is not next to this repo. Every step is also runnable on its own
(see README.md). Steps are idempotent: they overwrite their own outputs in work/ and output/.
"""
import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable, "-u", "-W", "ignore"]
TEST_PARTS = [f"m{k}" for k in range(10)]

TRAIN = ["m1", "m2", "m3", "m4", "m5"]          # training slices (50% of train); holdout (m0) is evaluation only
# final configuration: multi-slice matcher, stage-2 model choice by out-of-fold blend search,
# cross-encoder re-scoring of the uncertain band, tuned decision rule
os.environ.setdefault("BER_TRAIN_SUBSETS", ",".join(TRAIN))
os.environ.setdefault("BER_TUNE_SUBSETS", "m1,m2")
os.environ.setdefault("BER_SCORES", "b")
CE_ENV = {"BER_SCORES": "ce2", "BER_TUNE_SUBSETS": "m1,m2"}   # final scores = GBM + cross-encoder stack

STEPS = [
    ("learn transliteration map (train pairs)", [["learn_translit.py"]]),
    ("normalise all 6 source files", [["build_normalized.py"]]),
    ("key blocking: train slices + holdout, test",
     [["run_blocking.py", "--split", "train", "--subset", "all"],
      ["run_blocking.py", "--split", "train", "--subset", "holdout"],
      ["run_blocking.py", "--split", "test", "--subset", "all"]]),
    ("TF-IDF features train + test", [["tfidf_knn.py", "featurize", "train"], ["tfidf_knn.py", "featurize", "test"]]),
    ("TF-IDF nearest-neighbour search: holdout + training slices",
     [["tfidf_knn.py", "search", "train", s, "cpu"] for s in ["holdout"] + TRAIN]),
    ("candidate union + pruning model (fit on m1) + top-15 pruning for all train slices",
     [["union_trim.py", "build", "train", "m1"], ["union_trim.py", "build", "train", "holdout"],
      ["union_trim.py", "fit"]]
     + [c for s in TRAIN + ["holdout"] for c in (([] if s in ("m1", "holdout") else [["union_trim.py", "build", "train", s]])
                                                  + [["union_trim.py", "trim", "train", s, "15"]])]),
    ("pairwise features for all train slices", [["matcher.py", "features", "train", s] for s in TRAIN + ["holdout"]]),
    ("stage-1 matcher + stage-2 models (XGBoost, LightGBM, CatBoost; 5-fold CV grouped by S1)",
     [["train_big.py", "stage1"], ["train_big.py", "stage2"]]),
    ("stage-2 model weights by out-of-fold F0.5", [["blend.py"]]),
    ("test candidates: TF-IDF search + union + pruning, per tenth of test",
     [c for p in TEST_PARTS for c in (["tfidf_knn.py", "search", "test", p, "cpu"],
                                      ["union_trim.py", "build", "test", p],
                                      ["union_trim.py", "trim", "test", p, "15"])]),
    ("score test", [["train_big.py", "score_test"]]),
    ("cross-encoder A on the uncertain band: select pairs, fine-tune xlm-roberta-base on 180k pairs (m3-m5), score",
     [["cross_encoder.py", "band"], ["cross_encoder.py", "train"], ["cross_encoder.py", "score"]]),
    ("cross-encoder B: per-slice rival context, 500k field-labelled pairs, score; stacker with A + B (m1-m2)",
     [["cross_encoder.py", "band"], ["cross_encoder.py", "train"], ["cross_encoder.py", "score"],
      ["cross_encoder.py", "stack"]], {"BER_CE_VER": "2", "BER_CE_MAX_TRAIN": "500000"}),
    ("decision rule: tune on out-of-fold stacked scores, report holdout", [["postprocess.py", "tune"]], CE_ENV),
    ("write output/matching_results.tsv + output/candidate_pairs.tsv", [["postprocess.py", "write"]], CE_ENV),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=len(STEPS))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for i, (name, cmds, *env) in enumerate(STEPS, 1):
        if a.list:
            print(f"{i:>2}. {name}")
            continue
        if not (a.start <= i <= a.end):
            continue
        t0 = time.time()
        print(f"\n===== step {i}: {name}", flush=True)
        for c in cmds:
            r = subprocess.run(PY + c, cwd=HERE, env=dict(os.environ, **(env[0] if env else {})))
            if r.returncode != 0:
                sys.exit(f"step {i} failed: {' '.join(c)} (exit {r.returncode})")
        print(f"===== step {i} done ({time.time()-t0:.0f}s)", flush=True)
