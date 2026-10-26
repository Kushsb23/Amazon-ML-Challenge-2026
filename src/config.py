"""Paths and shared constants.

Default layout:
    <parent>/
        dataset/{train,test}/...           <- challenge data (not in this repo)
        <this repo>/
            src/  notebooks/  work/  output/
Override with env vars BER_DATA_DIR / BER_WORK_DIR / BER_OUT_DIR (e.g. on SageMaker or Colab).
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("BER_DATA_DIR", os.path.join(os.path.dirname(_REPO), "dataset"))
WORK_DIR = os.environ.get("BER_WORK_DIR", os.path.join(_REPO, "work"))
OUT_DIR = os.environ.get("BER_OUT_DIR", os.path.join(_REPO, "output"))

HOLDOUT_MOD = 10   # holdout = train S1 ids with int(id[3:]) % 10 == 0  (~10%, shared by whole team)

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
