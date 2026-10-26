"""Normalise every source file once and cache it as parquet (input to all blocking / matching).

    python build_normalized.py              # all splits and sources
    python build_normalized.py train 2      # one file

Output: WORK_DIR/norm_{split}_s{k}.parquet  (entity_id, country + normalize.NAME_COLS + ADDR_COLS)
Streams in chunks so peak memory stays low (~2-3 GB for a 5M-row file).
"""
import os, sys, time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR
from io_utils import read_source
from normalize import normalize_records, NAME_COLS, ADDR_COLS

CHUNK = 100_000


def _work(args):
    ids, ctr, names, addrs = args
    cols = normalize_records(names, addrs, ctr)
    cols = {"entity_id": ids, "country": ctr, **cols}
    return pa.table(cols)


def build(split, k, n_jobs=None):
    t0 = time.time()
    df = read_source(split, k)
    out = os.path.join(WORK_DIR, f"norm_{split}_s{k}.parquet")
    jobs = [(df.entity_id.iloc[i:i + CHUNK].tolist(), df.country.iloc[i:i + CHUNK].tolist(),
             df.business_name.iloc[i:i + CHUNK].tolist(), df.business_address.iloc[i:i + CHUNK].tolist())
            for i in range(0, len(df), CHUNK)]
    n = len(df)
    del df
    writer = None
    with Pool(n_jobs or os.cpu_count()) as pool:
        for t in pool.imap(_work, jobs):
            if writer is None:
                writer = pq.ParquetWriter(out, t.schema, compression="zstd")
            writer.write_table(t)
    writer.close()
    print(f"{split} S{k}: {n:,} rows -> {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        build(sys.argv[1], int(sys.argv[2]))
    else:
        for split in ("train", "test"):
            for k in (1, 2, 3):
                build(split, k)
