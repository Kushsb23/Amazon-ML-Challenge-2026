"""Fast, lossless IO for the challenge TSVs and the two submission files."""
import os
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.compute as pc

from config import DATA_DIR

_PO = pv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False)
_CO = pv.ConvertOptions(strings_can_be_null=False, quoted_strings_can_be_null=False)


def read_tsv(path, columns=None) -> pd.DataFrame:
    """Read a challenge TSV. No quote handling (names contain quotes), empty stays ''."""
    co = _CO if columns is None else pv.ConvertOptions(strings_can_be_null=False, quoted_strings_can_be_null=False,
                                                       include_columns=list(columns))
    return pv.read_csv(path, parse_options=_PO, convert_options=co).to_pandas().astype("string[pyarrow]")


def read_source(split: str, k: int, columns=None) -> pd.DataFrame:
    return read_tsv(os.path.join(DATA_DIR, split, f"{split}_source{k}.tsv"), columns)


def read_gt_pairs() -> pd.DataFrame:
    """Ground truth exploded to one row per (s1_id, rid). Singletons get no rows."""
    gt = read_tsv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"))
    arr = pa.array(gt.matched_entity_ids.to_numpy(dtype=object), type=pa.string())
    sp = pc.split_pattern(arr, ",")
    lens = pc.list_value_length(sp).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(sp)
    keep = pc.not_equal(pc.utf8_trim_whitespace(flat), "")
    s1 = pd.Series(gt.source1_entity_id.to_numpy(dtype=object)).repeat(lens).to_numpy()
    s1 = s1[keep.to_numpy(zero_copy_only=False)]
    rid = pc.utf8_trim_whitespace(pc.filter(flat, keep))
    return pd.DataFrame({"s1_id": pd.array(s1, dtype="string[pyarrow]"),
                         "rid": pd.array(rid, dtype="string[pyarrow]")})


def write_id_lists(df_pairs: pd.DataFrame, all_s1_ids, path: str, col_name: str):
    """Write the submission format: one row per S1 id, comma-joined ids (possibly empty).

    df_pairs: columns s1_id, rid (already de-duplicated / filtered).
    col_name: 'matched_entity_ids' or 'candidate_entity_ids'.
    """
    d = df_pairs.drop_duplicates(["s1_id", "rid"])
    lists = d.groupby("s1_id", sort=False).rid.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": pd.Series(list(all_s1_ids), dtype=object)})
    out[col_name] = out.source1_entity_id.map(lists).fillna("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for a, b in zip(out.source1_entity_id, out[col_name]):
            f.write(f"{a}\t{b}\n")
    return out
