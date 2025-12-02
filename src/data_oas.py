from typing import List
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

# Optionnel : mettre le cache sur E: si tu as un répertoire dédié
CACHE_DIR = Path("E:/hf_cache")  # ou None si tu utilises HF_DATASETS_CACHE


def inspect_oas_columns(split: str = "train") -> List[str]:
    """
    Inspect columns of the OAS paired human subset.
    """
    ds = load_dataset(
        "bloyal/oas-paired-sequence-data",
        "human",
        split=split,
        cache_dir=str(CACHE_DIR),
    )
    print(ds.column_names)
    return ds.column_names


def load_oas_human_paired(split: str = "train") -> pd.DataFrame:
    """
    Load human paired VH/VL sequences from the OAS HF dataset.

    Returns a tidy pandas DataFrame with:
      - run_id
      - vh_seq, vh_cdr1, vh_cdr2, vh_cdr3
      - vl_seq, vl_cdr1, vl_cdr2, vl_cdr3
    """
    ds: Dataset = load_dataset(
        "bloyal/oas-paired-sequence-data",
        "human",
        split=split,
        cache_dir=str(CACHE_DIR),
    )

    cols = [
        "data_unit",
        "sequence_alignment_aa_heavy",
        "cdr1_aa_heavy",
        "cdr2_aa_heavy",
        "cdr3_aa_heavy",
        "sequence_alignment_aa_light",
        "cdr1_aa_light",
        "cdr2_aa_light",
        "cdr3_aa_light",
    ]

    df = ds.to_pandas()[cols].copy()

    df = df.rename(
        columns={
            "data_unit": "run_id",
            "sequence_alignment_aa_heavy": "vh_seq",
            "cdr1_aa_heavy": "vh_cdr1",
            "cdr2_aa_heavy": "vh_cdr2",
            "cdr3_aa_heavy": "vh_cdr3",
            "sequence_alignment_aa_light": "vl_seq",
            "cdr1_aa_light": "vl_cdr1",
            "cdr2_aa_light": "vl_cdr2",
            "cdr3_aa_light": "vl_cdr3",
        }
    )

    return df.reset_index(drop=True)
