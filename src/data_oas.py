"""
Data loading helpers for the Observed Antibody Space dataset.

This module wraps the Hugging Face dataset ``bloyal/oas-paired-sequence-data``
for convenience when experimenting with antibody developability tasks.
"""

from typing import Iterable, List

import pandas as pd
from datasets import Dataset, load_dataset


def inspect_oas_columns(split: str = "train") -> List[str]:
    """
    Load the OAS human split and print its column names.

    Parameters
    ----------
    split:
        The dataset split to inspect (e.g. "train" or "validation").

    Returns
    -------
    list of str
        The column names for the requested split, in order.
    """
    ds = load_dataset("bloyal/oas-paired-sequence-data", "human", split=split)
    print(ds.column_names)
    return ds.column_names


def _choose_column(column_names: Iterable[str], options: List[str], label: str) -> str:
    """
    Select the first matching column name from a list of options.

    Parameters
    ----------
    column_names:
        Available columns in the dataset.
    options:
        Candidate column names to try in order.
    label:
        Human-readable description of the field for error reporting.

    Raises
    ------
    ValueError
        If none of the candidate names exist in the dataset.
    """
    for candidate in options:
        if candidate in column_names:
            return candidate
    raise ValueError(f"None of the candidate columns for {label} were found: {options}")


def load_oas_human_heavy(split: str = "train") -> pd.DataFrame:
    """
    Load human heavy-chain sequences from the OAS paired dataset.

    The function:
      - loads the HF dataset "bloyal/oas-paired-sequence-data" with config "human",
      - inspects the available columns,
      - maps them to a fixed schema,
      - filters isotypes to {"IgM", "IgG"},
      - returns a pandas DataFrame with columns:
        ['sequence_heavy', 'isotype', 'subject_id', 'study_id'].

    Parameters
    ----------
    split:
        Which split of the dataset to load, e.g. "train" or "validation".

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns ['sequence_heavy', 'isotype', 'subject_id', 'study_id'].
    """
    ds: Dataset = load_dataset("bloyal/oas-paired-sequence-data", "human", split=split)
    column_names = ds.column_names

    # À AJUSTER APRÈS INSPECTION DES COLONNES
    heavy_sequence_col = _choose_column(
        column_names,
        [
            "sequence_heavy",
            "sequence_alignment_aa_heavy",
            "sequence_aa_heavy",
            "sequence_heavy_aa",
        ],
        label="heavy-chain amino acid sequence",
    )
    isotype_col = _choose_column(
        column_names,
        [
            "isotype",
            "c_call_heavy",
            "heavy_isotype",
        ],
        label="isotype",
    )
    subject_col = _choose_column(
        column_names,
        ["subject_id", "subject"],
        label="subject ID",
    )
    study_col = _choose_column(
        column_names,
        ["study_id", "study"],
        label="study ID",
    )

    # Filtrer IgM / IgG
    ds_filtered = ds.filter(lambda row: row[isotype_col] in {"IgM", "IgG"})

    # Conversion en DataFrame pandas
    df = ds_filtered.to_pandas()[
        [heavy_sequence_col, isotype_col, subject_col, study_col]
    ].copy()

    # Normaliser les noms de colonnes
    df = df.rename(
        columns={
            heavy_sequence_col: "sequence_heavy",
            isotype_col: "isotype",
            subject_col: "subject_id",
            study_col: "study_id",
        }
    )

    return df.reset_index(drop=True)
