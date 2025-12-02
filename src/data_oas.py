"""
Utilities to load and filter antibody sequences from OAS
using the Hugging Face dataset 'bloyal/oas-paired-sequence-data'.
"""

import pandas as pd
from datasets import load_dataset

def inspect_oas_columns():
    """Print columns of the OAS human train split (debug only)."""
    ds = load_dataset("bloyal/oas-paired-sequence-data", "human", split="train")
    print(ds.column_names)
