import pandas as pd

# Kyte–Doolittle hydrophobicity 
AA_HYDRO = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

AA_CHARGE = {
    "K": +1.0,
    "R": +1.0,
    "H": +0.1,  # partiellement protonée
    "D": -1.0,
    "E": -1.0,
}


def compute_cdr3_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seqs = df["vh_cdr3"].astype(str)

    df["cdr3_len"] = seqs.str.len()

    df["cdr3_hydro_mean"] = seqs.apply(
        lambda s: sum(AA_HYDRO.get(a, 0.0) for a in s) / max(len(s), 1)
    )

    df["cdr3_charge"] = seqs.apply(
        lambda s: sum(AA_CHARGE.get(a, 0.0) for a in s)
    )

    df["cdr3_aromatic_frac"] = seqs.apply(
        lambda s: sum(a in {"F", "Y", "W"} for a in s) / max(len(s), 1)
    )

    return df
