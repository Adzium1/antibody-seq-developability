# src/generate_report.py

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_oas import load_oas_human_paired
from src.features_cdr import compute_cdr3_features


# ---------- 1. Small helpers ----------

def filter_reasonable_cdr3(
    df: pd.DataFrame,
    min_len: int = 5,
    max_len: int = 30,
) -> pd.DataFrame:
    """
    Filter out VH CDR3 sequences with extreme lengths.
    Keeps rows where len(vh_cdr3) is between min_len and max_len (inclusive).
    """
    df = df.copy()
    lengths = df["vh_cdr3"].str.len()
    return df[lengths.between(min_len, max_len)]


def build_devscore_dataset(
    n_sample: int = 200_000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OAS, filter CDR3 lengths, downsample, compute CDR3 features,
    and create a toy 'devscore' + binary label dev_label.
    """
    # 1) Load + filter
    df = load_oas_human_paired()
    df = filter_reasonable_cdr3(df, min_len=5, max_len=30)

    # 2) Subsample for speed
    if n_sample is not None and len(df) > n_sample:
        df = df.sample(n_sample, random_state=random_state).reset_index(drop=True)

    # 3) Compute CDR3 physico-chemical features
    df_feat = compute_cdr3_features(df)

    # 4) Standardize core features and build devscore
    for col in ["cdr3_len", "cdr3_hydro_mean", "cdr3_charge"]:
        mean = df_feat[col].mean()
        std = df_feat[col].std()
        df_feat[col + "_z"] = (df_feat[col] - mean) / std

    df_feat["devscore"] = (
        df_feat["cdr3_len_z"]
        + df_feat["cdr3_hydro_mean_z"]
        + df_feat["cdr3_charge_z"].abs()
    )

    # 5) Binarize using quantiles (30/70)
    q_low, q_high = df_feat["devscore"].quantile([0.3, 0.7])
    df_bin = df_feat[
        (df_feat["devscore"] <= q_low) | (df_feat["devscore"] >= q_high)
    ].copy()
    df_bin["dev_label"] = (df_bin["devscore"] >= q_high).astype(int)

    return df_feat, df_bin


# ---------- 2. Plots ----------

def plot_eda(df_feat: pd.DataFrame, out_dir: Path) -> None:
    """Generate basic EDA plots and save them as PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CDR3 length distribution
    plt.figure(figsize=(6, 4))
    df_feat["cdr3_len"].hist(bins=30)
    plt.xlabel("CDR-H3 length (aa)")
    plt.ylabel("Count")
    plt.title("CDR-H3 length distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "cdr3_length_hist.png", dpi=150)
    plt.close()

    # Hydrophobicity distribution
    plt.figure(figsize=(6, 4))
    df_feat["cdr3_hydro_mean"].hist(bins=30)
    plt.xlabel("Mean Kyte-Doolittle hydrophobicity")
    plt.ylabel("Count")
    plt.title("CDR-H3 hydrophobicity distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "cdr3_hydro_hist.png", dpi=150)
    plt.close()

    # Devscore distribution
    plt.figure(figsize=(6, 4))
    df_feat["devscore"].hist(bins=30)
    plt.xlabel("DevScore")
    plt.ylabel("Count")
    plt.title("Toy developability proxy (DevScore) distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "devscore_hist.png", dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_proba, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="LogReg (baseline)")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve – DevScore baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- 3. Baseline model ----------

def train_baseline(df_bin: pd.DataFrame) -> dict:
    """
    Train a logistic regression baseline on CDR3 features and return metrics.
    """
    features = ["cdr3_len", "cdr3_hydro_mean", "cdr3_charge", "cdr3_aromatic_frac"]
    X = df_bin[features].values
    y = df_bin["dev_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1_000)),
        ]
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    report_txt = classification_report(y_test, y_pred)

    metrics = {
        "roc_auc": roc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "class_balance": {
            "train": float(y_train.mean()),
            "test": float(y_test.mean()),
        },
        "classification_report": report_txt,
        "model": clf,
        "y_test": y_test,
        "y_proba": y_proba,
    }
    return metrics


# ---------- 4. Report generation ----------

def generate_markdown_report(
    out_dir: Path,
    df_feat: pd.DataFrame,
    df_bin: pd.DataFrame,
    metrics: dict,
) -> None:
    """Create a Markdown report summarizing EDA + baseline results."""
    report_path = out_dir / "report.md"
    today = date.today().isoformat()

    n_total = len(df_feat)
    n_bin = len(df_bin)

    length_desc = df_feat["cdr3_len"].describe()
    hydro_desc = df_feat["cdr3_hydro_mean"].describe()
    dev_desc = df_feat["devscore"].describe()

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Antibody CDR-H3 developability proxy – EDA & baseline ({today})\n\n")

        f.write("## 1. Dataset overview\n\n")
        f.write(f"- Total sequences used for features: **{n_total:,}**\n")
        f.write(f"- Sequences used after DevScore binarisation: **{n_bin:,}**\n")
        f.write(
            "- VH CDR3 length filter applied: **[5, 30] amino acids** "
            "(extreme lengths discarded)\n\n"
        )

        f.write("### 1.1 CDR-H3 length statistics\n\n")
        f.write("```text\n")
        f.write(str(length_desc))
        f.write("\n```\n\n")

        f.write("### 1.2 CDR-H3 hydrophobicity statistics\n\n")
        f.write("```text\n")
        f.write(str(hydro_desc))
        f.write("\n```\n\n")

        f.write("### 1.3 DevScore statistics (toy developability proxy)\n\n")
        f.write("```text\n")
        f.write(str(dev_desc))
        f.write("\n```\n\n")

        f.write("## 2. EDA figures\n\n")
        f.write("![CDR3 length distribution](cdr3_length_hist.png)\n\n")
        f.write("![CDR3 hydrophobicity distribution](cdr3_hydro_hist.png)\n\n")
        f.write("![DevScore distribution](devscore_hist.png)\n\n")

        f.write("## 3. Baseline model: logistic regression on CDR3 features\n\n")
        f.write(
            "Features used:\n\n"
            "- `cdr3_len`\n"
            "- `cdr3_hydro_mean`\n"
            "- `cdr3_charge`\n"
            "- `cdr3_aromatic_frac`\n\n"
        )

        f.write("### 3.1 Data split and class balance\n\n")
        f.write(
            f"- Train size: **{metrics['n_train']:,}** samples\n"
            f"- Test size: **{metrics['n_test']:,}** samples\n"
        )
        f.write(
            f"- Positive class proportion (train): "
            f"**{metrics['class_balance']['train']:.3f}**\n"
        )
        f.write(
            f"- Positive class proportion (test): "
            f"**{metrics['class_balance']['test']:.3f}**\n\n"
        )

        f.write("### 3.2 Performance metrics\n\n")
        f.write(f"- ROC–AUC: **{metrics['roc_auc']:.3f}**\n")
        f.write(f"- Accuracy: **{metrics['accuracy']:.3f}**\n")
        f.write(f"- Precision (class 1): **{metrics['precision']:.3f}**\n")
        f.write(f"- Recall (class 1): **{metrics['recall']:.3f}**\n")
        f.write(f"- F1 (class 1): **{metrics['f1']:.3f}**\n\n")

        f.write("### 3.3 Classification report\n\n")
        f.write("```text\n")
        f.write(metrics["classification_report"])
        f.write("\n```\n\n")

        f.write("### 3.4 ROC curve\n\n")
        f.write("![ROC curve](roc_curve.png)\n\n")

        f.write(
            "## 4. Notes\n\n"
            "- DevScore is a **toy proxy**, combining standardized CDR3 length, "
            "hydrophobicity and absolute charge.\n"
            "- Labels here do **not** correspond to experimental developability "
            "issues, only to sequence-derived risk heuristics.\n"
        )


def main() -> None:
    # 1) Create artifact directory with today's date
    artifacts_root = Path("artifacts")
    today_str = date.today().isoformat()  # e.g. "2025-12-02"
    out_dir = artifacts_root / today_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Build dataset (EDA + labels)
    df_feat, df_bin = build_devscore_dataset(n_sample=200_000, random_state=0)

    # 3) EDA figures
    plot_eda(df_feat, out_dir)

    # 4) Baseline model + ROC curve
    metrics = train_baseline(df_bin)
    roc_path = out_dir / "roc_curve.png"
    plot_roc_curve(metrics["y_test"], metrics["y_proba"], roc_path)

    # 5) Markdown report
    generate_markdown_report(out_dir, df_feat, df_bin, metrics)

    print(f"Report and figures written to: {out_dir}")


if __name__ == "__main__":
    main()
