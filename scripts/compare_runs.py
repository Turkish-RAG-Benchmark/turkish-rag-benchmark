"""İki koşunun özet tablolarını "RAG-n" önekine göre eşleştirip farkları gösterir.

Kullanım::

    python scripts/compare_runs.py results/runs/v1_compat/<zaman> results/runs/v1_fixed/<zaman> \
        --out results/v1_comparison.csv

İsteğe bağlı olarak v1 notebook'unun orijinal CSV'si de "eski" taraf olarak verilebilir::

    python scripts/compare_runs.py results/tables/rag_evaluation_results.csv results/runs/v1_fixed/<zaman>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

METRICS = ["Doc_HitRate@5", "Doc_MRR@5", "Exact_HitRate@5", "Exact_MRR@5", "Exact_HitRate@10"]


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p / "summary_wide.csv" if p.is_dir() else p)
    if "System" in df.columns:  # v1 notebook çıktısı
        df = df.rename(columns={"System": "system"})
    df["key"] = df["system"].map(lambda s: (re.match(r"(RAG-\d+)", s) or [None])[0])
    return df.dropna(subset=["key"]).set_index("key")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--out")
    args = p.parse_args()
    old, new = load(args.old), load(args.new)
    keys = [k for k in new.index if k in old.index]
    rows = []
    for k in keys:
        row = {"system": new.loc[k, "system"]}
        for m in METRICS:
            if m in old.columns and m in new.columns:
                row[f"{m} eski"] = round(float(old.loc[k, m]), 4)
                row[f"{m} yeni"] = round(float(new.loc[k, m]), 4)
                row[f"{m} Δ"] = round(float(new.loc[k, m] - old.loc[k, m]), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
