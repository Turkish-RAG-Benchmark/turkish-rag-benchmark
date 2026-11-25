"""README grafiklerini sonuç dosyalarından üretir.

Kullanım::

    python scripts/make_figures.py                       # results/v1_fixed → results/figures/v1.1/
    python scripts/make_figures.py --run results/v1_fixed --out results/figures/v1.1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Doğrulanmış kategorik palet (sabit sıra) ve metin/zemin renkleri
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SYSTEMS = [
    ("RAG-1 (bge-m3)", "bge-m3", "Dense"),
    ("RAG-2 (multilingual-e5-large)", "e5-large", "Dense"),
    ("RAG-3 (paraphrase-multilingual-mpnet-base-v2)", "mpnet ¹", "Dense"),
    ("RAG-4 (256/50)", "e5-large, 256", "Dense"),
    ("RAG-5 (1024/200)", "e5-large, 1024", "Dense"),
    ("RAG-6 (Header Based)", "e5-large, başlık", "Dense"),
    ("RAG-7 (BM25 Only)", "BM25", "BM25"),
    ("RAG-8 (Hybrid RRF)", "Hibrit (RRF)", "Hibrit"),
    ("ABL hybrid relative_score", "Hibrit (relative)", "Hibrit"),
    ("NEW Qwen3-Embedding-0.6B", "Qwen3-0.6B", "Dense"),
]
TYPE_STYLE = {"Dense": (SERIES[0], "o"), "BM25": (SERIES[1], "s"), "Hibrit": (SERIES[2], "D")}
DOMAINS = [("domain:matematik-fizik", "Mat.-Fizik"), ("domain:ml-ai", "ML-YZ"),
           ("domain:sosyal-bilimler", "Sosyal B."), ("domain:sosyal-toplum", "Sos.-Toplum"),
           ("domain:tip", "Tıp")]


FOOTNOTE = "¹ mpnet modelinin girdi sınırı 128 token; parçaların %98'i kesiliyor."


def footnote(fig, y: float = -0.09) -> None:
    fig.text(0.01, y, FOOTNOTE, fontsize=8, color=TEXT_2)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
        "xtick.color": TEXT_2, "ytick.color": TEXT, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
    })


def label(system: str) -> str:
    return next(lbl for s, lbl, _ in SYSTEMS if s == system)


def overall(summary: pd.DataFrame, out: Path) -> None:
    """Doc_MRR@5 ve Exact_MRR@5, %95 güven aralıklarıyla (nokta + hata çubuğu)."""
    s = summary[(summary.subset == "all") & (summary.k == 5) & (summary.metric == "MRR")]
    names = [n for n, _, _ in SYSTEMS if n in set(s.system)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, level, title in zip(axes, ["Doc", "Exact"], ["Doküman düzeyi (Doc_MRR@5)", "Cevap düzeyi (Exact_MRR@5)"]):
        d = s[s.level == level].set_index("system").loc[names]
        order = d.sort_values("value").index.tolist() if level == "Exact" else None
        if order is None:
            order = s[s.level == "Exact"].set_index("system").loc[names].sort_values("value").index.tolist()
        y = np.arange(len(order))
        for yi, name in zip(y, order):
            r = d.loc[name]
            kind = next(t for n, _, t in SYSTEMS if n == name)
            color, marker = TYPE_STYLE[kind]
            ax.plot([r.ci_low, r.ci_high], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(r.value, yi, marker=marker, ms=8, color=color, mec=SURFACE, mew=1.5, zorder=3)
            ax.text(r.ci_high + 0.006, yi, f"{r.value:.3f}", va="center", fontsize=8.5, color=TEXT_2)
        ax.set_yticks(y, [label(n) for n in order])
        ax.set_title(title, loc="left")
        lo, hi = d.ci_low.min(), d.ci_high.max()
        ax.set_xlim(lo - 0.02, hi + 0.05)
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", ms=8, label=t) for t, (c, m) in TYPE_STYLE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Sistemlerin genel performansı (K=5, %95 güven aralığı)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=TEXT)
    footnote(fig)
    fig.tight_layout()
    fig.savefig(out / "overall_performance.png")
    plt.close(fig)


def top_k(summary: pd.DataFrame, out: Path) -> None:
    """HitRate@K eğrileri (v1'in 8 sistemi)."""
    s = summary[(summary.subset == "all") & (summary.metric == "HitRate")]
    names = [n for n, _, _ in SYSTEMS[:8]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, level, title in zip(axes, ["Doc", "Exact"], ["Doküman düzeyi (Doc_HitRate@K)", "Cevap düzeyi (Exact_HitRate@K)"]):
        for i, name in enumerate(names):
            d = s[(s.level == level) & (s.system == name)].sort_values("k")
            ax.plot(d.k, d.value, color=SERIES[i], marker=MARKERS[i], ms=7, lw=2,
                    mec=SURFACE, mew=1.2, label=label(name))
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xlabel("K")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(0.76, 1.0)
    axes[1].set_ylim(0.15, 0.62)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("K değerine göre başarı", x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    footnote(fig, -0.15)
    fig.tight_layout()
    fig.savefig(out / "top_k_curves.png")
    plt.close(fig)


def v1_vs_fixed(summary_wide: pd.DataFrame, v1_csv: Path, out: Path) -> None:
    """v1 orijinal skorları ile v1.1 skorları (dumbbell)."""
    old = pd.read_csv(v1_csv)
    key = lambda s: re.match(r"(RAG-\d+)", s).group(1)  # noqa: E731
    old["key"] = old.System.map(key)
    new = summary_wide.copy()
    new = new[new.system.str.match(r"RAG-\d")]
    new["key"] = new.system.map(key)
    m = old.merge(new, on="key", suffixes=("_v1", "_v11"))
    m["label"] = m.system.map(label)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, metric, title in zip(axes, ["Doc_MRR@5", "Exact_MRR@5"], ["Doc_MRR@5", "Exact_MRR@5"]):
        d = m.sort_values(f"Exact_MRR@5_v11")
        y = np.arange(len(d))
        for yi, (_, r) in zip(y, d.iterrows()):
            a, b = r[f"{metric}_v1"], r[f"{metric}_v11"]
            ax.plot([a, b], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            ax.plot(a, yi, "o", ms=8, color=SERIES[1], mec=SURFACE, mew=1.5, zorder=3)
            ax.plot(b, yi, "s", ms=8, color=SERIES[0], mec=SURFACE, mew=1.5, zorder=3)
        ax.set_yticks(y, d.label)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=SERIES[1], marker="o", ls="", ms=8, label="v1 (orijinal)"),
               plt.Line2D([], [], color=SERIES[0], marker="s", ls="", ms=8, label="v1.1 (düzeltilmiş)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("v1 ile v1.1 karşılaştırması (K=5)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=TEXT)
    footnote(fig, -0.1)
    fig.tight_layout()
    fig.savefig(out / "v1_vs_v1.1.png")
    plt.close(fig)


def by_domain(summary: pd.DataFrame, out: Path) -> None:
    """Alanlara göre Exact_MRR@5 ısı haritası."""
    s = summary[(summary.k == 5) & (summary.metric == "MRR") & (summary.level == "Exact")]
    names = [n for n, _, _ in SYSTEMS if n in set(s.system)]
    grid = np.array([[s[(s.system == n) & (s.subset == d)].value.iloc[0] for d, _ in DOMAINS] for n in names])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=max(0.7, grid.max()), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if v >= 0.30 else TEXT)
    ax.set_xticks(range(len(DOMAINS)), [d for _, d in DOMAINS])
    ax.set_yticks(range(len(names)), [label(n) for n in names])
    ax.tick_params(axis="x", colors=TEXT)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(DOMAINS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.set_title("Alanlara göre cevap bulma başarısı (Exact_MRR@5)", loc="left")
    fig.tight_layout()
    footnote(fig, -0.02)
    fig.savefig(out / "exact_mrr_by_domain.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/v1_fixed")
    p.add_argument("--v1", default="results/tables/rag_evaluation_results.csv")
    p.add_argument("--out", default="results/figures/v1.1")
    args = p.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(run / "summary.csv")
    style()
    overall(summary, out)
    top_k(summary, out)
    v1_vs_fixed(pd.read_csv(run / "summary_wide.csv"), Path(args.v1), out)
    by_domain(summary, out)
    print("Grafikler:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
