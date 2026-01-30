"""第三问推荐（Part 3）：生成用于“推荐 Percent vs Rank + Judges' Save”的汇总图。

输出：
- recommendation_outputs/figures/fig_fan_favor_proxy_percent_vs_rank.png
- recommendation_outputs/figures/fig_case_counterfactual_final_place.png

依赖数据：
- method_compare_outputs2.1/weekly_method_comparison.csv
- controversy_outputs/case_summary.csv

运行：
D:/python/python.exe codeMCM/2026/c/recommendation_outputs/build_recommendation_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _stable_sort(df: pd.DataFrame, by: list[str], ascending: list[bool]) -> pd.DataFrame:
    return df.sort_values(by=by, ascending=ascending, kind="mergesort")


def main() -> None:
    here = Path(__file__).resolve().parent
    base_dir = here.parent

    weekly_path = base_dir / "method_compare_outputs2.1" / "weekly_method_comparison.csv"
    case_path = base_dir / "controversy_outputs" / "case_summary.csv"

    out_dir = here / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weekly_path.exists():
        raise FileNotFoundError(f"missing: {weekly_path}")
    if not case_path.exists():
        raise FileNotFoundError(f"missing: {case_path}")

    weekly = pd.read_csv(weekly_path)
    cases = pd.read_csv(case_path)

    # ---------- Figure 1: fan-favor proxy distribution (Percent vs Rank) ----------
    # Use per-week delta_mean among predicted eliminations.
    keep = weekly[["percent_elim_delta_mean", "rank_elim_delta_mean"]].copy()
    keep = keep.dropna()

    # For a clean plot, clip extreme values (rare with small n)
    for col in keep.columns:
        keep[col] = pd.to_numeric(keep[col], errors="coerce")
    keep = keep.replace([np.inf, -np.inf], np.nan).dropna()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(9.6, 4.8))

    # Boxplot + jitter for readability
    data = [keep["percent_elim_delta_mean"].values, keep["rank_elim_delta_mean"].values]
    ax.boxplot(
        data,
        labels=["Percent", "Rank"],
        showmeans=True,
        meanline=True,
        widths=0.45,
        patch_artist=True,
        boxprops={"facecolor": "#cce5ff", "edgecolor": "#1f77b4"},
        medianprops={"color": "#1f77b4", "linewidth": 2},
        meanprops={"color": "#d62728", "linewidth": 2},
    )

    rng = np.random.default_rng(42)
    for i, y in enumerate(data, start=1):
        x = rng.normal(i, 0.06, size=len(y))
        ax.scatter(x, y, s=10, alpha=0.18, color="#333333")

    ax.axhline(0.0, color="#666666", linewidth=1)
    ax.set_title("Fan-favor proxy across all elimination weeks (Percent vs Rank)")
    ax.set_ylabel("mean(fan percentile − judge percentile) among predicted eliminations")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    fig1_path = out_dir / "fig_fan_favor_proxy_percent_vs_rank.png"
    fig.savefig(fig1_path)
    plt.close(fig)

    # ---------- Figure 2: case counterfactual final place (four scenarios) ----------
    # Four scenarios from case_summary.csv
    show = cases[[
        "season",
        "celebrity_name",
        "actual_placement",
        "percent_final_place",
        "rank_final_place",
        "percent_judges_save_final_place",
        "rank_judges_save_final_place",
    ]].copy()

    # Keep order as in file, but create a short label
    show["label"] = show.apply(lambda r: f"S{int(r['season'])} {str(r['celebrity_name'])}", axis=1)

    # Convert to numeric where possible
    for col in [
        "actual_placement",
        "percent_final_place",
        "rank_final_place",
        "percent_judges_save_final_place",
        "rank_judges_save_final_place",
    ]:
        show[col] = pd.to_numeric(show[col], errors="coerce")

    # Plot as grouped bars: lower place number is better, so invert y-axis in plot
    fig, ax = plt.subplots(figsize=(11.0, 5.4))

    x = np.arange(len(show))
    width = 0.18

    series = [
        ("Actual", show["actual_placement"].values, "#4c78a8"),
        ("Percent", show["percent_final_place"].values, "#72b7b2"),
        ("Rank", show["rank_final_place"].values, "#f58518"),
        ("Percent + Save", show["percent_judges_save_final_place"].values, "#54a24b"),
        ("Rank + Save", show["rank_judges_save_final_place"].values, "#e45756"),
    ]

    offsets = np.linspace(-2, 2, num=len(series)) * width

    for (name, y, color), dx in zip(series, offsets, strict=False):
        ax.bar(x + dx, y, width=width, label=name, color=color, alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(show["label"], rotation=0)
    ax.set_ylabel("Final place (1=best)")
    ax.set_title("Counterfactual season outcomes for controversy cases")
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()

    fig2_path = out_dir / "fig_case_counterfactual_final_place.png"
    fig.savefig(fig2_path)
    plt.close(fig)

    print("[OK] wrote:")
    print(" -", fig1_path)
    print(" -", fig2_path)


if __name__ == "__main__":
    main()
