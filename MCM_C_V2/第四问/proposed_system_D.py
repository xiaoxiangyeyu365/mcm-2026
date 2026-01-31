"""Proposed System D: Consensus–Divergence Elimination (更公平且更刺激).

Only uses weekly judge scores and fan votes.

Definitions (within each season-week):
- Convert judge_percent and fan_vote_normalized into within-week percentiles P^J, P^F in [0,1]
  (robust to scaling and imperfect estimates).
- Consensus score: C = (P^J + P^F)/2  (overall agreement / combined support)
- Divergence score: D = |P^J - P^F|    (controversy / misalignment)

Rule:
1) Risk pool = bottom-m contestants by C (default m=4).
2) Controversy save: protect the contestant with maximum D within the risk pool.
3) Eliminate the remaining contestant in the risk pool with the minimum C.
   (Optional: if multiple eliminations exist, we still record weekly pick.)

Outputs:
- tables/systemD_weekly.csv
- tables/systemD_summary.csv
- figures/fig_systemD_example_week.png
- figures/fig_systemD_divergence_distribution.png
- figures/fig_systemD_geometry_example_week.png
- figures/fig_systemD_divergence_ecdf.png

Run:
D:/python/python.exe D:/codePYTHON/codeMCM/2026/c/proposed_system_outputs/proposed_system_D.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SystemDParams:
    risk_pool_m: int = 4


def _find_file(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError("未找到数据文件。已尝试路径：\n" + "\n".join(str(p) for p in candidates))


def percentile_from_score(values: pd.Series) -> pd.Series:
    """Return percentile in [0,1] where higher score => higher percentile.

    Uses rank with average ties. For n=1, returns 0.5.
    """

    v = pd.to_numeric(values, errors="coerce")
    n = v.notna().sum()
    if n <= 1:
        return pd.Series([0.5] * len(values), index=values.index, dtype=float)

    r = v.rank(method="average", ascending=True)
    # smallest rank=1 => percentile 0; largest rank=n => 1
    return (r - 1.0) / (n - 1.0)


def compute_week_systemD(g: pd.DataFrame, params: SystemDParams) -> Dict[str, object]:
    g = g.copy()

    g["Pj"] = percentile_from_score(g["judge_percent"])  # higher better
    g["Pf"] = percentile_from_score(g["fan_vote_normalized"])  # higher better

    g["C"] = (g["Pj"] + g["Pf"]) / 2.0
    g["D"] = (g["Pj"] - g["Pf"]).abs()

    m = int(min(params.risk_pool_m, len(g)))
    risk = g.sort_values("C", ascending=True).head(m)

    protected_name = ""
    elim_name = ""

    if len(risk) == 0:
        return {
            "elim_systemD": "",
            "protected_systemD": "",
            "risk_pool": "",
            "risk_pool_size": 0,
        }

    # protect the most divergent in risk pool
    idx_protect = risk["D"].astype(float).idxmax()
    protected_name = str(g.loc[idx_protect, "celebrity_name"])

    # eliminate lowest consensus among remaining risk contestants
    remaining = risk.drop(index=idx_protect, errors="ignore")
    if len(remaining) == 0:
        # degenerate: if m==1, no one to eliminate; fall back to global min C
        idx_elim = g["C"].astype(float).idxmin()
        elim_name = str(g.loc[idx_elim, "celebrity_name"])
    else:
        idx_elim = remaining["C"].astype(float).idxmin()
        elim_name = str(g.loc[idx_elim, "celebrity_name"])

    risk_names = ";".join(risk["celebrity_name"].astype(str).tolist())

    return {
        "elim_systemD": elim_name,
        "protected_systemD": protected_name,
        "risk_pool": risk_names,
        "risk_pool_size": int(len(risk)),
    }


def baseline_pct50_pick(g: pd.DataFrame) -> str:
    # score higher better: judge_percent + fan_vote_normalized
    score = pd.to_numeric(g["judge_percent"], errors="coerce") + pd.to_numeric(g["fan_vote_normalized"], errors="coerce")
    idx = score.idxmin()  # mimic elimination by lowest combined share
    return str(g.loc[idx, "celebrity_name"])


def choose_example_week(df: pd.DataFrame) -> Tuple[int, int]:
    """Choose an example week with large average divergence."""

    tmp = (
        df.groupby(["season", "week"], as_index=False)["D"]
        .mean()
        .sort_values("D", ascending=False)
    )
    if tmp.empty:
        return 0, 0
    r = tmp.iloc[0]
    return int(r["season"]), int(r["week"])


def plot_example_week(df: pd.DataFrame, preds: pd.DataFrame, out_path: Path, params: SystemDParams) -> None:
    import matplotlib.pyplot as plt

    season, week = choose_example_week(df)
    g = df[(df["season"] == season) & (df["week"] == week)].copy()
    if g.empty:
        return

    row = preds[(preds["season"] == season) & (preds["week"] == week)].iloc[0]
    risk_set = set(str(row["risk_pool"]).split(";")) if str(row["risk_pool"]).strip() else set()

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    colors = []
    for name in g["celebrity_name"].astype(str):
        if name == row["protected_systemD"]:
            colors.append("#54a24b")  # green
        elif name == row["elim_systemD"]:
            colors.append("#e45756")  # red
        elif name in risk_set:
            colors.append("#f58518")  # orange
        else:
            colors.append("#4c78a8")  # blue

    ax.scatter(g["Pj"], g["Pf"], s=60, c=colors, alpha=0.85, edgecolors="white", linewidth=0.6)

    for _, r in g.iterrows():
        if str(r["celebrity_name"]) in risk_set or str(r["celebrity_name"]) in [row["protected_systemD"], row["elim_systemD"]]:
            ax.annotate(str(r["celebrity_name"]), (r["Pj"], r["Pf"]), fontsize=8, alpha=0.9)

    ax.set_title(
        f"System D example (season={season}, week={week})\n"
        f"risk_pool_m={params.risk_pool_m}; protected=green; eliminated=red"
    )
    ax.set_xlabel("Judges percentile P^J (higher better)")
    ax.set_ylabel("Fans percentile P^F (higher better)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_divergence_distribution(df: pd.DataFrame, preds: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    # weekly predictions do not include celebrity_name; label within each (season, week)
    merged = df.merge(
        preds[["season", "week", "elim_systemD", "protected_systemD", "pred_pct50", "actual_eliminated"]],
        on=["season", "week"],
        how="left",
    )

    merged["is_elim_systemD"] = merged["celebrity_name"].astype(str) == merged["elim_systemD"].astype(str)
    merged["is_protected_systemD"] = merged["celebrity_name"].astype(str) == merged["protected_systemD"].astype(str)
    merged["is_elim_pct50"] = merged["celebrity_name"].astype(str) == merged["pred_pct50"].astype(str)

    d_all = merged["D"].astype(float).dropna().to_numpy()
    d_elim_d = merged.loc[merged["is_elim_systemD"], "D"].astype(float).dropna().to_numpy()
    d_elim_p = merged.loc[merged["is_elim_pct50"], "D"].astype(float).dropna().to_numpy()
    d_prot = merged.loc[merged["is_protected_systemD"], "D"].astype(float).dropna().to_numpy()

    bins = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.hist(d_all, bins=bins, alpha=0.30, label="All contestants", color="#999999")
    ax.hist(d_elim_p, bins=bins, alpha=0.55, label="Eliminated by pct50 baseline", color="#4c78a8")
    ax.hist(d_elim_d, bins=bins, alpha=0.55, label="Eliminated by System D", color="#e45756")
    ax.hist(d_prot, bins=bins, alpha=0.60, label="Protected (controversy save)", color="#54a24b")

    ax.set_title("Divergence distribution: baseline vs System D")
    ax.set_xlabel("Divergence D = |P^J - P^F|")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_geometry_example_week(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    out_path: Path,
    params: SystemDParams,
) -> None:
    """More advanced visualization for one week.

    Shows:
    - risk region boundary induced by bottom-m consensus (C-cutoff line)
    - diagonal (P^J = P^F)
    - divergence band (|P^J-P^F| = D_protected) for the protected contestant
    """

    import matplotlib.pyplot as plt

    season, week = choose_example_week(df)
    g = df[(df["season"] == season) & (df["week"] == week)].copy()
    if g.empty:
        return

    row = preds[(preds["season"] == season) & (preds["week"] == week)].iloc[0]
    risk_set = set(str(row["risk_pool"]).split(";")) if str(row["risk_pool"]).strip() else set()

    # Determine C-cutoff for risk pool (max C among bottom-m)
    m = int(min(params.risk_pool_m, len(g)))
    cutoff_c = float(g["C"].sort_values(ascending=True).head(m).max())
    cutoff_sum = 2.0 * cutoff_c  # Pj + Pf <= cutoff_sum defines risk half-plane

    # Protected contestant divergence
    protected_name = str(row["protected_systemD"])
    d_prot = float(g.loc[g["celebrity_name"].astype(str) == protected_name, "D"].iloc[0]) if protected_name else np.nan

    fig, ax = plt.subplots(figsize=(7.6, 6.4))

    # Shade risk region using a fine grid (robust to cutoff cases)
    xs = np.linspace(0, 1, 240)
    ys = np.linspace(0, 1, 240)
    X, Y = np.meshgrid(xs, ys)
    mask = (X + Y) <= cutoff_sum
    ax.contourf(X, Y, mask.astype(int), levels=[-0.5, 0.5, 1.5], colors=["white", "#f58518"], alpha=0.10)

    # Risk boundary line: Pf = cutoff_sum - Pj
    xline = np.linspace(0, 1, 400)
    yline = cutoff_sum - xline
    ok = (yline >= 0) & (yline <= 1)
    ax.plot(xline[ok], yline[ok], color="#f58518", linewidth=2.0, label="Risk boundary (bottom-m by C)")

    # Diagonal
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.4, linestyle="--", label="Consensus diagonal (P^J = P^F)")

    # Divergence band for protected contestant
    if np.isfinite(d_prot) and d_prot > 0:
        y_up = xline + d_prot
        y_dn = xline - d_prot
        ok1 = (y_up >= 0) & (y_up <= 1)
        ok2 = (y_dn >= 0) & (y_dn <= 1)
        ax.plot(xline[ok1], y_up[ok1], color="#54a24b", linewidth=1.6, alpha=0.9, label=f"|P^J-P^F| = D_protected ({d_prot:.2f})")
        ax.plot(xline[ok2], y_dn[ok2], color="#54a24b", linewidth=1.6, alpha=0.9)

    # Points
    colors = []
    for name in g["celebrity_name"].astype(str):
        if name == row["protected_systemD"]:
            colors.append("#54a24b")
        elif name == row["elim_systemD"]:
            colors.append("#e45756")
        elif name in risk_set:
            colors.append("#f58518")
        else:
            colors.append("#4c78a8")

    ax.scatter(g["Pj"], g["Pf"], s=70, c=colors, alpha=0.90, edgecolors="white", linewidth=0.7)

    # Label key points
    for _, r in g.iterrows():
        nm = str(r["celebrity_name"])
        if nm in risk_set or nm in [str(row["protected_systemD"]), str(row["elim_systemD"]), str(row["actual_eliminated"]), str(row["pred_pct50"])]:
            ax.annotate(nm, (float(r["Pj"]), float(r["Pf"])), fontsize=8, alpha=0.9)

    ax.set_title(
        f"System D decision geometry (season={season}, week={week})\n"
        f"risk_pool_m={params.risk_pool_m}; cutoff C={cutoff_c:.2f}; protected=green; eliminated=red"
    )
    ax.set_xlabel("Judges percentile P^J (higher better)")
    ax.set_ylabel("Fans percentile P^F (higher better)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_divergence_ecdf(df: pd.DataFrame, preds: pd.DataFrame, out_path: Path) -> None:
    """Advanced distribution view: ECDF comparison for divergence D."""

    import matplotlib.pyplot as plt

    merged = df.merge(
        preds[["season", "week", "elim_systemD", "protected_systemD", "pred_pct50", "actual_eliminated"]],
        on=["season", "week"],
        how="left",
    )
    merged["is_elim_systemD"] = merged["celebrity_name"].astype(str) == merged["elim_systemD"].astype(str)
    merged["is_protected_systemD"] = merged["celebrity_name"].astype(str) == merged["protected_systemD"].astype(str)
    merged["is_elim_pct50"] = merged["celebrity_name"].astype(str) == merged["pred_pct50"].astype(str)

    def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.sort(arr)
        y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
        return x, y

    series = {
        "Eliminated by pct50 baseline": merged.loc[merged["is_elim_pct50"], "D"].astype(float).dropna().to_numpy(),
        "Eliminated by System D": merged.loc[merged["is_elim_systemD"], "D"].astype(float).dropna().to_numpy(),
        "Protected (controversy save)": merged.loc[merged["is_protected_systemD"], "D"].astype(float).dropna().to_numpy(),
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    palette = {
        "Eliminated by pct50 baseline": "#4c78a8",
        "Eliminated by System D": "#e45756",
        "Protected (controversy save)": "#54a24b",
    }
    for label, arr in series.items():
        if arr.size == 0:
            continue
        x, y = ecdf(arr)
        ax.step(x, y, where="post", linewidth=2.2, label=f"{label} (n={arr.size})", color=palette.get(label, "#333333"))

    ax.set_title("ECDF of divergence D: baseline vs System D")
    ax.set_xlabel("Divergence D = |P^J - P^F|")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    # remove weeks without actual elimination
    df = df[df["actual_eliminated"].astype(str).str.len() > 0].copy()

    def hit(col: str) -> float:
        return float(np.mean(df[col].astype(str) == df["actual_eliminated"].astype(str)))

    out = [
        {"metric": "weeks_with_elimination", "value": float(len(df))},
        {"metric": "hit_rate_pct50", "value": hit("pred_pct50")},
        {"metric": "hit_rate_systemD", "value": hit("elim_systemD")},
        {
            "metric": "disagree_rate_systemD_vs_pct50",
            "value": float(np.mean(df["elim_systemD"].astype(str) != df["pred_pct50"].astype(str))),
        },
        {
            "metric": "protected_equals_pct50_elim_rate",
            "value": float(np.mean(df["protected_systemD"].astype(str) == df["pred_pct50"].astype(str))),
        },
    ]

    return pd.DataFrame(out)


def main() -> None:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]  # .../2026/c

    fan_path = _find_file(
        [
            project_root / "fan_vote_final_fixed.csv",
            project_root.parents[2] / "fan_vote_final_fixed.csv",
            Path("D:/codePYTHON/fan_vote_final_fixed.csv"),
            Path("d:/codePYTHON/fan_vote_final_fixed.csv"),
        ]
    )

    out_dir = here
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_csv(fan_path)

    required = ["season", "week", "celebrity_name", "judge_percent", "fan_vote_normalized"]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise ValueError(f"fan_vote_final_fixed.csv 缺少必要列: {missing}")

    df0["season"] = pd.to_numeric(df0["season"], errors="coerce")
    df0["week"] = pd.to_numeric(df0["week"], errors="coerce")

    # ensure actual elimination column
    if "is_eliminated" not in df0.columns:
        # derive from results if available
        if "results" in df0.columns:
            df0["is_eliminated"] = df0["results"].astype(str).str.contains("ELIM", case=False, na=False).astype(int)
        else:
            df0["is_eliminated"] = 0

    params = SystemDParams(risk_pool_m=4)

    # compute Pj/Pf/C/D per row
    rows = []
    for (season, week), g in df0.groupby(["season", "week"], sort=False):
        gg = g.copy()
        gg["Pj"] = percentile_from_score(gg["judge_percent"])
        gg["Pf"] = percentile_from_score(gg["fan_vote_normalized"])
        gg["C"] = (gg["Pj"] + gg["Pf"]) / 2.0
        gg["D"] = (gg["Pj"] - gg["Pf"]).abs()
        rows.append(gg)

    df = pd.concat(rows, ignore_index=True)

    # week-level predictions
    preds = []
    for (season, week), g in df.groupby(["season", "week"], sort=False):
        actual = g.loc[pd.to_numeric(g["is_eliminated"], errors="coerce") == 1, "celebrity_name"].astype(str).tolist()
        actual_one = actual[0] if len(actual) >= 1 else ""

        r = compute_week_systemD(g, params)
        preds.append(
            {
                "season": int(season),
                "week": int(week),
                "n": int(len(g)),
                "actual_eliminated": actual_one,
                "pred_pct50": baseline_pct50_pick(g),
                **r,
            }
        )

    preds_df = pd.DataFrame(preds)

    # save tables
    df_out = df[["season", "week", "celebrity_name", "judge_percent", "fan_vote_normalized", "Pj", "Pf", "C", "D", "is_eliminated"]].copy()
    df_out.to_csv(tab_dir / "systemD_row_scores.csv", index=False, encoding="utf-8-sig")

    preds_df.to_csv(tab_dir / "systemD_weekly.csv", index=False, encoding="utf-8-sig")

    summ = summarize(preds_df)
    summ.to_csv(tab_dir / "systemD_summary.csv", index=False, encoding="utf-8-sig")

    # figures
    plot_example_week(df, preds_df, fig_dir / "fig_systemD_example_week.png", params=params)
    plot_divergence_distribution(df, preds_df, fig_dir / "fig_systemD_divergence_distribution.png")
    plot_geometry_example_week(df, preds_df, fig_dir / "fig_systemD_geometry_example_week.png", params=params)
    plot_divergence_ecdf(df, preds_df, fig_dir / "fig_systemD_divergence_ecdf.png")

    print("[OK] wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
