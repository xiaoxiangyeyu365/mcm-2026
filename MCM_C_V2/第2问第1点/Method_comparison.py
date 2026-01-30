"""投票合成规则对比：Rank vs Percent（第二问-第一部分）

目标（对应题目要求）：
- 使用你已估计的观众投票份额（fan_vote_normalized）
- 在每个赛季的每一周同时应用两种合成方式：
  1) Percent（评委百分比 + 粉丝百分比，越小越危险）
  2) Rank（评委排名 + 粉丝排名，和越大越危险）
- 输出：
  - weekly_method_comparison.csv：逐周对比（两种方法预测淘汰、是否一致、是否命中真实淘汰等）
  - season_method_comparison.csv：逐赛季汇总（两方法准确率、分歧率、偏向粉丝的指标等）
  - figures/：论文用图（准确率对比、分歧率、偏向粉丝指标等）

说明：
- 本脚本只比较“合成规则（rank vs percent）”本身。
- Judges' Save（bottom-2 后评委选择）属于第二问的下一小问，会在后续脚本/扩展中单独分析。

运行：
D:/python/python.exe codeMCM/2026/c/Method_comparison.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _find_file(candidates: Sequence[Path]) -> Path:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError("未找到数据文件。已尝试路径：\n" + "\n".join(str(p) for p in candidates))


def _parse_is_eliminated(results: pd.Series, week: pd.Series) -> pd.Series:
    def one(row_results: object, row_week: object) -> bool:
        text = "" if pd.isna(row_results) else str(row_results)
        m = re.search(r"Eliminated\s*Week\s*(\d+)", text, flags=re.IGNORECASE)
        if not m:
            return False
        try:
            elim_week = int(m.group(1))
            return int(row_week) == elim_week
        except Exception:
            return False

    return pd.Series(
        [one(r, w) for r, w in zip(results.tolist(), week.tolist())],
        index=results.index,
        dtype=bool,
    )


def prepare_merged_data(cleaned_path: Path, fan_path: Path) -> pd.DataFrame:
    cleaned = pd.read_csv(cleaned_path)
    fan = pd.read_csv(fan_path)

    # 统一键
    for frame in (cleaned, fan):
        for col in ("celebrity_name", "ballroom_partner"):
            if col in frame.columns:
                frame[col] = frame[col].astype(str).str.strip()
        for col in ("season", "week"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "total_judge_score" not in cleaned.columns:
        raise ValueError("cleaned 数据缺少 total_judge_score")

    cleaned = cleaned.copy()
    cleaned["week_total_score"] = cleaned.groupby(["season", "week"])["total_judge_score"].transform("sum")
    cleaned["judge_percent"] = cleaned["total_judge_score"] / cleaned["week_total_score"].replace(0, np.nan)

    if "results" in cleaned.columns:
        cleaned["is_eliminated"] = _parse_is_eliminated(cleaned["results"], cleaned["week"])
    elif "is_eliminated" not in cleaned.columns:
        raise ValueError("cleaned 数据缺少 results/is_eliminated")

    fan = fan.copy()
    if "fan_vote_normalized" not in fan.columns:
        if "fan_vote_est_mean" not in fan.columns:
            raise ValueError("fan_vote_final_fixed.csv 缺少 fan_vote_normalized 且缺少 fan_vote_est_mean")
        fan["fan_vote_normalized"] = fan.groupby(["season", "week"])["fan_vote_est_mean"].transform(
            lambda x: x / x.sum()
        )

    keep_cols = [
        c
        for c in [
            "season",
            "week",
            "celebrity_name",
            "fan_vote_normalized",
        ]
        if c in fan.columns
    ]
    fan_small = fan[keep_cols].drop_duplicates(subset=["season", "week", "celebrity_name"], keep="last")

    merged = cleaned.merge(
        fan_small,
        on=["season", "week", "celebrity_name"],
        how="left",
        validate="m:1",
    )

    merged["fan_vote_normalized"] = pd.to_numeric(merged["fan_vote_normalized"], errors="coerce").clip(lower=0)
    return merged


def _sorted_elims(names: List[str]) -> str:
    return "; ".join(names)


def _predict_elims_percent(g: pd.DataFrame, n_elim: int) -> List[str]:
    gg = g.copy()
    gg["comb"] = gg["judge_percent"] + gg["fan_vote_normalized"]
    gg = gg.sort_values(
        ["comb", "judge_percent", "fan_vote_normalized", "celebrity_name"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    return gg["celebrity_name"].head(n_elim).astype(str).tolist()


def _predict_elims_rank(g: pd.DataFrame, n_elim: int) -> List[str]:
    gg = g.copy()
    gg["j_rank"] = gg["total_judge_score"].rank(ascending=False, method="min")
    gg["f_rank"] = gg["fan_vote_normalized"].rank(ascending=False, method="min")
    gg["comb_rank"] = gg["j_rank"] + gg["f_rank"]

    gg = gg.sort_values(
        ["comb_rank", "f_rank", "j_rank", "celebrity_name"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return gg["celebrity_name"].head(n_elim).astype(str).tolist()


def _rank_pct_ascending(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, ascending=True, method="average")


def compute_weekly_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []

    for (s, w), g in merged.groupby(["season", "week"], sort=True):
        g = g.copy()
        if g.empty:
            continue

        # 真实淘汰
        y_true = g["is_eliminated"].fillna(False).astype(bool)
        n_elim = int(y_true.sum())
        if n_elim == 0:
            continue

        actual = g.loc[y_true, "celebrity_name"].astype(str).tolist()

        # 两种方法预测淘汰
        pred_percent = _predict_elims_percent(g, n_elim)
        pred_rank = _predict_elims_rank(g, n_elim)

        actual_set = set(actual)
        percent_set = set(pred_percent)
        rank_set = set(pred_rank)

        percent_hit = int(actual_set.issubset(percent_set))
        rank_hit = int(actual_set.issubset(rank_set))
        methods_agree = int(percent_set == rank_set)

        # 量化“更偏向粉丝”
        # 用淘汰者的 fan 排名分位与 judge 排名分位对比：
        # delta = fan_pct - judge_pct（越负表示淘汰者相对更“粉丝不喜欢”而不是“评委不喜欢”）
        fan_pct = _rank_pct_ascending(g["fan_vote_normalized"])  # 低票=低分位
        judge_pct = _rank_pct_ascending(g["total_judge_score"])  # 低分=低分位

        def stats_for(pred_list: List[str]) -> Tuple[float, float, float, float]:
            mask = g["celebrity_name"].astype(str).isin(pred_list)
            if mask.sum() == 0:
                return np.nan, np.nan, np.nan, np.nan
            elim_fan = float(fan_pct.loc[mask].mean())
            elim_judge = float(judge_pct.loc[mask].mean())
            elim_delta = float((fan_pct.loc[mask] - judge_pct.loc[mask]).mean())
            # 被淘汰者是否“粉丝高票”（例如 fan_pct>0.5），比例越低表示越偏向粉丝
            elim_high_fan_rate = float((fan_pct.loc[mask] > 0.5).mean())
            return elim_fan, elim_judge, elim_delta, elim_high_fan_rate

        p_fan, p_judge, p_delta, p_highfan = stats_for(pred_percent)
        r_fan, r_judge, r_delta, r_highfan = stats_for(pred_rank)

        rows.append(
            {
                "season": int(s),
                "week": int(w),
                "n_competitors": int(g["celebrity_name"].nunique()),
                "n_eliminated": int(n_elim),
                "actual_elims": _sorted_elims(actual),
                "pred_percent_elims": _sorted_elims(pred_percent),
                "pred_rank_elims": _sorted_elims(pred_rank),
                "percent_hit": percent_hit,
                "rank_hit": rank_hit,
                "methods_agree": methods_agree,
                "percent_elim_fan_pct_mean": p_fan,
                "percent_elim_judge_pct_mean": p_judge,
                "percent_elim_delta_mean": p_delta,
                "percent_elim_high_fan_rate": p_highfan,
                "rank_elim_fan_pct_mean": r_fan,
                "rank_elim_judge_pct_mean": r_judge,
                "rank_elim_delta_mean": r_delta,
                "rank_elim_high_fan_rate": r_highfan,
            }
        )

    out = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("未生成任何逐周对比结果：请检查 is_eliminated 是否解析成功")
    return out


def compute_season_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    agg = (
        weekly.groupby("season", as_index=False)
        .agg(
            n_weeks=("week", "size"),
            percent_accuracy=("percent_hit", "mean"),
            rank_accuracy=("rank_hit", "mean"),
            method_agreement_rate=("methods_agree", "mean"),
            # 这两个值用来回答“是否更偏向粉丝”：
            # delta 越负，说明淘汰更靠“粉丝差”驱动；high_fan_rate 越低，说明更少淘汰高票选手。
            percent_delta_mean=("percent_elim_delta_mean", "mean"),
            rank_delta_mean=("rank_elim_delta_mean", "mean"),
            percent_high_fan_elim_rate=("percent_elim_high_fan_rate", "mean"),
            rank_high_fan_elim_rate=("rank_elim_high_fan_rate", "mean"),
        )
        .sort_values("season")
        .reset_index(drop=True)
    )

    # 差值列：更方便写正文
    agg["accuracy_gap_rank_minus_percent"] = agg["rank_accuracy"] - agg["percent_accuracy"]
    agg["delta_gap_rank_minus_percent"] = agg["rank_delta_mean"] - agg["percent_delta_mean"]
    agg["high_fan_elim_gap_rank_minus_percent"] = agg["rank_high_fan_elim_rate"] - agg["percent_high_fan_elim_rate"]
    return agg


def make_plots(season: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="whitegrid")
        use_sns = True
    except Exception:
        use_sns = False

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

    # 图1：两种方法的赛季准确率对比
    fig1, ax1 = plt.subplots(figsize=(11.5, 4.8))
    ax1.plot(season["season"], season["percent_accuracy"], marker="o", linewidth=2, label="Percent method", color="#1b9e77")
    ax1.plot(season["season"], season["rank_accuracy"], marker="o", linewidth=2, label="Rank method", color="#7570b3")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Elimination Consistency (Apply Both Methods to Every Season)")
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Weekly hit rate")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", frameon=True)
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig_accuracy_rank_vs_percent_by_season.png")

    # 图2：两种方法预测淘汰是否一致（分歧率 = 1 - agreement）
    fig2, ax2 = plt.subplots(figsize=(11.5, 4.8))
    disagree = 1.0 - season["method_agreement_rate"]
    ax2.bar(season["season"], disagree, color="#2b8cbe", alpha=0.85)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("How Often Do Rank and Percent Disagree?")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Disagreement rate")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_disagreement_rate_by_season.png")

    # 图3：偏向粉丝指标（淘汰者 fan vs judge 分位差）
    fig3, ax3 = plt.subplots(figsize=(11.5, 4.8))
    ax3.plot(season["season"], season["percent_delta_mean"], marker="o", linewidth=2, label="Percent: mean(fan_pct - judge_pct)", color="#d95f02")
    ax3.plot(season["season"], season["rank_delta_mean"], marker="o", linewidth=2, label="Rank: mean(fan_pct - judge_pct)", color="#1b9e77")
    ax3.axhline(0, color="#666666", linewidth=1, alpha=0.6)
    ax3.set_title("Fan-favor proxy: Eliminated (fan percentile − judge percentile)")
    ax3.set_xlabel("Season")
    ax3.set_ylabel("Mean delta (lower = more fan-driven eliminations)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="lower right", frameon=True)
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_fan_favor_proxy_delta_by_season.png")


def main() -> None:
    here = Path(__file__).resolve().parent
    workspace_root = here.parents[2]

    cleaned_path = _find_file(
        [
            here / "cleaned_dwts_data_V2.csv",
            workspace_root / "cleaned_dwts_data_V2.csv",
            Path("D:/codePYTHON/cleaned_dwts_data_V2.csv"),
            Path("d:/codePYTHON/cleaned_dwts_data_V2.csv"),
        ]
    )
    fan_path = _find_file(
        [
            here / "fan_vote_final_fixed.csv",
            workspace_root / "fan_vote_final_fixed.csv",
            Path("D:/codePYTHON/fan_vote_final_fixed.csv"),
            Path("d:/codePYTHON/fan_vote_final_fixed.csv"),
        ]
    )

    print(f"[INFO] cleaned: {cleaned_path}")
    print(f"[INFO] fan:     {fan_path}")

    merged = prepare_merged_data(cleaned_path=cleaned_path, fan_path=fan_path)
    weekly = compute_weekly_comparison(merged)
    season = compute_season_summary(weekly)

    out_root = here / "method_compare_outputs"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    weekly_path = out_root / "weekly_method_comparison.csv"
    season_path = out_root / "season_method_comparison.csv"
    weekly.to_csv(weekly_path, index=False, encoding="utf-8-sig")
    season.to_csv(season_path, index=False, encoding="utf-8-sig")

    make_plots(season, fig_dir)

    # 全局摘要（方便写正文）
    overall = {
        "n_weeks": int(weekly.shape[0]),
        "percent_accuracy": float(weekly["percent_hit"].mean()),
        "rank_accuracy": float(weekly["rank_hit"].mean()),
        "agreement_rate": float(weekly["methods_agree"].mean()),
    }
    print("\n[SUMMARY] Weeks: {n_weeks}".format(**overall))
    print("[SUMMARY] Percent accuracy: {:.2%}".format(overall["percent_accuracy"]))
    print("[SUMMARY] Rank accuracy:    {:.2%}".format(overall["rank_accuracy"]))
    print("[SUMMARY] Agreement rate:   {:.2%}".format(overall["agreement_rate"]))

    print(f"\n[OUTPUT] weekly:  {weekly_path}")
    print(f"[OUTPUT] season:  {season_path}")
    print(f"[OUTPUT] figures: {fig_dir}")


if __name__ == "__main__":
    main()
