"""第二问（Part 2）争议案例分析：Rank vs Percent + Judges' Save

题目要求：
- 针对“评委与粉丝意见冲突（controversy）”的典型选手，比较：
  1) 使用 Rank 合成 vs Percent 合成，是否会得到相同结果（淘汰周/最终名次）？
  2) 若增加 Judges' Save（先确定 bottom-2，再由评委决定淘汰谁）会怎样影响？

本脚本优先分析题目给出的 4 个案例：
- Season 2  Jerry Rice
- Season 4  Billy Ray Cyrus
- Season 11 Bristol Palin
- Season 27 Bobby Bones

实现口径（透明可写入论文）：
- Percent：C_i = judge_percent + fan_share（越小越危险）
- Rank：C_i = rank(judge) + rank(fan)（越大越危险）
- Judges' Save（模拟）：
  - 先按某规则得到 bottom-2（最危险 2 人）
  - 在 bottom-2 内淘汰“评委分更低者”（以 total_judge_score 更低为准）
  - 若当周有多淘汰（k>1），则重复 k 次：每次重新计算 bottom-2 再淘汰一人

输出：
- controversy_outputs/case_summary.csv：每个案例的实际 vs 反事实结果汇总
- controversy_outputs/case_weekly_<season>_<name>.csv：该选手赛季逐周轨迹（judge/fan/rank/是否在bottom2等）
- controversy_outputs/figures/：每个案例的周序列图
- controversy_outputs/controversy_cases_writeup.md：论文可用说明（自动生成）

运行：
D:/python/python.exe codeMCM/2026/c/Controversy_cases.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
            return int(row_week) == int(m.group(1))
        except Exception:
            return False

    return pd.Series([one(r, w) for r, w in zip(results.tolist(), week.tolist())], index=results.index, dtype=bool)


def _parse_actual_elim_week(results_text: object) -> Optional[int]:
    if pd.isna(results_text):
        return None
    text = str(results_text)
    m = re.search(r"Eliminated\s*Week\s*(\d+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def prepare_merged_data(cleaned_path: Path, fan_path: Path) -> pd.DataFrame:
    cleaned = pd.read_csv(cleaned_path)
    fan = pd.read_csv(fan_path)

    for frame in (cleaned, fan):
        if "celebrity_name" in frame.columns:
            frame["celebrity_name"] = frame["celebrity_name"].astype(str).str.strip()
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
        raise ValueError("cleaned 缺少 results/is_eliminated")

    fan = fan.copy()
    if "fan_vote_normalized" not in fan.columns:
        raise ValueError("fan_vote_final_fixed.csv 缺少 fan_vote_normalized")

    fan_small = fan[["season", "week", "celebrity_name", "fan_vote_normalized"]].drop_duplicates(
        subset=["season", "week", "celebrity_name"], keep="last"
    )

    merged = cleaned.merge(fan_small, on=["season", "week", "celebrity_name"], how="left", validate="m:1")
    merged["fan_vote_normalized"] = pd.to_numeric(merged["fan_vote_normalized"], errors="coerce").clip(lower=0)

    # 只保留当周仍在比赛的记录（cleaned 已处理过 0 分；这里再保险）
    merged = merged[pd.to_numeric(merged["total_judge_score"], errors="coerce") > 0].copy()

    return merged


def _stable_sort(df: pd.DataFrame, by: List[str], ascending: List[bool]) -> pd.DataFrame:
    return df.sort_values(by=by, ascending=ascending, kind="mergesort")


def bottom_k_percent(g: pd.DataFrame, k: int) -> pd.DataFrame:
    gg = g.copy()
    gg["comb_pct"] = gg["judge_percent"] + gg["fan_vote_normalized"]
    gg = _stable_sort(gg, by=["comb_pct", "judge_percent", "fan_vote_normalized", "celebrity_name"], ascending=[True, True, True, True])
    return gg.head(k)


def bottom_k_rank(g: pd.DataFrame, k: int) -> pd.DataFrame:
    gg = g.copy()
    gg["j_rank"] = gg["total_judge_score"].rank(ascending=False, method="min")
    gg["f_rank"] = gg["fan_vote_normalized"].rank(ascending=False, method="min")
    gg["comb_rank"] = gg["j_rank"] + gg["f_rank"]
    gg = _stable_sort(gg, by=["comb_rank", "f_rank", "j_rank", "celebrity_name"], ascending=[False, False, False, True])
    return gg.head(k)


def eliminate_under_method(g: pd.DataFrame, k: int, method: str) -> List[str]:
    if method == "percent":
        return bottom_k_percent(g, k)["celebrity_name"].astype(str).tolist()
    if method == "rank":
        return bottom_k_rank(g, k)["celebrity_name"].astype(str).tolist()
    raise ValueError(f"unknown method: {method}")


def eliminate_with_judges_save(g: pd.DataFrame, k: int, base_method: str) -> List[str]:
    """模拟 Judges' Save：先按 base_method 找 bottom-2，再淘汰评委分更低者。"""

    remaining = g.copy()
    eliminated: List[str] = []

    for _ in range(int(k)):
        if remaining.empty:
            break
        if len(remaining) == 1:
            eliminated.append(str(remaining["celebrity_name"].iloc[0]))
            break

        bottom2 = eliminate_under_method(remaining, min(2, len(remaining)), base_method)
        cand = remaining[remaining["celebrity_name"].astype(str).isin(bottom2)].copy()

        # 评委选择：淘汰评委分更低者
        cand = _stable_sort(cand, by=["total_judge_score", "fan_vote_normalized", "celebrity_name"], ascending=[True, True, True])
        out_name = str(cand["celebrity_name"].iloc[0])
        eliminated.append(out_name)

        remaining = remaining[remaining["celebrity_name"].astype(str) != out_name].copy()

    return eliminated


@dataclass
class ScenarioResult:
    scenario: str
    elim_week: Optional[int]
    final_place: Optional[int]


def simulate_season(merged: pd.DataFrame, season: int, scenario: str) -> Tuple[Dict[str, ScenarioResult], pd.DataFrame]:
    """按真实每周淘汰人数 k，模拟整季淘汰路径。

    返回：
    - 每位选手的（淘汰周/最终名次）
    - elimination_log：逐周淘汰名单（便于追溯）
    """

    season_df = merged[merged["season"] == season].copy()
    if season_df.empty:
        raise ValueError(f"season {season} 数据为空")

    # 每周真实淘汰人数（从 is_eliminated 解析）
    k_by_week = (
        season_df.groupby("week")["is_eliminated"].apply(lambda x: int(x.fillna(False).astype(bool).sum())).to_dict()
    )

    # 初始选手集合：以该季出现过的全部选手为准
    contestants = sorted(season_df["celebrity_name"].astype(str).unique().tolist())

    # 逐周更新：用该周仍在比赛的人（在 cleaned 里会有该周记录）
    elim_week: Dict[str, Optional[int]] = {c: None for c in contestants}

    elim_rows: List[Dict[str, object]] = []

    weeks = sorted(int(w) for w in season_df["week"].dropna().unique().tolist())
    alive: List[str] = contestants.copy()

    for w in weeks:
        k = int(k_by_week.get(w, 0))
        if k <= 0:
            continue

        # 当周可用数据：只对 alive 的选手
        g = season_df[(season_df["week"] == w) & (season_df["celebrity_name"].astype(str).isin(alive))].copy()
        g = g.dropna(subset=["fan_vote_normalized", "judge_percent", "total_judge_score"])
        if g.empty:
            continue

        if scenario == "percent":
            out = eliminate_under_method(g, min(k, len(g)), "percent")
        elif scenario == "rank":
            out = eliminate_under_method(g, min(k, len(g)), "rank")
        elif scenario == "percent_judges_save":
            out = eliminate_with_judges_save(g, min(k, len(g)), base_method="percent")
        elif scenario == "rank_judges_save":
            out = eliminate_with_judges_save(g, min(k, len(g)), base_method="rank")
        else:
            raise ValueError(f"unknown scenario: {scenario}")

        # 记录淘汰
        for name in out:
            if name in alive:
                alive.remove(name)
            if elim_week.get(name) is None:
                elim_week[name] = int(w)

        elim_rows.append({"season": season, "week": int(w), "k": int(k), "scenario": scenario, "eliminated": "; ".join(out)})

    # 最终名次：用模拟淘汰周排序（淘汰越晚名次越好），未被淘汰者按最后一周合成分排序
    # 先构造：淘汰周越大越好；未淘汰记为 +inf
    last_week = max(weeks)
    final_week_df = season_df[(season_df["week"] == last_week) & (season_df["celebrity_name"].astype(str).isin(alive))].copy()

    # 对决赛未淘汰者的最终排序
    if not final_week_df.empty:
        if scenario.startswith("percent"):
            final_week_df["final_score"] = final_week_df["judge_percent"] + final_week_df["fan_vote_normalized"]
            # 越大越好（安全）
            final_week_df = _stable_sort(final_week_df, by=["final_score", "fan_vote_normalized", "total_judge_score", "celebrity_name"], ascending=[False, False, False, True])
        else:
            final_week_df["j_rank"] = final_week_df["total_judge_score"].rank(ascending=False, method="min")
            final_week_df["f_rank"] = final_week_df["fan_vote_normalized"].rank(ascending=False, method="min")
            final_week_df["final_score"] = -(final_week_df["j_rank"] + final_week_df["f_rank"])  # 越大越好
            final_week_df = _stable_sort(final_week_df, by=["final_score", "fan_vote_normalized", "total_judge_score", "celebrity_name"], ascending=[False, False, False, True])

        alive_order = final_week_df["celebrity_name"].astype(str).tolist()
    else:
        alive_order = alive

    # 名次：先把未淘汰者（alive_order）放最前，再按淘汰周从大到小
    eliminated_names = [c for c in contestants if elim_week.get(c) is not None]
    eliminated_sorted = sorted(eliminated_names, key=lambda x: int(elim_week[x] or 0), reverse=True)

    overall_order = alive_order + eliminated_sorted

    # 转成名次（1最好）
    place: Dict[str, int] = {name: i + 1 for i, name in enumerate(overall_order)}

    result: Dict[str, ScenarioResult] = {
        name: ScenarioResult(scenario=scenario, elim_week=elim_week.get(name), final_place=place.get(name)) for name in contestants
    }

    elim_log = pd.DataFrame(elim_rows).sort_values(["week"]).reset_index(drop=True)
    return result, elim_log


def _rank_pct_ascending(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, ascending=True, method="average")


def build_case_weekly_table(merged: pd.DataFrame, season: int, celeb: str) -> pd.DataFrame:
    df = merged[merged["season"] == season].copy()
    if df.empty:
        raise ValueError(f"season {season} 无数据")

    # 每周内部计算 rank
    rows: List[pd.DataFrame] = []
    for (s, w), g in df.groupby(["season", "week"], sort=True):
        g = g.copy()
        g["judge_rank_worst1"] = g["total_judge_score"].rank(ascending=True, method="min")  # 1=最差
        g["fan_rank_worst1"] = g["fan_vote_normalized"].rank(ascending=True, method="min")
        g["judge_pct_rank"] = _rank_pct_ascending(g["total_judge_score"])
        g["fan_pct_rank"] = _rank_pct_ascending(g["fan_vote_normalized"])

        # 危险集合
        b2_pct = set(bottom_k_percent(g, min(2, len(g)))["celebrity_name"].astype(str).tolist())
        b2_rank = set(bottom_k_rank(g, min(2, len(g)))["celebrity_name"].astype(str).tolist())
        g["in_bottom2_percent"] = g["celebrity_name"].astype(str).isin(b2_pct)
        g["in_bottom2_rank"] = g["celebrity_name"].astype(str).isin(b2_rank)

        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out = out[out["celebrity_name"].astype(str) == celeb].copy()
    out = out.sort_values(["week"]).reset_index(drop=True)

    # 额外统计：评委与粉丝分歧（本选手）
    out["fan_minus_judge_pct"] = out["fan_pct_rank"] - out["judge_pct_rank"]
    return out


def _match_name(season_df: pd.DataFrame, target: str) -> str:
    names = season_df["celebrity_name"].astype(str).unique().tolist()
    if target in names:
        return target
    low = target.strip().lower()
    # contains
    cand = [n for n in names if low in n.lower()]
    if len(cand) == 1:
        return cand[0]
    # fallback: closest by simple heuristic (same last name)
    last = low.split()[-1]
    cand2 = [n for n in names if n.lower().endswith(last)]
    if len(cand2) >= 1:
        return sorted(cand2, key=len)[0]
    # last resort
    return names[0]


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

    merged = prepare_merged_data(cleaned_path, fan_path)

    cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    ]

    out_root = here / "controversy_outputs"
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 画图尽量用 seaborn，否则 matplotlib
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

    summary_rows: List[Dict[str, object]] = []

    writeup_sections: List[str] = []

    scenarios = ["percent", "rank", "percent_judges_save", "rank_judges_save"]

    for season, target_name in cases:
        season_df = merged[merged["season"] == season].copy()
        if season_df.empty:
            print(f"[WARN] Season {season} not found")
            continue

        celeb = _match_name(season_df, target_name)

        # 基本真实信息
        one_row = season_df[season_df["celebrity_name"].astype(str) == celeb].head(1)
        actual_place = None
        actual_elim_week = None
        if not one_row.empty:
            if "placement" in one_row.columns:
                try:
                    actual_place = int(pd.to_numeric(one_row["placement"].iloc[0], errors="coerce"))
                except Exception:
                    actual_place = None
            if "results" in one_row.columns:
                actual_elim_week = _parse_actual_elim_week(one_row["results"].iloc[0])

        # 逐周轨迹
        weekly_table = build_case_weekly_table(merged, season, celeb)
        weekly_path = out_root / f"case_weekly_S{season}_{celeb.replace(' ', '_')}.csv"
        weekly_table.to_csv(weekly_path, index=False, encoding="utf-8-sig")

        # 统计“争议”特征：评委最差次数、粉丝较好次数
        judge_worst_weeks = int((weekly_table["judge_rank_worst1"] == 1).sum()) if "judge_rank_worst1" in weekly_table.columns else np.nan
        fan_top_half_weeks = int((weekly_table["fan_pct_rank"] > 0.5).sum()) if "fan_pct_rank" in weekly_table.columns else np.nan

        # 赛季模拟
        sim_results: Dict[str, ScenarioResult] = {}
        for sc in scenarios:
            res, _log = simulate_season(merged, season, sc)
            sim_results[sc] = res.get(celeb)  # type: ignore

        row: Dict[str, object] = {
            "season": int(season),
            "celebrity_name": celeb,
            "actual_placement": actual_place,
            "actual_elim_week": actual_elim_week,
            "judge_worst_weeks": judge_worst_weeks,
            "fan_top_half_weeks": fan_top_half_weeks,
            "weekly_table": str(weekly_path),
        }
        for sc in scenarios:
            r = sim_results.get(sc)
            row[f"{sc}_elim_week"] = None if r is None else r.elim_week
            row[f"{sc}_final_place"] = None if r is None else r.final_place

        summary_rows.append(row)

        # 绘图：judge_rank_worst1 与 fan_rank_worst1 随周变化（1=最差）
        fig, ax = plt.subplots(figsize=(10.8, 4.8))
        ax.plot(weekly_table["week"], weekly_table["judge_rank_worst1"], marker="o", linewidth=2, label="Judge rank (1=worst)", color="#d95f02")
        ax.plot(weekly_table["week"], weekly_table["fan_rank_worst1"], marker="o", linewidth=2, label="Fan rank (1=worst)", color="#1b9e77")
        ax.invert_yaxis()  # 让“更差(1)”显示在上方，更直观
        ax.set_title(f"S{season} {celeb}: weekly judge vs fan rank (1=worst)")
        ax.set_xlabel("Week")
        ax.set_ylabel("Rank (1=worst)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig_path = fig_dir / f"fig_case_S{season}_{celeb.replace(' ', '_')}_ranks.png"
        fig.savefig(fig_path)
        plt.close(fig)

        # 写作段落（先占位，最终在 md 里展开）
        writeup_sections.append(
            "\n".join(
                [
                    f"### S{season} – {celeb}",
                    f"- 真实结果：placement={actual_place}, elim_week={actual_elim_week}",
                    f"- ‘争议’迹象（基于本模型投票份额）：评委当周最差次数={judge_worst_weeks}；粉丝份额处于上半区次数={fan_top_half_weeks}",
                    f"- 反事实（Percent / Rank / Percent+Save / Rank+Save）见汇总表。",
                    f"- 周度轨迹表：`{weekly_path.name}`；周序列图：`{fig_path.name}`",
                ]
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_root / "case_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 生成 Markdown 写作稿
    md_lines: List[str] = []
    md_lines.append("# 第二问（Part 2）争议选手反事实分析：Rank vs Percent + Judges’ Save\n")
    md_lines.append("本节聚焦题目给出的争议案例，回答两个问题：\n")
    md_lines.append("1) 若用 Rank 或 Percent 合成评委分与观众投票，争议选手的淘汰周/最终名次是否改变？\n")
    md_lines.append("2) 若加入 Judges’ Save（bottom-2 后由评委选择淘汰谁），结果会怎样变化？\n")

    md_lines.append("## 方法与公式（可直接放论文）\n")
    md_lines.append("设当周仍在比赛的选手为 $i=1,\\dots,n$。评委总分为 $J_i$，观众投票份额估计为 $F_i$（周内归一化，$\\sum_i F_i=1$）。\n")
    md_lines.append("- **Percent 合成（越小越危险）**：$C^{pct}_i = J_i/\\sum_j J_j + F_i$，淘汰 $C^{pct}$ 最小的 $k$ 人。\n")
    md_lines.append("- **Rank 合成（越大越危险）**：$C^{rank}_i = R^J_i + R^F_i$（两者均为从高到低的名次），淘汰 $C^{rank}$ 最大的 $k$ 人。\n")
    md_lines.append("- **Judges’ Save（模拟）**：先用某合成规则确定 bottom-2，然后在 bottom-2 内淘汰评委分更低者（以 $J_i$ 更小为准）；若当周多淘汰（$k>1$），则重复该过程 $k$ 次。\n")

    md_lines.append("## 输出文件\n")
    md_lines.append(f"- 汇总表：`{summary_path.name}`（实际 vs 反事实淘汰周/名次）\n")
    md_lines.append("- 每个案例周度表：`case_weekly_S<season>_<name>.csv`\n")
    md_lines.append("- 每个案例图：`fig_case_S<season>_<name>_ranks.png`\n")

    md_lines.append("## 案例速览（待写作扩展）\n")
    md_lines.extend(writeup_sections)

    md_lines.append("\n## 写作提示：如何在论文里解读 Judges’ Save\n")
    md_lines.append("Judges’ Save 的核心作用是：当观众投票把某位‘技术较差’的选手推离淘汰边缘时，评委在 bottom-2 内可纠正这一结果；因此它通常会降低‘粉丝主导的反直觉淘汰’发生概率，但也会削弱观众投票的决定性。\n")

    md_path = out_root / "controversy_cases_writeup.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OUTPUT] summary: {summary_path}")
    print(f"[OUTPUT] writeup: {md_path}")
    print(f"[OUTPUT] figures: {fig_dir}")


if __name__ == "__main__":
    main()
