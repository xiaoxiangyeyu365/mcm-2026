"""一致性度量（Consistency Metrics）与论文绘图

用途：
- 读取你最终模型输出的 fan_vote_final_fixed.csv（包含每周归一化后的观众投票份额预测）
- 读取 cleaned_dwts_data_V2.csv（包含真实比赛的每周评委分数与淘汰结果）
- 按 DWTS 规则（S1-2 排名法、S3-27 百分比法、S28+ Judges' Save）计算“一致性”
- 输出逐周/逐季的一致性指标文件，并画可直接用于论文的图

说明：
- 一致性度量的核心思想：
  我们并不知道真实观众投票数（被保密），但我们知道每周谁被淘汰。
  在给定某种“评委+粉丝”合成规则后，如果模型预测的粉丝份额与真实淘汰结果能匹配，
  则说明该粉丝份额估计与节目淘汰机制具有“规则一致性”。
  这里的“一致性”不是回归误差，而是：模型在每周是否把真实淘汰者判到规则要求的“淘汰集合”。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _find_file(candidates: list[Path]) -> Path:
	"""在多个候选路径中找到第一个存在的文件。"""
	for p in candidates:
		if p.exists() and p.is_file():
			return p
	raise FileNotFoundError(
		"未找到数据文件。已尝试路径：\n" + "\n".join(str(p) for p in candidates)
	)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
	"""Spearman 相关（对常数列/缺失做鲁棒处理）。"""
	x = pd.to_numeric(x, errors="coerce")
	y = pd.to_numeric(y, errors="coerce")
	if x.notna().sum() < 3 or y.notna().sum() < 3:
		return np.nan
	if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
		return np.nan
	return float(x.corr(y, method="spearman"))


def _parse_is_eliminated(results: pd.Series, week: pd.Series) -> pd.Series:
	"""从 results 字段解析淘汰周。

原理（中文注释）：
- cleaned 数据里通常有 results 文本，如 “Eliminated Week 6”。
- 若文本中标注的淘汰周 == 当前记录的 week，则该记录为该周被淘汰。
"""

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


def prepare_merged_data(
	cleaned_path: Path,
	fan_path: Path,
) -> pd.DataFrame:
	"""读取并合并数据，统一关键字段、补齐所需特征。"""

	cleaned = pd.read_csv(cleaned_path)
	fan = pd.read_csv(fan_path)

	# 统一关键键（避免空格/大小写导致的 join 失败）
	for frame in (cleaned, fan):
		for col in ("celebrity_name", "ballroom_partner"):
			if col in frame.columns:
				frame[col] = frame[col].astype(str).str.strip()
		for col in ("season", "week"):
			if col in frame.columns:
				frame[col] = pd.to_numeric(frame[col], errors="coerce")

	# 在 cleaned 上构造 judge_percent 与 is_eliminated（让度量严格依赖真实规则字段）
	if "total_judge_score" not in cleaned.columns:
		raise ValueError("cleaned 数据缺少 total_judge_score 列，无法计算 judge_percent")

	cleaned = cleaned.copy()
	cleaned["week_total_score"] = cleaned.groupby(["season", "week"])["total_judge_score"].transform(
		"sum"
	)
	cleaned["judge_percent"] = cleaned["total_judge_score"] / cleaned["week_total_score"].replace(0, np.nan)

	if "results" in cleaned.columns:
		cleaned["is_eliminated"] = _parse_is_eliminated(cleaned["results"], cleaned["week"])
	elif "is_eliminated" not in cleaned.columns:
		raise ValueError("cleaned 数据缺少 results/is_eliminated，无法识别真实淘汰者")

	# fan_vote_final_fixed 里应有 fan_vote_normalized；若没有则用 fan_vote_est_mean 归一化得到
	fan = fan.copy()
	if "fan_vote_normalized" not in fan.columns:
		if "fan_vote_est_mean" not in fan.columns:
			raise ValueError(
				"fan_vote_final_fixed.csv 缺少 fan_vote_normalized 且缺少 fan_vote_est_mean，无法得到粉丝份额"
			)
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
			"fan_vote_est_mean",
			"fan_vote_est_std",
			"certainty",
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

	# 避免极端数值影响排名
	merged["fan_vote_normalized"] = pd.to_numeric(merged["fan_vote_normalized"], errors="coerce")
	merged["fan_vote_normalized"] = merged["fan_vote_normalized"].clip(lower=0)

	return merged


def weekly_consistency_metrics(group: pd.DataFrame) -> Optional[Dict[str, float]]:
	"""对单个 (season, week) 计算一致性指标。

原理（中文注释）：
- 节目合成规则分段：
  * S3-27：百分比法（Percent Method）
	评委分数转为当周百分比 judge_percent，与粉丝份额 fan_vote_normalized 相加得到合成分 comb。
	合成分越小越危险；真实淘汰者应落在 comb 最小的 n_elim 人里。
  * S1-2、S28-34：排名法（Rank Method）
	将评委分数与粉丝份额各自转为排名，相加得到 comb_rank。
	comb_rank 越大越危险；真实淘汰者应落在 comb_rank 最大的 n_elim 人里。
	其中 S28+ 含 Judges' Save：淘汰者来自 bottom-2，因此命中条件改为“真实淘汰者在 bottom-2 内”。

- 一致性（hit）定义：
  每周若模型根据规则得到的“危险集合”包含真实淘汰者，则 hit=1，否则 hit=0。
"""

	if group.empty:
		return None

	season = int(group["season"].iloc[0])
	week = int(group["week"].iloc[0])
	n_comp = int(group["celebrity_name"].nunique())

	if "is_eliminated" not in group.columns:
		return None
	elim_mask = group["is_eliminated"].fillna(False).astype(bool)
	n_elim = int(elim_mask.sum())
	if n_elim == 0:
		return None

	g = group.copy()
	# 基础相关性：评委与粉丝是否“同向”
	spearman_judge_fan = _safe_spearman(g["judge_percent"], g["fan_vote_normalized"])

	# 争议度：越接近 1 表示越“反向/不一致”（用 1-|rho| 表示强弱，不区分正负）
	controversy = np.nan
	if pd.notna(spearman_judge_fan):
		controversy = float(1.0 - abs(spearman_judge_fan))

	# 规则分段
	if 3 <= season <= 27:
		method = "percent"
		g["comb"] = g["judge_percent"] + g["fan_vote_normalized"]
		# 预测危险集合：comb 最小的 n_elim
		pred_danger_idx = g["comb"].nsmallest(n_elim).index
		hit = int(all(i in pred_danger_idx for i in g[elim_mask].index))

		# margin：用“真实淘汰者里 comb 最大者”与“危险集合阈值”比较
		# margin>0 表示淘汰者确实在危险集合中且有安全余量
		cutoff = float(g["comb"].nsmallest(n_elim).max())
		elim_worst = float(g.loc[elim_mask, "comb"].max())
		margin = float(cutoff - elim_worst)

	else:
		method = "rank"
		j_rank = g["total_judge_score"].rank(ascending=False, method="min")
		f_rank = g["fan_vote_normalized"].rank(ascending=False, method="min")
		g["comb"] = j_rank + f_rank

		if season >= 28:
			method = "rank_judges_save"
			pred_bottom_two = g["comb"].nlargest(2).index
			hit = int(any(i in pred_bottom_two for i in g[elim_mask].index))
			cutoff = float(g["comb"].nlargest(2).min())
			elim_best = float(g.loc[elim_mask, "comb"].min())
			# margin>0：淘汰者 comb >= bottom2 的阈值
			margin = float(elim_best - cutoff)
		else:
			pred_danger_idx = g["comb"].nlargest(n_elim).index
			hit = int(all(i in pred_danger_idx for i in g[elim_mask].index))
			cutoff = float(g["comb"].nlargest(n_elim).min())
			elim_best = float(g.loc[elim_mask, "comb"].min())
			margin = float(elim_best - cutoff)

	# 淘汰者的粉丝份额分位（越低越符合直觉：票少更危险）
	fan_rank_pct = g["fan_vote_normalized"].rank(pct=True, ascending=True)
	elim_fan_pct_mean = float(fan_rank_pct.loc[elim_mask].mean())

	certainty_elim_mean = np.nan
	if "certainty" in g.columns:
		certainty_elim_mean = float(pd.to_numeric(g.loc[elim_mask, "certainty"], errors="coerce").mean())

	return {
		"season": season,
		"week": week,
		"n_competitors": n_comp,
		"n_eliminated": n_elim,
		"method": method,
		"hit": hit,
		"margin": margin,
		"spearman_judge_fan": spearman_judge_fan,
		"controversy": controversy,
		"elim_fan_pct_mean": elim_fan_pct_mean,
		"elim_certainty_mean": certainty_elim_mean,
	}


def compute_metrics(merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
	weekly_rows: list[Dict[str, float]] = []
	for (_, _), g in merged.groupby(["season", "week"], sort=True):
		row = weekly_consistency_metrics(g)
		if row is not None:
			weekly_rows.append(row)

	weekly = pd.DataFrame(weekly_rows).sort_values(["season", "week"]).reset_index(drop=True)
	if weekly.empty:
		raise RuntimeError("未计算出任何周度指标：请检查 is_eliminated 或数据是否为空")

	season = (
		weekly.groupby(["season", "method"], as_index=False)
		.agg(
			n_weeks=("hit", "size"),
			accuracy=("hit", "mean"),
			margin_mean=("margin", "mean"),
			controversy_mean=("controversy", "mean"),
		)
		.sort_values(["season", "method"])
	)

	overall = {
		"overall_accuracy": float(weekly["hit"].mean()),
		"n_weeks": int(weekly.shape[0]),
		"n_hits": int(weekly["hit"].sum()),
		"n_misses": int((1 - weekly["hit"]).sum()),
	}
	return weekly, season, overall


def make_paper_plots(weekly: pd.DataFrame, out_dir: Path) -> None:
	"""生成论文风格图并保存到 out_dir。"""

	out_dir.mkdir(parents=True, exist_ok=True)

	# 尽量使用 seaborn；若环境没有 seaborn，则回退到 matplotlib
	try:
		import seaborn as sns  # type: ignore

		sns.set_theme(style="whitegrid")
		use_sns = True
	except Exception:
		use_sns = False

	import matplotlib as mpl
	import matplotlib.pyplot as plt

	# 字体：优先英文字体；若需要中文标题可在本地改为 SimHei
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

	# (1) Season-Week 命中热力图
	pivot = weekly.pivot(index="season", columns="week", values="hit")
	fig1, ax1 = plt.subplots(figsize=(11.5, 6.3))
	if use_sns:
		import seaborn as sns  # type: ignore

		sns.heatmap(
			pivot,
			ax=ax1,
			cmap=sns.color_palette(["#d73027", "#1a9850"], as_cmap=True),
			cbar_kws={"label": "Hit (1) / Miss (0)"},
			linewidths=0.2,
			linecolor="#eeeeee",
			vmin=0,
			vmax=1,
		)
	else:
		im = ax1.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
		cb = fig1.colorbar(im, ax=ax1)
		cb.set_label("Hit (1) / Miss (0)")
		ax1.set_xticks(np.arange(pivot.shape[1]))
		ax1.set_xticklabels(pivot.columns.tolist())
		ax1.set_yticks(np.arange(pivot.shape[0]))
		ax1.set_yticklabels(pivot.index.tolist())
	ax1.set_title("Rule-Consistency Heatmap by Season & Week")
	ax1.set_xlabel("Week")
	ax1.set_ylabel("Season")
	fig1.tight_layout()
	fig1.savefig(out_dir / "fig_consistency_heatmap.png")

	# (2) 各赛季一致性（折线）+ 规则切换标注
	season_acc = weekly.groupby("season", as_index=False).agg(accuracy=("hit", "mean"), n_weeks=("hit", "size"))
	fig2, ax2 = plt.subplots(figsize=(11.5, 4.6))
	ax2.plot(season_acc["season"], season_acc["accuracy"], marker="o", color="#2b8cbe", linewidth=2)
	ax2.set_ylim(0, 1.02)
	ax2.set_title("Consistency Accuracy by Season")
	ax2.set_xlabel("Season")
	ax2.set_ylabel("Accuracy")
	# 规则区间阴影
	ax2.axvspan(0.5, 2.5, color="#fee8c8", alpha=0.35, label="Rank (S1-2)")
	ax2.axvspan(2.5, 27.5, color="#e5f5e0", alpha=0.35, label="Percent (S3-27)")
	ax2.axvspan(27.5, season_acc["season"].max() + 0.5, color="#deebf7", alpha=0.35, label="Rank + Judges' Save (S28+)")
	ax2.legend(loc="lower right", frameon=True)
	ax2.grid(True, alpha=0.25)
	fig2.tight_layout()
	fig2.savefig(out_dir / "fig_season_accuracy.png")

	# (3) “争议度”与错误的关系：散点图（越争议越容易 miss，用于论文解释）
	fig3, ax3 = plt.subplots(figsize=(7.2, 5.2))
	plot_df = weekly.copy()
	plot_df["hit_label"] = plot_df["hit"].map({0: "Miss", 1: "Hit"})
	if use_sns:
		import seaborn as sns  # type: ignore

		sns.scatterplot(
			data=plot_df,
			x="controversy",
			y="margin",
			hue="hit_label",
			style="method",
			palette={"Hit": "#1a9850", "Miss": "#d73027"},
			ax=ax3,
			alpha=0.85,
		)
	else:
		for label, color in [("Hit", "#1a9850"), ("Miss", "#d73027")]:
			sub = plot_df[plot_df["hit_label"] == label]
			ax3.scatter(sub["controversy"], sub["margin"], s=24, alpha=0.85, label=label, c=color)
	ax3.axhline(0, color="#666666", linewidth=1, alpha=0.6)
	ax3.set_title("Misses Concentrate Under High Controversy")
	ax3.set_xlabel("Controversy = 1 - |Spearman(judge, fan)|")
	ax3.set_ylabel("Safety Margin (rule-specific)")
	ax3.legend(frameon=True)
	ax3.grid(True, alpha=0.25)
	fig3.tight_layout()
	fig3.savefig(out_dir / "fig_controversy_vs_margin.png")


def main() -> None:
	here = Path(__file__).resolve().parent
	workspace_root = here.parents[2]  # .../codeMCM

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
	weekly, season, overall = compute_metrics(merged)

	out_dir = here / "consistency_outputs"
	out_dir.mkdir(parents=True, exist_ok=True)

	weekly_path = out_dir / "consistency_metrics_weekly.csv"
	season_path = out_dir / "consistency_metrics_season.csv"
	weekly.to_csv(weekly_path, index=False, encoding="utf-8-sig")
	season.to_csv(season_path, index=False, encoding="utf-8-sig")

	fig_dir = out_dir / "figures"
	make_paper_plots(weekly, fig_dir)

	print("\n[RESULT] Overall accuracy: {:.2%} (hits={}, weeks={})".format(
		overall["overall_accuracy"], overall["n_hits"], overall["n_weeks"]
	))
	print(f"[OUTPUT] weekly metrics: {weekly_path}")
	print(f"[OUTPUT] season metrics: {season_path}")
	print(f"[OUTPUT] figures dir:    {fig_dir}")


if __name__ == "__main__":
	main()

