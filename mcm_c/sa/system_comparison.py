import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ===================== 全局工具函数 =====================
# 清理列名中的BOM字符（复用第一问逻辑）
def clean_bom_columns(df):
    """移除列名开头的BOM字符（ï»¿/ufeff）"""
    df.columns = [col.lstrip('\ufeff').lstrip('ï»¿') for col in df.columns]
    return df


# ===================== 1. 配置与加载数据 =====================
# 可视化字体（解决乱码）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载第一问生成的投票估算结果（cp1252编码）
df = pd.read_csv("estimated_votes.csv", encoding="cp1252")
# 加载原始数据（用于特征分析）
df_original = pd.read_csv("2026_MCM_Problem_C_Data.csv", encoding="cp1252")
df_original = clean_bom_columns(df_original)

# 清理空值和异常值
df = df.dropna(subset=["weekly_judge_total", "estimated_fan_vote", "season", "week"])
df["week"] = df["week"].astype(int)
df["season"] = df["season"].astype(int)


# ===================== 2. 定义两种赛制的计算逻辑 =====================
def calculate_rank_based_system(df):
    """
    排名制：
    - 评委分排名（降序，1=最高） + 粉丝票排名（降序，1=最高） = 总排名
    - 总排名越高（数值越大），淘汰风险越高
    """
    df_copy = df.copy()
    # 计算每周内的评委分排名（降序）
    df_copy["judge_rank"] = df_copy.groupby(["season", "week"])["weekly_judge_total"].rank(ascending=False,
                                                                                           method="min")
    # 计算每周内的粉丝票排名（降序）
    df_copy["fan_vote_rank"] = df_copy.groupby(["season", "week"])["estimated_fan_vote"].rank(ascending=False,
                                                                                              method="min")
    # 总排名（越小越好）
    df_copy["total_rank"] = df_copy["judge_rank"] + df_copy["fan_vote_rank"]
    # 判定是否淘汰（每周总排名最高的选手）
    df_copy["rank_based_eliminated"] = df_copy.groupby(["season", "week"])["total_rank"].transform("max") == df_copy[
        "total_rank"]
    df_copy["rank_based_eliminated"] = df_copy["rank_based_eliminated"].astype(int)
    return df_copy


def calculate_percent_based_system(df):
    """
    百分比制：
    - 评委分占比（个人评委分/当周所有选手评委分总和） + 粉丝票占比（个人票数/当周所有选手票数总和） = 总占比
    - 总占比越低，淘汰风险越高
    """
    df_copy = df.copy()
    # 计算每周内的评委分总和
    df_copy["weekly_judge_sum"] = df_copy.groupby(["season", "week"])["weekly_judge_total"].transform("sum")
    # 评委分占比（避免除0）
    df_copy["judge_pct"] = df_copy["weekly_judge_total"] / df_copy["weekly_judge_sum"].replace(0, 1)

    # 计算每周内的粉丝票总和
    df_copy["weekly_fan_sum"] = df_copy.groupby(["season", "week"])["estimated_fan_vote"].transform("sum")
    # 粉丝票占比（避免除0）
    df_copy["fan_vote_pct"] = df_copy["estimated_fan_vote"] / df_copy["weekly_fan_sum"].replace(0, 1)

    # 总占比（越大越好）
    df_copy["total_pct"] = df_copy["judge_pct"] + df_copy["fan_vote_pct"]
    # 判定是否淘汰（每周总占比最低的选手）
    df_copy["percent_based_eliminated"] = df_copy.groupby(["season", "week"])["total_pct"].transform("min") == df_copy[
        "total_pct"]
    df_copy["percent_based_eliminated"] = df_copy["percent_based_eliminated"].astype(int)
    return df_copy


# ===================== 3. 新增：评委额外筛选环节模拟 =====================
def simulate_judge_extra_screening(df_analysis):
    """
    模拟评委额外筛选环节：
    - 评委每周先淘汰评委分排名最后两位的选手
    - 最终淘汰：总排名最高 且 属于评委分最后两位
    """
    df_sim = df_analysis.copy()
    # 每周按评委分排名，标记最后两位
    df_sim["judge_rank_max"] = df_sim.groupby(["season", "week"])["judge_rank"].transform("max")
    df_sim["is_judge_last_two"] = (df_sim["judge_rank_max"] - df_sim["judge_rank"]) <= 1
    # 最终淘汰：总排名最高 + 评委分最后两位
    df_sim["simulated_eliminated"] = (df_sim["total_rank"] == df_sim.groupby(["season", "week"])[
        "total_rank"].transform("max")) & df_sim["is_judge_last_two"]
    df_sim["simulated_eliminated"] = df_sim["simulated_eliminated"].astype(int)
    # 计算与原排名制的差异率
    sim_diff = (df_sim["simulated_eliminated"] != df_sim["rank_based_eliminated"]).mean()
    return df_sim, sim_diff


# ===================== 4. 执行赛制计算与分析 =====================
# 4.1 应用两种基础赛制
df_analysis = calculate_rank_based_system(df)
df_analysis = calculate_percent_based_system(df_analysis)
# 4.2 模拟评委额外筛选
df_analysis, sim_diff = simulate_judge_extra_screening(df_analysis)
# 4.3 计算两种赛制的淘汰结果差异
df_analysis["elimination_diff"] = df_analysis["rank_based_eliminated"] != df_analysis["percent_based_eliminated"]

# ===================== 5. 核心结果统计 =====================
# 5.1 整体差异率
total_weeks = len(df_analysis.groupby(["season", "week"]).size())
diff_weeks = len(df_analysis[df_analysis["elimination_diff"]].groupby(["season", "week"]).size())
overall_diff_rate = diff_weeks / total_weeks if total_weeks > 0 else 0.0

# 5.2 争议选手分析（Bobby Bones）
controversial_name = "Bobby Bones"
if controversial_name in df_analysis["celebrity_name"].values:
    bobby_data = df_analysis[df_analysis["celebrity_name"] == controversial_name].sort_values(["season", "week"])
    bobby_elim_rank = bobby_data["rank_based_eliminated"].sum()
    bobby_elim_percent = bobby_data["percent_based_eliminated"].sum()
    bobby_elim_sim = bobby_data["simulated_eliminated"].sum()
else:
    # 若无Bobby Bones，选评委分最低但粉丝票最高的争议选手
    df_analysis["judge_fan_gap"] = df_analysis["fan_vote_rank"] - df_analysis["judge_rank"]
    gap_by_player = df_analysis.groupby("celebrity_name")["judge_fan_gap"].mean()
    controversial_idx = gap_by_player.idxmax() if not gap_by_player.empty else df_analysis["celebrity_name"].iloc[0]

    bobby_data = df_analysis[df_analysis["celebrity_name"] == controversial_idx].sort_values(["season", "week"])
    bobby_elim_rank = bobby_data["rank_based_eliminated"].sum()
    bobby_elim_percent = bobby_data["percent_based_eliminated"].sum()
    bobby_elim_sim = bobby_data["simulated_eliminated"].sum()
    controversial_name = controversial_idx

# 5.3 赛制友好度分析（评委分低但人气高的选手）
df_analysis["is_popular_underdog"] = df_analysis["judge_rank"] > df_analysis["fan_vote_rank"]
rank_underdog_elim = df_analysis[df_analysis["is_popular_underdog"]]["rank_based_eliminated"].mean() if df_analysis[
    "is_popular_underdog"].any() else 0.0
percent_underdog_elim = df_analysis[df_analysis["is_popular_underdog"]]["percent_based_eliminated"].mean() if \
df_analysis["is_popular_underdog"].any() else 0.0
sim_underdog_elim = df_analysis[df_analysis["is_popular_underdog"]]["simulated_eliminated"].mean() if df_analysis[
    "is_popular_underdog"].any() else 0.0

# 5.4 特征相关性分析（行业/年龄对评委分/粉丝票的影响）
df_feature = pd.merge(df_analysis, df_original[["celebrity_name", "celebrity_industry", "celebrity_age_during_season"]],
                      on="celebrity_name")
# 行业对评委分/粉丝票的影响
industry_judge = df_feature.groupby("celebrity_industry")["weekly_judge_total"].mean().sort_values(ascending=False)
industry_fan = df_feature.groupby("celebrity_industry")["estimated_fan_vote"].mean().sort_values(ascending=False)
# 年龄相关性
age_corr_judge = df_feature["celebrity_age_during_season"].corr(df_feature["weekly_judge_total"])
age_corr_fan = df_feature["celebrity_age_during_season"].corr(df_feature["estimated_fan_vote"])

# ===================== 6. 投票份额占比验证 =====================
print("=== 投票份额占比验证 ===")
# 粉丝票占比总和验证
fan_pct_check = df_analysis.groupby(["season", "week"])["fan_vote_pct"].sum().reset_index()
is_fan_pct_valid = (fan_pct_check["fan_vote_pct"] - 1).abs().max() < 1e-10
print(f"粉丝票占比总和是否≈1：{is_fan_pct_valid}")
# 评委分占比总和验证
judge_pct_check = df_analysis.groupby(["season", "week"])["judge_pct"].sum().reset_index()
is_judge_pct_valid = (judge_pct_check["judge_pct"] - 1).abs().max() < 1e-10
print(f"评委分占比总和是否≈1：{is_judge_pct_valid}")
print("=" * 60 + "\n")

# ===================== 7. 完整结果输出 =====================
print("===== 第二问：赛制对比分析完整结果 =====")
print(f"1. 两种基础赛制淘汰结果整体差异率：{overall_diff_rate:.2%}")
print(f"2. 增加评委额外筛选后，与原排名制的差异率：{sim_diff:.2%}")

print(f"\n3. 争议选手【{controversial_name}】淘汰情况：")
print(f"   - 排名制下被淘汰次数：{bobby_elim_rank}")
print(f"   - 百分比制下被淘汰次数：{bobby_elim_percent}")
print(f"   - 增加评委筛选后被淘汰次数：{bobby_elim_sim}")

print(f"\n4. 评委分低但人气高的选手淘汰率：")
print(f"   - 排名制：{rank_underdog_elim:.2%}")
print(f"   - 百分比制：{percent_underdog_elim:.2%}")
print(f"   - 增加评委筛选后：{sim_underdog_elim:.2%}")

print(f"\n5. 特征相关性分析：")
print(f"   - 年龄与评委分相关性：{age_corr_judge:.3f}")
print(f"   - 年龄与粉丝票相关性：{age_corr_fan:.3f}")
print(f"   - 评委分最高的行业：{industry_judge.index[0]}（平均分：{industry_judge.iloc[0]:.2f}）")
print(f"   - 粉丝票最高的行业：{industry_fan.index[0]}（平均票数：{industry_fan.iloc[0]:.0f}）")

print(f"\n6. 赛制推荐结论：")
if rank_underdog_elim < percent_underdog_elim:
    print("   ✅ 推荐基础赛制：排名制")
    print("   理由：更平衡评委专业度与观众喜好，降低人气选手淘汰率，提升节目观赏性")
else:
    print("   ✅ 推荐基础赛制：百分比制")
    print("   理由：更贴合评委专业评判逻辑，减少争议选手晋级，保证比赛专业性")
print(f"   📌 是否建议增加评委额外筛选：是（差异率{sim_diff:.2%}，可降低极端人气选手“躺赢”概率）")

# ===================== 8. 可视化输出（4张核心图表） =====================
# 图1：两种赛制+评委筛选的人气选手淘汰率对比
plt.figure(figsize=(10, 6))
labels = ["排名制", "百分比制", "排名制+评委筛选"]
underdog_rates = [rank_underdog_elim, percent_underdog_elim, sim_underdog_elim]
colors = ["orange", "green", "blue"]

plt.bar(labels, underdog_rates, color=colors)
plt.ylabel("淘汰率")
plt.title("评委分低但人气高的选手淘汰率对比")
plt.ylim(0, max(underdog_rates) * 1.2 if underdog_rates else 1.0)
# 标注数值
for i, v in enumerate(underdog_rates):
    plt.text(i, v + 0.01, f"{v:.2%}", ha="center", fontsize=12)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("underdog_elimination_rate_all.png", dpi=300, bbox_inches="tight")
plt.show()

# 图2：争议选手在三种规则下的淘汰风险
plt.figure(figsize=(12, 6))
bobby_data_plot = bobby_data.head(10) if not bobby_data.empty else bobby_data
if not bobby_data_plot.empty:
    x = bobby_data_plot["week"]
    # 归一化风险值（0-1）
    y1 = bobby_data_plot["total_rank"] / (
        bobby_data_plot["total_rank"].max() if bobby_data_plot["total_rank"].max() > 0 else 1)
    y2 = 1 - (bobby_data_plot["total_pct"] / (
        bobby_data_plot["total_pct"].max() if bobby_data_plot["total_pct"].max() > 0 else 1))
    y3 = bobby_data_plot["simulated_eliminated"].astype(float) * 1.0  # 模拟筛选的淘汰风险

    plt.plot(x, y1, label="排名制", color="orange", marker="o", linewidth=2)
    plt.plot(x, y2, label="百分比制", color="green", marker="s", linewidth=2)
    plt.plot(x, y3, label="排名制+评委筛选", color="blue", marker="^", linewidth=2)
    plt.xlabel("比赛周数")
    plt.ylabel("淘汰风险（0=无风险，1=极高风险）")
    plt.title(f"{controversial_name}在三种规则下的淘汰风险对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("controversial_player_risk_all.png", dpi=300, bbox_inches="tight")
    plt.show()

# 图3：行业对评委分/粉丝票的影响（前8个行业）
plt.figure(figsize=(12, 6))
top_industries = min(len(industry_judge), 8)
x = np.arange(top_industries)
width = 0.35

# 归一化行业得分（便于对比）
judge_norm = industry_judge.head(top_industries) / industry_judge.max()
fan_norm = industry_fan.head(top_industries) / industry_fan.max()

plt.bar(x - width / 2, judge_norm, width, label="评委分（归一化）", color="gray")
plt.bar(x + width / 2, fan_norm, width, label="粉丝票（归一化）", color="red")
plt.xlabel("行业")
plt.ylabel("归一化得分")
plt.title("各行业评委分与粉丝票对比（前8）")
plt.xticks(x, industry_judge.head(top_industries).index, rotation=45, ha="right")
plt.legend()
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("industry_impact.png", dpi=300, bbox_inches="tight")
plt.show()

# 图4：两种基础赛制每周淘汰结果差异
plt.figure(figsize=(12, 6))
week_diff = df_analysis.groupby(["season", "week"])["elimination_diff"].mean().reset_index()
week_diff["season_week"] = week_diff["season"].astype(str) + "-W" + week_diff["week"].astype(str)
week_diff = week_diff.head(20) if not week_diff.empty else week_diff

if not week_diff.empty:
    plt.bar(week_diff["season_week"], week_diff["elimination_diff"],
            color=["red" if x else "green" for x in week_diff["elimination_diff"]])
    plt.xlabel("赛季-周数")
    plt.ylabel("淘汰结果差异（1=不同，0=相同）")
    plt.title("两种基础赛制每周淘汰结果差异")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("system_elimination_diff.png", dpi=300, bbox_inches="tight")
    plt.show()

# ===================== 9. 保存完整结果 =====================
output_cols = [
    "celebrity_name", "season", "week", "weekly_judge_total", "estimated_fan_vote",
    "judge_rank", "fan_vote_rank", "total_rank", "rank_based_eliminated",
    "judge_pct", "fan_vote_pct", "total_pct", "percent_based_eliminated",
    "is_judge_last_two", "simulated_eliminated", "is_popular_underdog"
]
df_analysis[output_cols].to_csv("system_comparison_full_result.csv", index=False, encoding="cp1252")

# 保存特征分析结果
feature_cols = ["celebrity_name", "celebrity_industry", "celebrity_age_during_season",
                "weekly_judge_total", "estimated_fan_vote", "judge_rank", "fan_vote_rank"]
df_feature[feature_cols].drop_duplicates().to_csv("feature_analysis_result.csv", index=False, encoding="cp1252")

print("\n✅ 第二问完整结果已保存：")
print("   - system_comparison_full_result.csv（所有赛制的淘汰结果）")
print("   - feature_analysis_result.csv（特征相关性分析结果）")
print("✅ 生成核心可视化图表：")
print("   - underdog_elimination_rate_all.png（人气选手淘汰率对比）")
print("   - controversial_player_risk_all.png（争议选手淘汰风险）")
print("   - industry_impact.png（行业影响分析）")
print("   - system_elimination_diff.png（每周淘汰结果差异）")