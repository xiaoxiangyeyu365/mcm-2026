import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据加载与辅助函数
# ==========================================
# 请确保此时使用的是上一模块生成的 'cleaned_dwts_data.csv'
DATA_PATH = 'cleaned_dwts_data.csv'
df = pd.read_csv(DATA_PATH)


def get_scoring_method(season):
    """根据赛季返回计分规则"""
    if season <= 2: return 'rank'
    if season <= 27: return 'percent'
    return 'rank_save'  # S28+ 引入评委拯救


# ==========================================
# 2. Q1: 观众票数反演 (MCMC/Rejection Sampling)
# ==========================================
print("正在执行 Q1: 观众票数蒙特卡洛反演...")

estimated_votes = []
np.random.seed(2026)  # 固定随机种子以复现结果

# 按周遍历进行反演
for (season, week), week_data in df.groupby(['season', 'week']):
    if len(week_data) <= 1: continue  # 决赛或异常数据跳过

    contestants = week_data['celebrity_name'].values
    judge_scores = week_data['avg_score'].values
    n_contestants = len(contestants)

    # 识别本周被淘汰者
    # 注意：需解析 results 字段，这里简化逻辑：若 results 包含 "Week X" 且 X==week
    elim_mask = week_data['results'].astype(str).str.contains(f"Week {week}", regex=False).values
    has_elimination = any(elim_mask)

    valid_samples = []
    max_iter = 500  # 模拟次数

    for _ in range(max_iter):
        # 1. 随机生成观众票比例 (Dirichlet分布保证和为1)
        fan_shares = np.random.dirichlet(np.ones(n_contestants) * 2.0)

        # 2. 计算综合得分
        method = get_scoring_method(season)
        total_score = np.zeros(n_contestants)

        if 'rank' in method:
            # 排名制: 分数越高 -> Rank数值越小 (1st)。总分 = Judge_Rank + Fan_Rank
            # 注意: argsort两次得到排名 (0-based)
            j_rank = np.argsort(np.argsort(-judge_scores))  # 降序排，分高Rank小
            f_rank = np.argsort(np.argsort(-fan_shares))
            total_score = j_rank + f_rank  # 数值越小越好

            # 淘汰判定: Rank和 最大者被淘汰
            sim_elim_idx = np.argmax(total_score)

        else:  # percent
            # 百分比制: (Judge% + Fan%) / 2
            j_pct = judge_scores / (np.sum(judge_scores) + 1e-9)
            total_score = (j_pct + fan_shares) * 50
            # 淘汰判定: 总分 最小者被淘汰
            sim_elim_idx = np.argmin(total_score)

        # 3. 验证是否符合历史事实
        if has_elimination:
            actual_elim_idx = np.where(elim_mask)[0][0]

            # 宽松验证: 模拟的淘汰者 是 真实淘汰者 (或在 Rank制下处于 Bottom 2)
            # 考虑到模拟的随机性，放宽到 Bottom 2 以保证有解
            if 'rank' in method:
                bottom_2 = np.argsort(total_score)[-2:]  # Rank和最大的两人
                if actual_elim_idx in bottom_2:
                    valid_samples.append(fan_shares)
            else:
                bottom_2 = np.argsort(total_score)[:2]  # 分数最小的两人
                if actual_elim_idx in bottom_2:
                    valid_samples.append(fan_shares)
        else:
            valid_samples.append(fan_shares)

    # 聚合结果
    if valid_samples:
        avg_shares = np.mean(valid_samples, axis=0)
    else:
        # 若无可行解，假设均匀分布但给予淘汰者惩罚 (Fallback)
        avg_shares = np.ones(n_contestants) / n_contestants
        if has_elimination:
            avg_shares[np.where(elim_mask)[0][0]] *= 0.8
            avg_shares /= avg_shares.sum()

    for i, name in enumerate(contestants):
        estimated_votes.append({
            'season': season,
            'week': week,
            'celebrity_name': name,
            'est_fan_share': avg_shares[i]
        })

# 合并反演结果
est_df = pd.DataFrame(estimated_votes)
full_df = pd.merge(df, est_df, on=['season', 'week', 'celebrity_name'], how='left')
full_df['est_fan_share'] = full_df['est_fan_share'].fillna(0.1)  # 填充缺失
print("Q1 完成。估算数据已合并。")

# ==========================================
# 3. Q2: 赛制对比 (Counterfactual Simulation)
# ==========================================
# 简单计算：若所有赛季都用 百分比制，结果差异多大？
# 计算 "Score Gap"：评委分占比 - 观众分占比
full_df['score_gap'] = (full_df['avg_score'] / 10) - (full_df['est_fan_share'] * 10)
# 注: est_fan_share是单周占比，需乘以系数调整量纲对比

# ==========================================
# 4. Q4: 驱动因素分析 (Random Forest)
# ==========================================
print("正在执行 Q4: 随机森林归因分析...")

# 特征准备: 年龄 + 评委分 + 行业Dummy
ind_cols = [c for c in full_df.columns if c.startswith('ind_')]
feature_cols = ['celebrity_age_during_season', 'avg_score'] + ind_cols

# 剔除无法训练的行
model_data = full_df.dropna(subset=feature_cols + ['est_fan_share'])

X = model_data[feature_cols]
y = model_data['est_fan_share']

# 模型训练
rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X, y)

# 提取特征重要性
importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

print("\n=== 影响观众票数的 Top 5 因素 ===")
print(feature_imp_df.head(5))

# ==========================================
# 5. 保存结果用于可视化
# ==========================================
full_df.to_csv('solved_dwts_results.csv', index=False)
print("\n结果已保存至 'solved_dwts_results.csv'")