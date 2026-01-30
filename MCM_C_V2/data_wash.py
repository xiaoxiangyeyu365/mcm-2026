import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. 加载数据
df = pd.read_csv('codeMCM\\2026\\c\\2026_MCM_Problem_C_Data.csv')

# 2. 宽表转长表 (Wide to Long)
# 提取所有包含 'weekX_judgeY_score' 的列 
score_cols = [col for col in df.columns if 'week' in col and 'score' in col]
id_vars = ['celebrity_name', 'ballroom_partner', 'celebrity_industry',
           'celebrity_age_during_season', 'season', 'results', 'placement']

# 使用 melt 将分数列堆叠
df_long = pd.melt(df, id_vars=id_vars, value_vars=score_cols,
                  var_name='week_judge', value_name='score')

# 提取 'week' 编号
df_long['week'] = df_long['week_judge'].str.extract(r'week(\d+)').astype(int)

# 3. 处理评委分数：计算每周总分与均分
# 考虑通常有 3-4 个评委，NaN 代表无该评委或该周未运行 [cite: 527, 528]
weekly_scores = df_long.groupby(['season', 'week', 'celebrity_name']).agg(
    total_judge_score=('score', 'sum'), # 每周总评委分数
    avg_judge_score=('score', 'mean')    # 每周平均评委分数
).reset_index()

# 合并回选手个人静态信息
base_info = df[id_vars].copy()
df_cleaned = pd.merge(weekly_scores, base_info, on=['season', 'celebrity_name'], how='left')

# 4. 数据清洗过滤
# 剔除总分为 0 的记录（代表该周选手已淘汰或数据无效） [cite: 531]
df_cleaned = df_cleaned[df_cleaned['total_judge_score'] > 0].copy()

# 5. 特征工程 (Feature Engineering)
# A. 行业 (Industry) - 哑变量转换 (One-hot Encoding)
df_cleaned = pd.get_dummies(df_cleaned, columns=['celebrity_industry'], prefix='ind')

# B. 年龄 (Age) - Min-Max 归一化，适合 XGBoost 处理
scaler = MinMaxScaler()
df_cleaned['age_scaled'] = scaler.fit_transform(df_cleaned[['celebrity_age_during_season']])

# C. 累计参赛周数 (Cumulative Weeks)
df_cleaned['cum_weeks'] = df_cleaned.groupby(['season', 'celebrity_name'])['week'].cumcount() + 1

# D. 赛季进度 (Season Progress)
max_weeks = df_cleaned.groupby('season')['week'].transform('max')
df_cleaned['season_progress'] = df_cleaned['week'] / max_weeks

# 6. 保存清洗后的数据供模型调用
df_cleaned.to_csv('cleaned_dwts_data.csv', index=False)

print("清洗完成！处理后样本数：", len(df_cleaned))
