import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 路径配置 (请在此处修改为你的本地路径)
# ==========================================
INPUT_PATH = '2026_MCM_Problem_C_Data.csv'  # 输入文件路径
OUTPUT_DIR = 'processed_data/'              # 输出文件夹
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"正在读取数据: {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH)

# ==========================================
# 2. 数据清洗与重构 (Wide to Long)
# ==========================================
# 2.1 处理 N/A 值
df.replace('N/A', np.nan, inplace=True)

# 2.2 转换宽表为长表 (Melt)
# 提取所有评委打分列
score_cols = [c for c in df.columns if 'judge' in c]
meta_cols = [c for c in df.columns if c not in score_cols]

# 使用 melt 将周次和评委分摊平
df_long = pd.melt(df, id_vars=meta_cols, value_vars=score_cols,
                  var_name='week_judge_str', value_name='raw_score')

# 2.3 解析 'week_judge_str' (例如: week1_judge1_score)
# 提取 Week 和 Judge 编号
df_long['week'] = df_long['week_judge_str'].str.extract(r'week(\d+)_').astype(int)
df_long['judge'] = df_long['week_judge_str'].str.extract(r'judge(\d+)_').astype(int)

# 2.4 类型转换
df_long['raw_score'] = pd.to_numeric(df_long['raw_score'], errors='coerce')

# ==========================================
# 3. 特征工程
# ==========================================
# 3.1 聚合计算每周的总分和均分 (处理评委人数变化)
# 按 [Season, Celebrity, Week] 分组
df_weekly = df_long.groupby(['season', 'celebrity_name', 'week'])['raw_score'].agg(
    total_score='sum',
    judge_count='count',  # 有效评委数 (不含 NaN)
    avg_score='mean',
    score_std='std'
).reset_index()

# 过滤掉未参赛的周次 (即 judge_count == 0 的行)
df_weekly = df_weekly[df_weekly['judge_count'] > 0].copy()

# 3.2 合并元数据
meta_df = df[meta_cols].drop_duplicates(subset=['season', 'celebrity_name'])
final_df = pd.merge(df_weekly, meta_df, on=['season', 'celebrity_name'], how='left')

# 3.3 行业 One-Hot 编码
final_df = pd.get_dummies(final_df, columns=['celebrity_industry'], prefix='ind', dummy_na=True)

# ==========================================
# 4. 可视化分析模块
# ==========================================
plt.figure(figsize=(12, 6))

# 图1: 处理后的评分分布 (检查是否符合 1-10 分逻辑)
plt.subplot(1, 2, 1)
sns.histplot(final_df['avg_score'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Average Weekly Scores (After Cleaning)')
plt.xlabel('Average Score')

# 图2: 缺失值/零分检查
plt.subplot(1, 2, 2)
sns.boxplot(x='judge_count', y='avg_score', data=final_df)
plt.title('Score vs Number of Judges')
plt.xlabel('Number of Judges')
plt.ylabel('Average Score')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'preprocessing_quality_check.png'), dpi=300)
print("可视化图表已保存至输出目录。")

# ==========================================
# 5. 数据保存
# ==========================================
output_file = os.path.join(OUTPUT_DIR, 'cleaned_dwts_data.csv')
final_df.to_csv(output_file, index=False)

print("="*50)
print(f"处理完成! 数据已保存至: {output_file}")
print(f"数据维度: {final_df.shape}")
print("前5行预览:")
print(final_df[['season', 'celebrity_name', 'week', 'avg_score', 'results']].head())
print("="*50)