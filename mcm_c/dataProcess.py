import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 路径配置
# ==========================================
INPUT_PATH = '2026_MCM_Problem_C_Data.csv'
OUTPUT_DIR = 'processed_data/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"正在读取数据: {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH)

# ==========================================
# 2. 数据清洗与重构 (Wide to Long)
# ==========================================
# 2.1 处理 N/A 值
df.replace(['N/A', 'NA', 'nan', ' '], np.nan, inplace=True) # 增加对空字符串的处理

# 2.2 转换宽表为长表 (Melt)
score_cols = [c for c in df.columns if 'judge' in c]
meta_cols = [c for c in df.columns if c not in score_cols]

df_long = pd.melt(df, id_vars=meta_cols, value_vars=score_cols,
                  var_name='week_judge_str', value_name='raw_score')

# 2.3 解析 'week_judge_str'
df_long['week'] = df_long['week_judge_str'].str.extract(r'week(\d+)_').astype(float).astype('Int64') # 安全转换为Int
df_long['judge'] = df_long['week_judge_str'].str.extract(r'judge(\d+)_').astype(float).astype('Int64')

# 2.4 类型转换
df_long['raw_score'] = pd.to_numeric(df_long['raw_score'], errors='coerce')

# ==========================================
# 3. 特征工程 (修复核心逻辑)
# ==========================================
# 3.1 聚合计算
# 注意：这里我们过滤掉 raw_score 为 0 的行，因为那通常意味着没跳舞
df_long_valid = df_long[df_long['raw_score'] > 0].copy()

df_weekly = df_long_valid.groupby(['season', 'celebrity_name', 'week'])['raw_score'].agg(
    total_score='sum',
    judge_count='count',
    avg_score='mean',
    score_std='std'
).reset_index()

# 3.2 合并元数据
meta_df = df[meta_cols].drop_duplicates(subset=['season', 'celebrity_name'])
final_df = pd.merge(df_weekly, meta_df, on=['season', 'celebrity_name'], how='left')

# 3.3 行业 One-Hot 编码
final_df = pd.get_dummies(final_df, columns=['celebrity_industry'], prefix='ind', dummy_na=True)

# ==========================================
# 4. 可视化分析模块 (异常修复版)
# ==========================================
# 设置更美观的绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# 【关键修复】：创建专门用于绘图的数据集，严格剔除 0 分
# 真实比赛最低分通常是 1-4 分，0 分绝对是未参赛数据
plot_df = final_df[final_df['avg_score'] > 0].copy()

plt.figure(figsize=(14, 6))

# 图1: 评分分布 (Histogram)
# 修复点：移除了0值干扰，添加了平均线，颜色更柔和
plt.subplot(1, 2, 1)
sns.histplot(plot_df['avg_score'], bins=20, kde=True, color='#3498db', edgecolor='white')
plt.axvline(plot_df['avg_score'].mean(), color='red', linestyle='--', label=f"Mean: {plot_df['avg_score'].mean():.2f}")
plt.title('Distribution of Valid Dance Scores (1-10 Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Average Judge Score')
plt.ylabel('Frequency')
plt.legend()

# 图2: 评委人数与分数的箱线图 (Boxplot)
# 修复点：将 judge_count 设为分类变量，避免X轴错乱；过滤掉极少数非标准评委数(如1或2人)
valid_judge_counts = plot_df[plot_df['judge_count'].isin([3, 4])]

plt.subplot(1, 2, 2)
sns.boxplot(x='judge_count', y='avg_score', data=valid_judge_counts, palette="Set2", width=0.5)
plt.title('Score Distribution by Panel Size (3 vs 4 Judges)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Judges')
plt.ylabel('Average Score')
# 添加抖动点显示实际数据密度
sns.stripplot(x='judge_count', y='avg_score', data=valid_judge_counts,
              size=2, color=".3", alpha=0.4, jitter=True)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'preprocessing_quality_check_fixed.png')
plt.savefig(save_path, dpi=300)
print(f"修复后的可视化图表已保存至: {save_path}")

# ==========================================
# 5. 数据保存
# ==========================================
output_file = os.path.join(OUTPUT_DIR, 'cleaned_dwts_data.csv')
final_df.to_csv(output_file, index=False)

print("="*50)
print(f"处理完成! 数据已保存至: {output_file}")
print(f"清洗后数据行数: {final_df.shape[0]}")
print("前5行预览 (Valid Scores Only):")
print(final_df[['season', 'celebrity_name', 'week', 'avg_score', 'judge_count']].head())
print("="*50)