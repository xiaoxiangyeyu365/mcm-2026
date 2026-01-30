# =============================================
# 数据预处理代码（2026 MCM Problem C）
# 作者：MCM参赛团队
# 核心功能：数据清洗、特征工程、可视化分析、数据保存
# =============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns

# --------------------------
# 1. 数据加载（请替换为你的本地路径）
# --------------------------
DATA_PATH = "2026_MCM_Problem_C_Data.csv"  # 原始数据路径
OUTPUT_PATH = "processed_data\\cleaned_dwts_data.csv"          # 处理后数据输出路径
df = pd.read_csv(DATA_PATH, encoding='utf-8')

# --------------------------
# 2. 预处理前可视化分析（必须做：验证数据问题）
# --------------------------
plt.rcParams['font.sans-serif'] = ['Arial']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Preprocessing - Before Cleaning', fontsize=16, fontweight='bold')

# 2.1 缺失值热力图
missing_data = df.isnull().sum().sort_values(ascending=False)[:10]  # 前10个缺失字段
axes[0,0].barh(range(len(missing_data)), missing_data.values, color='#3498db')
axes[0,0].set_yticks(range(len(missing_data)))
axes[0,0].set_yticklabels(missing_data.index)
axes[0,0].set_xlabel('Missing Values Count')
axes[0,0].set_title('Top 10 Missing Fields')

# 2.2 评委分数分布（以week1为例）
week1_scores = df[['week1_judge1_score', 'week1_judge2_score', 'week1_judge3_score']].values.flatten()
week1_scores = week1_scores[~np.isnan(week1_scores)]  # 剔除NaN
axes[0,1].hist(week1_scores, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[0,1].set_xlabel('Judge Score')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Week 1 Judge Scores Distribution')

# 2.3 选手行业分布
industry_counts = df['celebrity_industry'].value_counts()
axes[1,0].pie(industry_counts.values, labels=industry_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette('Set3'), startangle=90)
axes[1,0].set_title('Celebrity Industry Distribution')

# 2.4 异常值检测（箱线图）
valid_scores = []
for col in df.columns:
    if 'judge' in col and 'score' in col:
        valid_scores.extend(df[col].dropna().values)
axes[1,1].boxplot(valid_scores, patch_artist=True, boxprops=dict(facecolor='#2ecc71'))
axes[1,1].set_ylabel('Judge Score')
axes[1,1].set_title('Judge Scores Boxplot (Outlier Detection)')
axes[1,1].set_ylim(0, 12)  # 限定范围，突出异常值

plt.tight_layout()
plt.savefig('preprocessing_before.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------
# 3. 数据清洗（核心步骤）
# --------------------------
# 3.1 缺失值处理
# 第4评委分数填充
judge_cols = ['week{}_judge{}_score'.format(w, j) for w in range(1,12) for j in range(1,5)]
for w in range(1,12):
    week_judge_cols = ['week{}_judge{}_score'.format(w, j) for j in range(1,5)]
    df[week_judge_cols] = df[week_judge_cols].fillna(df[week_judge_cols].mean(axis=1), axis=0)

# 基础字段缺失填充
df['celebrity_homestate'] = df['celebrity_homestate'].fillna('Unknown')
df['celebrity_homecountry/region'] = df['celebrity_homecountry/region'].fillna('Unknown')

# 3.2 特殊值（0分）处理
df['is_eliminated'] = 0
for idx, row in df.iterrows():
    if 'Eliminated Week' in row['results']:
        eliminate_week = int(row['results'].split(' ')[-1])
        current_week = int(row['week'] if 'week' in row else 0)  # 需根据数据结构调整week提取逻辑
        if current_week > eliminate_week:
            df.loc[idx, 'is_eliminated'] = 1

# 3.3 异常值处理（IQR法）
valid_scores = []
for col in judge_cols:
    valid_scores.extend(df[col].dropna().values)
Q1 = np.percentile(valid_scores, 25)
Q3 = np.percentile(valid_scores, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 修正异常值
for col in judge_cols:
    df[col] = np.where(df[col] < lower_bound, 1.0, df[col])
    df[col] = np.where(df[col] > upper_bound, 10.0, df[col])

# --------------------------
# 4. 特征工程
# --------------------------
# 4.1 衍生特征
# 每周总得分（均值）
for w in range(1,12):
    week_judge_cols = ['week{}_judge{}_score'.format(w, j) for j in range(1,5)]
    df[f'week{w}_total_score'] = df[week_judge_cols].mean(axis=1)

# 每周评分标准差
for w in range(1,12):
    week_judge_cols = ['week{}_judge{}_score'.format(w, j) for j in range(1,5)]
    df[f'week{w}_score_std'] = df[week_judge_cols].std(axis=1)

# 年龄分组
df['age_group'] = pd.cut(df['celebrity_age_during_season'], bins=[0,25,40,100], labels=['<25', '25-40', '>40'])

# 国家二值编码
df['is_usa'] = (df['celebrity_homecountry/region'] == 'United States').astype(int)

# 4.2 分类型特征One-Hot编码
cat_features = ['celebrity_industry', 'ballroom_partner', 'age_group']
encoder = OneHotEncoder(sparse_output=False, drop='first')
cat_encoded = encoder.fit_transform(df[cat_features])
cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_features))

# 4.3 数值特征标准化
num_features = ['celebrity_age_during_season', 'week1_total_score', 'week1_score_std']  # 可扩展其他周特征
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# 4.4 合并特征（核心建模数据集）
model_df = pd.concat([df[['season', 'is_eliminated', 'is_usa'] + num_features], cat_df], axis=1)

# --------------------------
# 5. 数据划分
# --------------------------
X = model_df.drop('is_eliminated', axis=1)  # 示例：以淘汰状态为目标变量，可根据建模需求替换
y = model_df['is_eliminated']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=df['season'])
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# --------------------------
# 6. 预处理后可视化分析
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Preprocessing - After Cleaning', fontsize=16, fontweight='bold')

# 6.1 清洗后评分分布
cleaned_scores = df['week1_total_score'].dropna()
axes[0,0].hist(cleaned_scores, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[0,0].set_xlabel('Week 1 Total Score')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Cleaned Total Score Distribution')

# 6.2 训练/验证/测试集分布
dataset_labels = ['Train', 'Validation', 'Test']
dataset_sizes = [len(X_train), len(X_val), len(X_test)]
axes[0,1].bar(dataset_labels, dataset_sizes, color=['#3498db', '#e67e22', '#2ecc71'], alpha=0.8)
axes[0,1].set_ylabel('Sample Count')
axes[0,1].set_title('Train/Validation/Test Split')
for i, v in enumerate(dataset_sizes):
    axes[0,1].text(i, v+2, str(v), ha='center', fontweight='bold')

# 6.3 年龄分组分布
age_group_counts = df['age_group'].value_counts()
axes[1,0].bar(age_group_counts.index, age_group_counts.values, color='#f39c12', alpha=0.7)
axes[1,0].set_xlabel('Age Group')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Celebrity Age Group Distribution (After Processing)')

# 6.4 特征相关性热力图（前10个核心特征）
corr_matrix = model_df.iloc[:, :10].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1,1])
axes[1,1].set_title('Core Features Correlation Heatmap')

plt.tight_layout()
plt.savefig('preprocessing_after.png', dpi=300, bbox_inches='tight')
plt.show()

# --------------------------
# 7. 数据保存（通用格式）
# --------------------------
# 保存完整处理后数据（CSV格式）
model_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

# 保存训练/验证/测试集（CSV格式）
train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
val_df = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

train_df.to_csv('train_data.csv', index=False, encoding='utf-8')
val_df.to_csv('val_data.csv', index=False, encoding='utf-8')
test_df.to_csv('test_data.csv', index=False, encoding='utf-8')

# Matlab格式保存（可选）
# from scipy.io import savemat
# savemat('processed_data.mat', {'model_df': model_df.values, 'feature_names': model_df.columns})

print("数据预处理完成！文件保存至：", OUTPUT_PATH)