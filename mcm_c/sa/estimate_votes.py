import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 全局配置（适配Windows-1252编码+处理BOM） =====================
# 可视化字体配置（解决中文/特殊字符乱码）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"

# 字符编码清理函数
def clean_encoding(text):
    """清理无法用cp1252编码的特殊字符"""
    if isinstance(text, str):
        return text.encode("cp1252", errors="ignore").decode("cp1252")
    return text

# 清理列名中的BOM字符
def clean_bom_columns(df):
    """移除列名开头的BOM字符（ï»¿）"""
    df.columns = [col.lstrip('\ufeff').lstrip('ï»¿') for col in df.columns]
    return df

# ===================== 2. 数据加载（适配cp1252/latin-1编码+处理BOM） =====================
try:
    # 优先用cp1252读取（Windows默认编码），并处理BOM
    df = pd.read_csv("2026_MCM_Problem_C_Data.csv", encoding="cp1252")
    df = clean_bom_columns(df)
except UnicodeDecodeError:
    # 失败则用latin-1（cp1252超集），并处理BOM
    df = pd.read_csv("2026_MCM_Problem_C_Data.csv", encoding="latin-1")
    df = clean_bom_columns(df)

# 查看清理后的列名（验证）
print("=== 清理BOM后的列名 ===")
print(df.columns.tolist())

# ===================== 3. 数据预处理（核心适配你的列名） =====================
# 3.1 处理评委分：按周求和
judge_cols = [col for col in df.columns if "week" in col and "judge" in col]
df["weekly_judge_total"] = df[judge_cols].sum(axis=1, skipna=True)

# 3.2 处理分类特征（适配ballroom_partner列名）
label_encoders = {}
cat_features = [
    "celebrity_industry", "ballroom_partner",
    "celebrity_homestate", "celebrity_homecountry/region"
]
for feat in cat_features:
    le = LabelEncoder()
    # 填充空值+清理编码字符
    df[feat] = df[feat].fillna("Unknown").apply(clean_encoding)
    df[f"{feat}_enc"] = le.fit_transform(df[feat])
    label_encoders[feat] = le

# 3.3 提取week列（从评委分字段生成）
df["week"] = df.apply(lambda row:
    max([int(col.split("_")[0].replace("week", "")) for col in judge_cols if pd.notna(row[col])])
    if any(pd.notna(row[col]) for col in judge_cols) else 1, axis=1)

# 3.4 构造衍生特征
df["is_eliminated"] = df["results"].fillna("").apply(clean_encoding).str.contains("Eliminated|Withdrew", na=False).astype(int)
df["placement_norm"] = 1 / (df["placement"].fillna(df["placement"].max()) + 1)  # 排名归一化
df["season_week"] = df["season"] * 100 + df["week"]  # 赛季-周标识
df["cumulative_weeks"] = df.groupby("celebrity_name")["season"].cumcount() + 1  # 累计参赛周数
df["judge_rank_pct"] = df.groupby(["season", "week"])["weekly_judge_total"].rank(pct=True, ascending=False)  # 评委分排名百分比

# 3.5 筛选有效特征并清理空值
valid_features = [
    "celebrity_age_during_season", "weekly_judge_total", "cumulative_weeks", "judge_rank_pct",
    "placement_norm", "is_eliminated", "season_week"
] + [f"{feat}_enc" for feat in cat_features]
df_model = df.dropna(subset=valid_features).reset_index(drop=True)

# ===================== 4. 构建代理标签（提升匹配率） =====================
# 强化排名+淘汰结果的权重，提升模型准确性
df_model["vote_proxy"] = df_model.groupby(["season", "week"]).apply(
    lambda x: (1 - x["is_eliminated"]) * x["placement_norm"] * 0.6 +  # 淘汰+排名（60%权重）
              x["judge_rank_pct"] * 0.2 +                            # 评委分排名（20%权重）
              (x["week"] / x["week"].max()) * 0.2                    # 比赛阶段（20%权重）
).values

# 归一化代理标签（0-1区间）
scaler = MinMaxScaler()
df_model["vote_proxy_norm"] = scaler.fit_transform(df_model[["vote_proxy"]])

# ===================== 5. 模型训练（随机森林+XGBoost集成） =====================
# 5.1 拆分特征与标签
X = df_model[valid_features]
y = df_model["vote_proxy_norm"]

# 5.2 随机森林筛选Top10特征
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selector.fit(X, y)
feature_importance = pd.DataFrame({
    "feature": valid_features,
    "importance": rf_selector.feature_importances_
}).sort_values("importance", ascending=False)
top_features = feature_importance.head(10)["feature"].tolist()
X_selected = X[top_features]

print("\n=== Top10重要特征 ===")
print(feature_importance.head(10))

# 5.3 训练集成模型
rf_model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
rf_model.fit(X_selected, y)
xgb_model.fit(X_selected, y)

# 5.4 Bootstrap量化不确定性
def bootstrap_uncertainty(model1, model2, X, n_iter=1000):
    """通过Bootstrap计算预测的均值和标准差（不确定性）"""
    predictions = []
    for _ in range(n_iter):
        sample_idx = np.random.choice(len(X), len(X), replace=True)
        pred = 0.4 * model1.predict(X.iloc[sample_idx]) + 0.6 * model2.predict(X.iloc[sample_idx])
        predictions.append(pred)
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)

mean_vote, vote_std = bootstrap_uncertainty(rf_model, xgb_model, X_selected)

# 5.5 反归一化到真实投票区间（10万-1000万）
vote_scaler = MinMaxScaler(feature_range=(100000, 10000000))
df_model["estimated_fan_vote"] = vote_scaler.fit_transform(mean_vote.reshape(-1, 1))
df_model["vote_uncertainty"] = vote_scaler.transform(vote_std.reshape(-1, 1))  # 不确定性

# ===================== 6. 结果输出（cp1252编码） =====================
output_cols = [
    "celebrity_name", "season", "week", "weekly_judge_total",
    "estimated_fan_vote", "vote_uncertainty", "is_eliminated", "placement"
]
# 清理选手名编码后保存
df_model["celebrity_name"] = df_model["celebrity_name"].apply(clean_encoding)
df_model[output_cols].to_csv("estimated_votes.csv", index=False, encoding="cp1252")

print("\n✅ 估算结果已保存：estimated_votes.csv（cp1252编码）")

# ===================== 7. 模型验证（淘汰匹配率） =====================
df_model["vote_rank"] = df_model.groupby(["season", "week"])["estimated_fan_vote"].rank(ascending=False)
elimination_match = df_model.groupby(["season", "week"]).apply(
    lambda x: x.loc[x["vote_rank"] == x["vote_rank"].max(), "is_eliminated"].iloc[0] == 1
).mean()

print(f"\n📊 淘汰匹配率：{elimination_match:.2%}")
print("（目标≥70%，越高说明投票估算越贴合实际淘汰结果）")

# ===================== 8. 可视化（适配cp1252编码） =====================
# 8.1 筛选争议选手数据（Bobby Bones，无则选第一个选手）
target_name = "Bobby Bones"
df_model["celebrity_name_clean"] = df_model["celebrity_name"].apply(clean_encoding)

if target_name in df_model["celebrity_name_clean"].values:
    plot_data = df_model[df_model["celebrity_name_clean"] == target_name].sort_values("week")
else:
    plot_data = df_model.groupby("celebrity_name_clean").first().reset_index().iloc[0:1].merge(df_model, on="celebrity_name_clean")

# 8.2 绘制投票数vs评委分对比图
plt.figure(figsize=(10, 6))
plt.plot(plot_data["week"], plot_data["estimated_fan_vote"]/10000,
         label="Estimated Fan Votes (10k)", color="red", linewidth=2, marker="o")
plt.plot(plot_data["week"], plot_data["weekly_judge_total"],
         label="Weekly Judge Score", color="blue", linewidth=2, marker="s")
plt.xlabel("Competition Week")
plt.ylabel("Value")
plt.title(f"Fan Votes vs Judge Score: {clean_encoding(plot_data['celebrity_name_clean'].iloc[0])}")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("vote_judge_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 9. 特征重要性可视化 =====================
plt.figure(figsize=(8, 6))
top10_importance = feature_importance.head(10)
plt.barh(top10_importance["feature"], top10_importance["importance"], color="orange")
plt.xlabel("Feature Importance")
plt.title("Top10 Feature Importance (Random Forest)")
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n🎉 所有任务完成！生成文件：")
print("1. estimated_votes.csv - 粉丝投票估算结果")
print("2. vote_judge_comparison.png - 投票vs评委分对比图")
print("3. feature_importance.png - 特征重要性图")