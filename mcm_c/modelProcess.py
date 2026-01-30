import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib  # 用于保存模型

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def train_scoring_model(file_path):
    """
    加载清洗后的数据，训练随机森林回归模型，并输出评估结果。
    """
    print("--------------------------------------------------")
    print("Step 1: 数据输入与特征矩阵构建")
    print("--------------------------------------------------")

    try:
        df = pd.read_csv(file_path)
        print(f"数据加载成功，样本量: {df.shape[0]}, 特征数: {df.shape[1]}")
    except FileNotFoundError:
        print("错误：未找到文件，请检查路径。")
        return None, None, None

    # 1.1 特征选择 (X)
    # 排除非数值列和目标列。保留 'week', 'judge_count', 'age' 和所有 'ind_' 开头的职业特征
    feature_cols = ['week', 'judge_count', 'celebrity_age_during_season'] + \
                   [col for col in df.columns if col.startswith('ind_')]

    X = df[feature_cols]

    # 1.2 目标变量 (y)
    # 使用平均分作为预测目标，消除评委人数不同带来的偏差
    y = df['avg_score']

    # 检查并处理可能的缺失值 (尽管已预处理，但在建模前做最终防御)
    if X.isnull().sum().sum() > 0:
        print("警告：特征矩阵中发现缺失值，将填充为0或均值。")
        X = X.fillna(0)

    print(f"特征矩阵构建完成。特征维度: {X.shape}")

    # 1.3 数据集划分
    # random_state=42 保证结果可复现 (论文写作必须)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--------------------------------------------------")
    print("Step 2: 模型初始化与参数调优 (GridSearchCV)")
    print("--------------------------------------------------")

    # 初始化模型
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # 定义超参数网格
    # n_estimators: 树的数量，越多越稳但越慢
    # max_depth: 树的深度，限制深度防止过拟合
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    print("开始网格搜索 (Grid Search)... 这可能需要几秒钟...")
    # cv=5 表示5折交叉验证，是美赛中验证模型稳健性的标准操作
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳参数组合: {grid_search.best_params_}")

    print("\n--------------------------------------------------")
    print("Step 3: 模型训练与结果预测")
    print("--------------------------------------------------")

    # 使用最佳模型进行预测
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # 评估指标
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    print(f"测试集 MSE (均方误差): {mse:.4f}")
    print(f"测试集 R² (拟合优度): {r2:.4f} (越接近1越好)")
    print(f"测试集 MAE (平均绝对误差): {mae:.4f}")

    # 保存模型 (模拟实际工程操作)
    joblib.dump(best_model, 'dwts_rf_model.pkl')
    print("模型已保存为 'dwts_rf_model.pkl'")

    return best_model, X_train.columns, (y_test, y_pred_test)


# --- 执行主程序 ---
if __name__ == "__main__":
    # 这里的路径对应你上传的文件名
    file_path = 'cleaned_dwts_data.csv'
    model, feature_names, results = train_scoring_model(file_path)

    if model:
        # 简单打印一下最重要的3个特征，详细图表见可视化部分
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\n[关键结论] 影响评分的前三位因素:")
        for i in range(3):
            print(f"{i + 1}. {feature_names[indices[i]]} (权重: {importances[indices[i]]:.4f})")