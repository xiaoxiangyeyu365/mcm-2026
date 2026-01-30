import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.exceptions import DataConversionWarning
import warnings
import joblib
import os

# 异常处理：忽略数据转换警告，捕获文件读取错误
warnings.filterwarnings('ignore', category=DataConversionWarning)


def estimate_fan_votes():
    try:
        # 1. 数据输入（已预处理，直接读取）
        data_path = 'cleaned_dwts_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件未找到：{data_path}")
        df = pd.read_csv(data_path)

        # 筛选有效数据：仅保留参赛周（avg_score>0）和有明确淘汰结果的记录
        df_valid = df[(df['avg_score'] > 0) & (df['results'].notna())].copy()
        print(f"有效训练数据量：{df_valid.shape[0]}")

        # 2. 特征矩阵构建
        # 标签编码：专业舞者（分类特征）
        le_partner = LabelEncoder()
        df_valid['partner_encoded'] = le_partner.fit_transform(df_valid['ballroom_partner'])

        # 选择特征：评委平均分、周次、年龄、专业舞者编码、行业One-Hot（已预处理）
        industry_cols = [col for col in df_valid.columns if col.startswith('ind_')]
        feature_cols = ['avg_score', 'week', 'celebrity_age_during_season', 'partner_encoded'] + industry_cols
        X = df_valid[feature_cols].fillna(0)  # 填充可能的NaN（行业未知）

        # 目标变量构建：通过淘汰结果反向映射粉丝投票等级（1-5级，1=最高）
        elimination_mapping = {
            '1st Place': 1, '2nd Place': 2, '3rd Place': 3,
            'Eliminated Week 10': 4, 'Eliminated Week 9': 4, 'Eliminated Week 8': 4,
            'Eliminated Week 7': 5, 'Eliminated Week 6': 5, 'Eliminated Week 5': 5,
            'Eliminated Week 4': 5, 'Eliminated Week 3': 5, 'Eliminated Week 2': 5, 'Eliminated Week 1': 5
        }
        df_valid['fan_vote_rank'] = df_valid['results'].map(elimination_mapping).fillna(3)  # 缺失结果默认3级
        y = df_valid['fan_vote_rank']

        # 3. 数据划分（避免数据泄露：按赛季划分，确保验证集赛季不包含在训练集中）
        seasons = df_valid['season'].unique()
        train_seasons = seasons[:int(len(seasons) * 0.8)]
        X_train = df_valid[df_valid['season'].isin(train_seasons)][feature_cols].fillna(0)
        y_train = df_valid[df_valid['season'].isin(train_seasons)]['fan_vote_rank']
        X_test = df_valid[~df_valid['season'].isin(train_seasons)][feature_cols].fillna(0)
        y_test = df_valid[~df_valid['season'].isin(train_seasons)]['fan_vote_rank']
        print(f"训练集样本数：{X_train.shape[0]}, 验证集样本数：{X_test.shape[0]}")

        # 4. 模型初始化（随机森林回归：集成模型泛化能力强，适配美赛大数据）
        rf_model = RandomForestRegressor(
            n_estimators=100,  # 决策树数量：平衡偏差与方差
            max_depth=10,  # 树深度：限制复杂度，避免过拟合
            min_samples_split=5,  # 最小分裂样本数：过滤噪声
            random_state=42,
            max_features='sqrt'  # 特征采样：提升泛化能力
        )

        # 5. 参数调优（网格搜索：美赛必做的精度优化步骤）
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [8, 10, 12],
            'min_samples_leaf': [2, 3, 5]
        }
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=5,  # 5折交叉验证：美赛标准验证方法
            scoring='r2',  # 评估指标：决定系数，反映模型解释力
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最优参数：{grid_search.best_params_}")

        # 6. 模型训练与评估
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"模型评估：MAE={mae:.3f}, R²={r2:.3f}")
        print("注：R²>0.6表示模型解释力良好，符合美赛精度要求")

        # 7. 结果预测：生成所有参赛选手的粉丝投票估计值
        df_valid['estimated_fan_votes'] = best_model.predict(X)
        # 归一化粉丝投票（0-100分）
        df_valid['estimated_fan_votes_norm'] = (df_valid['estimated_fan_votes'] - df_valid[
            'estimated_fan_votes'].min()) / \
                                               (df_valid['estimated_fan_votes'].max() - df_valid[
                                                   'estimated_fan_votes'].min()) * 100

        # 8. 模型保存（美赛可复用，避免重复训练）
        joblib.dump(best_model, 'processed_data/fan_vote_estimator.pkl')
        joblib.dump(le_partner, 'processed_data/partner_encoder.pkl')
        # 保存预测结果
        result_cols = ['season', 'celebrity_name', 'week', 'avg_score', 'results', 'estimated_fan_votes_norm']
        df_valid[result_cols].to_csv('processed_data/estimated_fan_votes.csv', index=False)
        print("粉丝投票估计结果已保存至：processed_data/estimated_fan_votes.csv")

        return best_model, df_valid[result_cols], mae, r2

    except FileNotFoundError as e:
        print(f"错误：{e}，请确保数据处理脚本已运行")
        return None, None, None, None
    except Exception as e:
        print(f"模型训练异常：{str(e)}")
        return None, None, None, None


# 运行模型
model, fan_vote_results, mae, r2 = estimate_fan_votes()