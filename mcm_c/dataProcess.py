import pandas as pd
import numpy as np
import re


def preprocess_dwts_data(file_path):
    """
    读取并清洗 DWTS 数据集，将其转换为适合建模的长格式。
    """
    # 1. 读取数据
    df = pd.read_csv(file_path)

    # 2. 提取"淘汰周次"信息 (解析 results 列)
    # 逻辑：从 'Eliminated Week 4' 中提取数字 4。
    # 对于 '1st Place', '2nd Place', 'Withdrew' 等特殊情况，暂时标记为 -1 或 NaN
    def extract_elimination_week(result_str):
        if pd.isna(result_str):
            return np.nan
        match = re.search(r'Week (\d+)', str(result_str), re.IGNORECASE)
        if match:
            return int(match.group(1))
        return np.nan  # 对于冠军、亚军或退赛，不适用常规"淘汰周"逻辑

    df['eliminated_week_num'] = df['results'].apply(extract_elimination_week)

    # 3. 数据重塑 (Wide to Long)
    # 将 week1_judge1_score, week1_judge2_score... 转换为行
    # 这里的 id_vars 是我们保留的静态列
    id_vars = [
        'celebrity_name', 'ballroom_partner', 'celebrity_industry',
        'celebrity_age_during_season', 'season', 'results',
        'placement', 'eliminated_week_num'
    ]

    # 使用 melt 将宽表变长表
    df_long = df.melt(id_vars=id_vars, var_name='week_judge_str', value_name='score')

    # 4. 解析 'week_judge_str' 列
    # 格式如: week1_judge1_score
    # 我们需要提取 week_num
    pattern = r'week(\d+)_judge(\d+)_score'
    extracted = df_long['week_judge_str'].str.extract(pattern)
    df_long['week'] = extracted[0].astype(float)
    df_long['judge'] = extracted[1].astype(float)

    # 5. 清洗分数
    # 去除 score 为 NaN 的行 (表示该周该评委不存在)
    df_long = df_long.dropna(subset=['score', 'week'])
    df_long['week'] = df_long['week'].astype(int)

    # 关键处理：题目说明 "A 0 score is recorded for celebrities who are eliminated"
    # 因此，我们只保留 score > 0 的行，代表该选手通过跳舞获得了分数
    # 注意：如果某选手当周确实跳了但得了0分（极不可能），这里会被误删。但在DWTS中最低分通常是1分。
    # 只有被淘汰后的 0 分才应该被剔除。
    df_active = df_long[df_long['score'] > 0].copy()

    # 6. 聚合：计算每周的总评委分
    # 按照 赛季-周-选手 聚合，避免重复列名
    groupby_cols = ['season', 'week', 'celebrity_name']
    # 移除已经在分组键中的列，避免重复
    remaining_id_vars = [col for col in id_vars if col not in groupby_cols]

    df_weekly = df_active.groupby(groupby_cols + remaining_id_vars, as_index=False).agg(
        total_judge_score=('score', 'sum'),
        judge_count=('judge', 'count')  # 记录有几个评委打分
    )

    # 7. 标记"本周被淘汰者" (Target Variable)
    # 如果当前周 == 该选手记录的 eliminated_week_num，则标记为 True
    df_weekly['is_eliminated'] = (df_weekly['week'] == df_weekly['eliminated_week_num'])

    # 8. 处理决赛选手 (1st, 2nd, 3rd Place)
    # 他们的 results 字段不是 'Eliminated Week X'。
    # 我们需要根据每一赛季的最后一周来标记。
    # 计算每个赛季的最大周数
    season_max_week = df_weekly.groupby('season')['week'].max().reset_index()
    season_max_week.columns = ['season', 'max_week']

    df_weekly = pd.merge(df_weekly, season_max_week, on='season', how='left')

    # 如果是最后一周，且没有被标记为淘汰，通常意味着他们是决赛选手
    # 题目要求我们关注的是"淘汰机制"，决赛周的淘汰规则可能不同（直接按排名）
    df_weekly['is_final'] = (df_weekly['week'] == df_weekly['max_week'])

    # 9. 排序与整理
    df_weekly = df_weekly.sort_values(by=['season', 'week', 'total_judge_score'], ascending=[True, True, False])

    return df_weekly


# --- 使用示例 ---
file_path = '2026_MCM_Problem_C_Data.csv'
cleaned_data = preprocess_dwts_data(file_path)
print(cleaned_data.head())
cleaned_data.to_csv('cleaned_dwts_data.csv', index=False)
