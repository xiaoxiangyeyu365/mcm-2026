import pandas as pd

# 1. 加载你归一化处理后的最终数据
# 确保路径指向你最新的结果文件
df_final = pd.read_csv('fan_vote_final_fixed.csv')

# 2. 提取核心列：赛季、周次、选手姓名、以及归一化后的投票份额
# 我们使用 'fan_vote_normalized' 作为最终的观众投票量指标
output_columns = [
    'season', 
    'week', 
    'celebrity_name', 
    'fan_vote_normalized', 
    'fan_vote_est_std',  # 保留标准差以体现不确定性
    'certainty'          # 保留确定性评分
]

# 3. 创建精简版 DataFrame
# 按照赛季和周次进行排序，确保输出文件逻辑清晰
weekly_fan_votes = df_final[output_columns].sort_values(by=['season', 'week', 'fan_vote_normalized'], ascending=[True, True, False])

# 4. 格式化输出：将投票份额转换为百分比（可选，为了更直观）
# 如果你希望保留 0-1 的小数，可以跳过下面这一行
# weekly_fan_votes['fan_vote_percentage'] = (weekly_fan_votes['fan_vote_normalized'] * 100).round(2).astype(str) + '%'

# 5. 输出文件
output_filename = 'celebrity_weekly_fan_votes.csv'
weekly_fan_votes.to_csv(output_filename, index=False)

print(f"成功导出！文件 '{output_filename}' 已生成。")
print(f"共包含 {len(weekly_fan_votes)} 条选手周次投票记录。")

# 预览前几行结果
print("\n前5行数据预览：")
print(weekly_fan_votes.head())