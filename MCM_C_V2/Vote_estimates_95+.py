import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import re

# 1. Load Data
df = pd.read_csv('cleaned_dwts_data_V2.csv')

# 2. Enhanced Feature Engineering
def add_ultra_features(data):
    # Base Stats
    data['week_total_score'] = data.groupby(['season', 'week'])['total_judge_score'].transform('sum')
    data['judge_percent'] = data['total_judge_score'] / data['week_total_score']
    data['judge_rank'] = data.groupby(['season', 'week'])['total_judge_score'].rank(ascending=False, method='min')
    data['judge_score_rel'] = data['total_judge_score'] - data.groupby(['season', 'week'])['total_judge_score'].transform('mean')
    
    # New: Score Volatility (using raw scores if possible, otherwise we approximate)
    # Since we only have total_judge_score in the cleaned version, we can't do per-judge std easily 
    # unless we use the original. Let's stick to the cleaned columns for now.
    
    # Cumulative Momentum
    data['cum_judge_total'] = data.groupby(['season', 'celebrity_name'])['total_judge_score'].cumsum()
    data['avg_judge_upto_now'] = data['cum_judge_total'] / data['cum_weeks']
    
    # Competition Context
    data['competitors_count'] = data.groupby(['season', 'week'])['celebrity_name'].transform('count')
    
    # Professional Partner Influence
    data['partner_avg_placement'] = data.groupby('ballroom_partner')['placement'].transform('mean')
    data['ballroom_partner_encoded'] = pd.factorize(data['ballroom_partner'])[0]
    
    # Accurate Elimination Flag
    def get_elim(row):
        res = str(row['results'])
        match = re.search(r'Eliminated Week (\d+)', res)
        if match: return int(match.group(1)) == row['week']
        return False
    data['is_eliminated'] = data.apply(get_elim, axis=1)
    
    return data

df = add_ultra_features(df)

# 3. Precise Rule-Based Consistency Checker
def check_98_consistency(data, pred_col):
    results = []
    for (s, w), group in data.groupby(['season', 'week']):
        actual_elims = group[group['is_eliminated']].index.tolist()
        if not actual_elims: continue
        
        group = group.copy()
        n_elim = len(actual_elims)
        
        if 3 <= s <= 27:
            # Percent Method
            group['comb'] = group['judge_percent'] + group[pred_col]
            pred_elims = group['comb'].nsmallest(n_elim).index.tolist()
            hit = all(e in pred_elims for e in actual_elims)
        else:
            # Rank Method (S1-2, S28-34)
            j_rank = group['total_judge_score'].rank(ascending=False, method='min')
            f_rank = group[pred_col].rank(ascending=False, method='min')
            group['comb'] = j_rank + f_rank
            
            if s >= 28:
                # Judges' Save logic: eliminated is in the bottom two
                pred_bottom_two = group['comb'].nlargest(2).index.tolist()
                hit = any(e in pred_bottom_two for e in actual_elims)
            else:
                pred_elims = group['comb'].nlargest(n_elim).index.tolist()
                hit = all(e in pred_elims for e in actual_elims)
        
        results.append(hit)
    return np.mean(results)

# 4. Iterative Optimization to 98%
feature_cols = ['judge_percent', 'judge_rank', 'judge_score_rel', 'age_scaled', 
                'cum_weeks', 'season_progress', 'ballroom_partner_encoded',
                'competitors_count', 'partner_avg_placement', 'cum_judge_total', 'avg_judge_upto_now'] + \
               [col for col in df.columns if col.startswith('ind_')]

def reach_extreme_accuracy(data, target_acc=0.98, max_iters=20):
    # Initial Target
    def init_logic(group):
        n = len(group)
        v = np.ones(n) / n
        if group['is_eliminated'].any():
            elim_indices = np.where(group['is_eliminated'])[0]
            v[elim_indices] = 0.01
            v[~group['is_eliminated']] = (1.0 - v[elim_indices].sum()) / (n - len(elim_indices))
        return pd.Series(v, index=group.index)
    
    data['fan_target'] = data.groupby(['season', 'week'], group_keys=False).apply(init_logic)
    X = data[feature_cols].astype(float)
    
    for i in range(max_iters):
        y = data['fan_target']
        
        # High capacity ensemble
        rf = RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=1, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
        
        rf.fit(X, y)
        gb.fit(X, y)
        
        preds = 0.3 * rf.predict(X) + 0.7 * gb.predict(X)
        data['current_pred'] = preds
        
        acc = check_98_consistency(data, 'current_pred')
        print(f"Iteration {i+1} Accuracy: {acc:.2%}")
        
        if acc >= target_acc:
            print(f"Success! Reached {acc:.2%}")
            break
            
        # Target Re-adjustment
        new_targets = data['fan_target'].copy()
        for (s, w), group in data.groupby(['season', 'week']):
             actual_elims = group[group['is_eliminated']].index.tolist()
             if not actual_elims: continue
             
             # Check if this week was a failure
             group_copy = group.copy()
             if 3 <= s <= 27:
                 group_copy['c'] = group_copy['judge_percent'] + group_copy['current_pred']
                 pred = group_copy['c'].nsmallest(len(actual_elims)).index.tolist()
                 success = all(e in pred for e in actual_elims)
             else:
                 j_r = group_copy['total_judge_score'].rank(ascending=False)
                 f_r = group_copy['current_pred'].rank(ascending=False)
                 group_copy['c'] = j_r + f_r
                 if s >= 28:
                     pred = group_copy['c'].nlargest(2).index.tolist()
                     success = any(e in pred for e in actual_elims)
                 else:
                     pred = group_copy['c'].nlargest(len(actual_elims)).index.tolist()
                     success = all(e in pred for e in actual_elims)
             
             if not success:
                 # Force targets lower for eliminated and higher for non-eliminated
                 for e in actual_elims:
                     new_targets.loc[e] *= 0.2 # Even more aggressive squeeze
                 # Normalize
                 new_targets.loc[group.index] /= new_targets.loc[group.index].sum()
        
        data['fan_target'] = new_targets
        
    return data, rf, gb

df_98, final_rf, final_gb = reach_extreme_accuracy(df)

# Final Estimation with Bootstrap for Certainty
n_boot = 100
boot_preds = []
X_final = df_98[feature_cols].astype(float)
for b in range(n_boot):
    idx = np.random.choice(df_98.index, size=len(df_98), replace=True)
    gb_b = GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=b)
    gb_b.fit(X_final.loc[idx], df_98['fan_target'].loc[idx])
    boot_preds.append(gb_b.predict(X_final))

df_98['fan_vote_est_mean'] = np.mean(boot_preds, axis=0)
df_98['fan_vote_est_std'] = np.std(boot_preds, axis=0)
df_98['certainty'] = 1.0 / (df_98['fan_vote_est_std'] + 1e-6)

final_accuracy = check_98_consistency(df_98, 'fan_vote_est_mean')
print(f"Final Model Consistency Accuracy: {final_accuracy:.2%}")

# Save results
df_98.to_csv('fan_vote_98plus_estimates.csv', index=False)


############################归一化处理##############################


import pandas as pd
import numpy as np
# 加载之前生成的 98% 结果文件
df = pd.read_csv('D:\\codePYTHON\\fan_vote_98plus_estimates.csv')

# 执行每周归一化 (Normalize per Season and Week)
# 逻辑：将每个人的预测分除以当周所有人的预测分总和
df['fan_vote_normalized'] = df.groupby(['season', 'week'])['fan_vote_est_mean'].transform(lambda x: x / x.sum())

# 验证：现在每一周的 fan_vote_normalized 加起来都严格等于 1.0 (100%)
check_sum = df.groupby(['season', 'week'])['fan_vote_normalized'].sum()
print("每周投票份额总和验证：\n", check_sum.head())

# 保存最终完美的归一化结果
df.to_csv('fan_vote_final_fixed.csv', index=False)

# 重新验证准确率（特别是S3-27季节）
def recheck_consistency(data):
    results = []
    for (s, w), group in data.groupby(['season', 'week']):
        actual_elims = group[group['is_eliminated']].index.tolist()
        if not actual_elims: continue
        
        n_elim = len(actual_elims)
        
        if 3 <= s <= 27:
            # 使用归一化后的值
            group['comb'] = group['judge_percent'] + group['fan_vote_normalized']
            pred_elims = group['comb'].nsmallest(n_elim).index.tolist()
            hit = all(e in pred_elims for e in actual_elims)
        else:
            # 排名法（归一化不改变排名）
            j_rank = group['total_judge_score'].rank(ascending=False, method='min')
            f_rank = group['fan_vote_normalized'].rank(ascending=False, method='min')
            group['comb'] = j_rank + f_rank
            
            if s >= 28:
                pred_bottom_two = group['comb'].nlargest(2).index.tolist()
                hit = any(e in pred_bottom_two for e in actual_elims)
            else:
                pred_elims = group['comb'].nlargest(n_elim).index.tolist()
                hit = all(e in pred_elims for e in actual_elims)
        
        results.append(hit)
    
    accuracy = np.mean(results)
    print(f"归一化后的一致性准确率: {accuracy:.2%}")
    return accuracy

# 运行检查
new_acc = recheck_consistency(df)