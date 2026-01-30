
import pandas as pd

# 基础参数设置
TOTAL_MASS = 1e8  # 1亿公吨材料 [cite: 68]
SES_YEARLY_CAP = 3 * 179000  # 3个港口的总运力 
ROCKET_PAYLOAD = 125  # 单次火箭载荷 (100-150吨均值) [cite: 76]

# 2050年成本预估 (美元/公吨)
COST_SES = 100000    # 约 $100/kg
COST_ROCKET = 500000 # 约 $500/kg
INFRA_COST_SES = 200e9 # 电梯建设固定投入

def analyze_scenario_c(years):
    # 计算总运力缺口
    total_needed_per_year = TOTAL_MASS / years
    
    # 太空电梯全力运行
    ses_annual = min(SES_YEARLY_CAP, total_needed_per_year)
    # 剩余部分由火箭承担
    rocket_annual = max(0, total_needed_per_year - ses_annual)
    
    # 成本计算
    cost = (ses_annual * years * COST_SES) + (rocket_annual * years * COST_ROCKET) + INFRA_COST_SES
    
    # 物流压力计算
    daily_launches = rocket_annual / ROCKET_PAYLOAD / 365
    
    return {
        "Duration": f"{years} Years",
        "Total Cost (T$)": round(cost / 1e12, 2),
        "Rocket Share (%)": round((rocket_annual / total_needed_per_year) * 100, 1),
        "Daily Launches (Global)": round(daily_launches, 1),
        "Launch per Site (10)": round(daily_launches / 10, 2)
    }

# 评估不同时间跨度
results = [analyze_scenario_c(y) for y in [30, 50, 75, 100, 150]]
df = pd.DataFrame(results)
print(df)
