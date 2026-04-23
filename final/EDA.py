import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载原始数据
df = pd.read_csv('mental_health_risk_dataset.csv')

# --- 自动检查列名 (防止拼写错误) ---
print("你的数据集列名有：", df.columns.tolist())

# 2. 目标变量处理
# 假设你的风险列名为 'mental_health_risk'
df['is_high_risk'] = df['mental_health_risk'].apply(lambda x: 1 if x >= 1 else 0)

# 设置风格并解决之前的警告
sns.set_theme(style="whitegrid")
plt.figure(figsize=(18, 6))

# --- 图 1：压力水平 (Stress Level) ---
plt.subplot(1, 3, 1)
# 修正警告：显式指定 hue
sns.violinplot(x='is_high_risk', y='stress_level', hue='is_high_risk', data=df, palette='muted', legend=False)
plt.title('1. Stress Level Distribution')

# --- 图 2：睡眠时长 (Sleep Hours) ---
plt.subplot(1, 3, 2)
sns.boxplot(x='is_high_risk', y='sleep_hours', hue='is_high_risk', data=df, palette='pastel', legend=False)
plt.title('2. Sleep Hours vs Risk')

# --- 图 3：职业满意度 (Job Satisfaction) ---
# 如果 job_satisfaction 报错，这里换成数据集里实际存在的列，例如 'occupational_stress' 或 'work_life_balance'
# 我这里增加一个判断，如果 'job_satisfaction' 不在列里，就打印提醒
target_col = 'job_satisfaction'
if target_col not in df.columns:
    print(f"警告: 没找到 '{target_col}'，正在尝试自动匹配相似列...")
    # 尝试寻找包含 'satisfaction' 或 'work' 的列
    possible_cols = [c for c in df.columns if 'satisfaction' in c.lower() or 'work' in c.lower()]
    if possible_cols:
        target_col = possible_cols[0]
        print(f"已自动切换到列: {target_col}")

plt.subplot(1, 3, 3)
sat_risk_pct = df.groupby(target_col)['is_high_risk'].mean() * 100
sat_risk_pct.plot(kind='bar', color='teal', alpha=0.7)
plt.title(f'3. % of Risk by {target_col}')
plt.ylabel('Risk Percentage (%)')

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()