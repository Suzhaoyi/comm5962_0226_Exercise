import pandas as pd

# 1. 加载数据 (请确保 csv 文件在项目目录下)
df = pd.read_csv('mental_health_risk_dataset.csv')

# 2. 快速查看数据
print(df.info())
print(df.describe())

# 3. 目标变量重新标记 (将中/高风险合并为 1，低风险为 0)
df['is_high_risk'] = df['mental_health_risk'].apply(lambda x: 1 if x >= 1 else 0)

# 4. 特征处理：将分类变量转为哑变量 (Dummy Variables)
df_processed = pd.get_dummies(df, columns=['gender', 'employment_status'], drop_first=True)

# 5. 保存清洗后的数据供建模使用
df_processed.to_csv('cleaned_mental_health_data.csv', index=False)