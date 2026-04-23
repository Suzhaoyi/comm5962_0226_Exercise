import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score

# 1. 加载数据
df = pd.read_csv('cleaned_mental_health_data.csv')

# 2. 目标变量处理
df['is_high_risk'] = df['mental_health_risk'].apply(lambda x: 1 if x >= 1 else 0)

# 3. 自适应特征选择 - 自动匹配你数据集中的列名
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

# 尝试匹配你报错的那几个列
col_sleep = find_col(['sleep_hours', 'sleep_duration'])
col_work_hours = find_col(['work_hours_per_week', 'weekly_working_hours', 'working_hours'])
col_sat = find_col(['job_satisfaction', 'occupational_satisfaction'])
col_stress = find_col(['work_stress_level', 'stress_level'])

# 组建特征列表 (剔除没找到的)
features = [col for col in [col_sleep, col_work_hours, col_sat, col_stress] if col is not None]
print(f"模型将使用以下特征进行训练: {features}")

X = df[features]
y = df['is_high_risk']

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练随机森林 (使用 class_weight='balanced' 来强行提升 Recall)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced' # 核心步骤：平衡权重以提高对风险组的敏感度
)
rf_model.fit(X_train, y_train)

# 6. 输出 Recall 结果
rf_preds = rf_model.predict(X_test)
print("\n--- Random Forest Performance ---")
print(f"Recall (召回率): {recall_score(y_test, rf_preds):.2f}")
print("\nDetailed Report:")
print(classification_report(y_test, rf_preds))

# 7. 绘制特征重要性图
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=features)
feat_importances.sort_values().plot(kind='barh', color='salmon')
plt.title('Critical Factors Predicting Mental Health Risk')
plt.show()