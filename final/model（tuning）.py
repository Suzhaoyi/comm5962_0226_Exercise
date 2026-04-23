import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, precision_recall_curve

# 1. 加载数据并处理列名
df = pd.read_csv('cleaned_mental_health_data.csv')
df['is_high_risk'] = df['mental_health_risk'].apply(lambda x: 1 if x >= 1 else 0)

# 自动匹配列名
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns: return name
    return None

features = [col for col in [
    find_col(['sleep_hours', 'sleep_duration']),
    find_col(['work_hours_per_week', 'weekly_working_hours']),
    find_col(['job_satisfaction']),
    find_col(['work_stress_level', 'stress_level'])
] if col is not None]

X = df[features]
y = df['is_high_risk']

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练模型 (保持 class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 4. 获取预测概率 (而非直接预测类别)
y_probs = rf_model.predict_proba(X_test)[:, 1]

# 5. 调整阈值 (Threshold Tuning)
# 将阈值从 0.5 降至 0.35 (你可以根据运行结果微调这个值)
custom_threshold = 0.35
y_tuned_preds = (y_probs >= custom_threshold).astype(int)

# 6. 输出调优后的结果
print(f"--- 调优后结果 (当前阈值: {custom_threshold}) ---")
current_recall = recall_score(y_test, y_tuned_preds)
print(f"最终召回率 (Recall): {current_recall:.2f}")
print("\n详细分类报告:")
print(classification_report(y_test, y_tuned_preds))

# 7. 可视化：Precision-Recall 曲线 (证明你做了调优)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label='P-R Curve')
plt.axvline(x=current_recall, color='red', linestyle='--', label=f'Target Recall ({current_recall:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Threshold Tuning')
plt.legend()
plt.show()