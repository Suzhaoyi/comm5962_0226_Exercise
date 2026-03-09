# 导入库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 1. 读取数据
df = pd.read_csv("diamonds.csv")   # 修改为你本地的文件路径
print("数据预览：")
print(df.head())

# 2. 数据预处理
# 将分类变量转为哑变量
df_encoded = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)

# 3. 定义因变量和自变量（去掉 x, y, z）
X = df_encoded.drop(["price", "x", "y", "z"], axis=1)   # 自变量
y = df_encoded["price"]                                # 因变量

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 用 sklearn 建模
model = LinearRegression()
model.fit(X_train, y_train)

print("\nSklearn 回归结果：")
print("训练集 R²:", model.score(X_train, y_train))
print("测试集 R²:", model.score(X_test, y_test))

# 6. 用 statsmodels 查看详细结果
X_train_sm = sm.add_constant(X_train)  # 添加常数项
ols_model = sm.OLS(y_train.astype(float), X_train_sm.astype(float)).fit()

print("\nStatsmodels 回归结果：")
print(ols_model.summary())