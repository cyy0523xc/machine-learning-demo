# -*- coding: utf-8 -*-

# 房屋面积与价格的线性模型
# Author: Alex
# Created Time: 2017年01月07日 星期六 10时31分46秒

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 房屋size, 房间数num与price历史数据(csv文件)
csv_file = './data02.csv'

# 读入dataframe
df = pd.read_csv(csv_file)
print(df)

# 整理数据
feature_cols = ["size", "num"]
X = df[feature_cols]
y = df["price"]

# 建立线性回归模型
lr = linear_model.LinearRegression()

# 拟合
lr.fit(X, y)

# 不难得到直线的斜率、截距
a, b = lr.coef_, lr.intercept_
print("a, b:")
print(a, b)

# 误差计算
y_pred = lr.predict(X=X)
MAE = mean_absolute_error(y, y_pred)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
print("MAE, MSE, RMSE：")
print(MAE, MSE, RMSE)
print("MSE：")

# 探索各个特征的线性关系
# 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带
sns.pairplot(df, x_vars=feature_cols, y_vars='price',
             size=7, aspect=0.8, kind='reg')
plt.show()

# 画图
# 误差的曲线图
plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
plt.plot(range(len(y_pred)), y, 'r', label="test")
plt.legend(loc="upper left")  # 图中标签的位置
plt.xlabel("the number of point")
plt.ylabel('value of price')
plt.show()
