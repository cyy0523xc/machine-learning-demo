# -*- coding: utf-8 -*-

# 一元线性回归模型
# Author: Alex
# Created Time: 2017年01月07日 星期六 10时31分46秒

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 房屋size与price历史数据(csv文件)
csv_file = './data01.csv'

# 读入dataframe
df = pd.read_csv(csv_file)
print(df)

# 定义数据
X = df[["size"]]
y = df["price"]
"""
array([[ 70],
       [ 81],
       [ 85],
       [ 93],
       [ 99],
       [105],
       [111]])
"""

# 建立线性回归模型
lr = linear_model.LinearRegression()

# 拟合
# 注意此处.reshape(-1, 1)，因为X是一维的！
lr.fit(X, y)

# MSE
y_pred = lr.predict(X=X)
print("MSE：")
print(mean_squared_error(y, y_pred))

# 不难得到直线的斜率、截距
a, b = lr.coef_, lr.intercept_

# 给出待预测面积
area = 140

# 方式1：根据直线方程计算的价格
print("直接计算的值：")
print(a * area + b)

# 方式2：根据predict方法预测的价格
print("预测得到的值：")
print(lr.predict(area))

# 画图
# 1.真实的点
plt.scatter(df['size'], df['price'], color='blue')

# 2.拟合的直线
plt.plot(df['size'], y_pred, color='red', linewidth=4)

plt.show()
