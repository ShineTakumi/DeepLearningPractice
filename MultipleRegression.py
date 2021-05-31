from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# 糖尿病患者のデータセットをロード
dataset = datasets.load_diabetes()

# 説明変数
X = dataset.data
np.savetxt('X.csv',X,delimiter=',')
# 目的変数
y = dataset.target
np.savetxt('y.csv',y,delimiter=',')

# 学習用、テスト用にデータを分割（1:1）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 予測モデルを作成(重回帰)
clf = linear_model.LinearRegression()

# 学習
clf.fit(X_train, y_train)

# 回帰係数と切片の抽出
a = clf.coef_
b = clf.intercept_  

# 回帰係数
print("回帰係数:", a)
print("切片:", b) 
print("決定係数(学習用):", clf.score(X_train, y_train))
print("決定係数(テスト用):", clf.score(X_test, y_test))

for i in range(X.shape[1]):
    plt.plot(X[:, i], y, 'o')

plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.show()
