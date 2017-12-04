import numpy as np
import json, codecs
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn import linear_model
reg = linear_model.LinearRegression()

filepath = './data/reg_data.mat'
data_mat = scipy.io.loadmat(filepath)
reg_data = np.array(data_mat["X"], np.float)
#reg_data = [x, y, t]
print(reg_data.shape)

#標準化
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

#データ間の距離
dt = 0.01 #sec
dx = 0.0204 #m
dy = 0.0204 #m

#データの勾配
f_t = np.gradient(reg_data, axis=2)/dt
f_x = np.gradient(reg_data, axis=1)/dx
f_y = np.gradient(reg_data, axis=0)/dy
f_xx = np.gradient(f_x, axis=1)/dx
f_yy = np.gradient(f_y, axis=0)/dy

#回帰分析できる形に変形
f = reg_data.flatten()
f_t_col = f_t.flatten()
f_x_col = f_x.flatten()
f_y_col = f_y.flatten()
f_xx_col = f_xx.flatten()
f_yy_col = f_yy.flatten()
partial = np.c_[f_xx_col, f_yy_col, f_x_col, f_y_col]

### 変化のない部分と外れ値をdeleteする
partial = np.delete(partial, np.where(np.absolute(f_t_col)<0.01) ,0)
f_t_col = np.delete(f_t_col, np.where(np.absolute(f_t_col)<0.01))
# partial = np.delete(partial, np.where(np.absolute(f_t_col)>1000) ,0)
# f_t_col = np.delete(f_t_col, np.where(np.absolute(f_t_col)>1000))

### 定義した偏微分方程式の残差プロットの表示
# plt.scatter(f_t_col,1e-3*(partial[:,0]+partial[:,1])-0.15*partial[:,2]-0.02*partial[:,3]-f_t_col, s=1, c='purple', marker='s', label='Residual error')
# plt.xlabel('f_t')
# plt.ylabel('Residual error')
# plt.savefig("./data/reg_mat0.png")
# plt.show()

#配列サイズ確認
# print('f',f.shape)
# print('f_t_col',f_t_col.shape)
# print('f_x_col',f_x_col.shape)
# print('f_y_col',f_y_col.shape)
# print('f_xx_col',f_xx_col.shape)
# print('f_yy_col',f_yy_col.shape)
print('partial',partial.shape)

#標準化
# partial = zscore(partial, axis=0)
# f_t_col = zscore(f_t_col, axis=0)

print('f_t_col ave:',np.average(f_t_col))
print('|f_t_col| ave:',np.average(np.absolute(f_t_col)))

#予測モデル
reg.fit(partial,f_t_col)

#回帰係数
print(pd.DataFrame({"Name":['f_xx', 'f_yy', 'f_x', 'f_y'],
                    "Coefficients":reg.coef_}).sort_values(by='Coefficients'))
print("切片",reg.intercept_)
print("R2",reg.score(partial, f_t_col))

### 残差プロット
X = reg.predict(partial)
plt.scatter(f_t_col, X - f_t_col, s=10, c='purple', marker='s', label='Residual error')
plt.hlines(y=0, xmin=-1000, xmax=1000, lw=2, color='red')
plt.xlabel('f_t')
plt.ylabel('Residual error')
plt.savefig("./data/reg_mat.png")
# plt.show()
