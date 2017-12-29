import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
reg = linear_model.LinearRegression()
from scipy import stats

filepath = './data/data_1D.csv'

data = np.loadtxt(filepath, delimiter=',')

print(data.shape)

#標準化
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

#データ間の距離
dt = 0.1 #sec
dx = 0.1 #m

#データの勾配
f_t = np.gradient(data, axis=0)/dt
f_tt = np.gradient(f_t, axis=0)/dt
f_ttt = np.gradient(f_tt, axis=0)/dt
f_tttt = np.gradient(f_ttt, axis=0)/dt

f_x = np.gradient(data, axis=1)/dx
f_xx = np.gradient(f_x, axis=1)/dx
f_xxx = np.gradient(f_xx, axis=1)/dx
f_xxxx = np.gradient(f_xxx, axis=1)/dx

#回帰分析できる形に変形
f = data.flatten()
f_t_col = f_t.flatten()
f_tt_col = f_tt.flatten()
f_ttt_col = f_ttt.flatten()
f_tttt_col = f_tttt.flatten()
f_x_col = f_x.flatten()
f_xx_col = f_xx.flatten()
f_xxx_col = f_xxx.flatten()
f_xxxx_col = f_xxxx.flatten()

#partial = np.c_[f_tt_col, f_ttt_col, f_tttt_col, f_x_col, f_xx_col, f_xxx_col, f_xxxx_col]
partial = np.c_[f_x_col, f_xx_col]

###変化のない部分と外れ値をdeleteする
partial = np.delete( partial, np.where(np.absolute(f)<0.05), 0 )
f_t_col = np.delete( f_t_col, np.where(np.absolute(f)<0.05) )
f = np.delete( f, np.where(np.absolute(f)<0.1), 0 )
partial = np.delete( partial, np.where(np.absolute(f)>0.95), 0 )
f_t_col = np.delete( f_t_col, np.where(np.absolute(f)>0.95) )
f = np.delete( f, np.where(np.absolute(f)>0.9), 0 )

print("partial_size:", partial.shape)
print("f_t_col_size:", f_t_col.shape)

#標準化
# partial = zscore(partial, axis=0)
# f_t_col = zscore(f_t_col, axis=0)

#目的関数の平均値
print('f_t_col ave:',np.average(f_t_col))
print('|f_t_col| ave:',np.average(np.absolute(f_t_col)))

#予測モデル
reg.fit(partial,f_t_col)

#回帰係数
#print(pd.DataFrame({"Name":['f_tt_col', 'f_ttt_col', 'f_tttt_col', 'f_x_col', 'f_xx_col', 'f_xxx_col', 'f_xxxx_col'], "Coefficients":np.abs(clf.coef_)}))
print(pd.DataFrame({"Name":['f_x_col', 'f_xx_col'], "Coefficients":reg.coef_}))
print("Segment",reg.intercept_)
print("R2",reg.score(partial, f_t_col))

##残差プロット
X = reg.predict(partial)
plt.scatter(f_t_col, X - f_t_col, s=10, c='purple', marker='s', label='Residual error')
plt.hlines(y=0, xmin=-0.1, xmax=4, lw=1, color='red')
plt.xlabel('f_t')
plt.ylabel('Residual error')
plt.savefig("./data/reg_1D.png")
plt.show()
