# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:01:11 2016

@author: Issac
"""

from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd


# 画时序图，观察平稳性
def ts_plot(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号
    data.plot()
    plt.show()

    
# 平稳性检验 
def stationarityTest(data):
    diff = 0
    adf_data = ADF(data)
    while adf_data[1] >= 0.05:
        diff += 1
        adf_data = ADF(data.diff(diff).dropna())
    return (diff,adf_data[1])

    
# 白噪声检验    
def whitenoiseTest(data,lagnum=1):
    lb,p = acorr_ljungbox(data,lags=1)
    h = (p<0.05).sum()      # p < 0.05 是非白噪声
    if h>0:
        return False     # 序列为非白噪声序列    
    else:
        return True    # 序列为白噪声序列 

        
# 模型识别        
def find_optimal_model(data,pmax,qmax,d):
    kwargs={'warn_convergence':False}   # 关闭收敛警告
    bic_matrix = []
    for p in range(pmax):
        tmp = []
        for q in range(qmax):
            try:
                tmp.append(ARIMA(data, (p,d,q)).fit(**kwargs).bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p,q = bic_matrix.stack().idxmin()
    return p,q


# 模型检验，检验其残差序列是否为白噪声    
def arimaModelCheck(data,p,q,d):
    arima = ARIMA(data,(p,d,q)).fit()    # 建立并训练模型
    return arima
