# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


#import numpy as np

df_priceData = pd.read_csv("priceHistoryy.csv")
df_priceData = df_priceData.iloc[0:480,:]
df_priceData = df_priceData.T
num_pc = 2

[row,col] = df_priceData.shape

stocks = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14']
    

stock_returns = []
for i in range(len(stocks)):
    priceList = df_priceData.loc[stocks[i]]
    stock_returns.append(priceList.pct_change(1))
    
stock_returns = np.asarray(stock_returns)
stock_returns = stock_returns[:,1:]
[n,m] = stock_returns.shape

print('The number of timestamps is {}.'.format(m))
print('The number of stocks is {}.'.format(n))

pca = PCA(n_components=num_pc) # number of principal components
pca.fit(stock_returns)

percentage =  pca.explained_variance_ratio_
percentage_cum = np.cumsum(percentage)

print ('{0:.2f}% of the variance is explained by the first 5 PCs'.format(percentage_cum[-1]*100))

pca_components = pca.components_

W_i = []
W_i.append(np.array([0.0]*966))
for i in range(1,6):
    Z = np.random.standard_normal(size = 966)
    W_t = Z * np.sqrt(i)
    W_i.append(W_t)

dW_i = []
for i in range(5):
    dW_i.append(np.array(W_i[i+1]-W_i[i]))
    
dW_i = np.asarray(dW_i)

dY_k = np.diff(stock_returns)

#dW_i = dW_i.T

p1 = np.linalg.inv(np.dot(dW_i,np.transpose(dW_i)))
p2 = np.dot(dW_i, np.transpose(dY_k))

sigma_ki = np.dot(p1,p2)

sigma_ki = np.transpose(sigma_ki)




 
#    for j in range(len(priceList) - 1):
#        returns.append((self.price_data[i+1]/self.price_data[i])-1)

