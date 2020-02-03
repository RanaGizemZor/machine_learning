#!/usr/bin/env python
# coding: utf-8

# In[69]:


import matplotlib as plt 
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('Desktop/train.csv')
df


# In[70]:


x = df.select_dtypes(include=['object'])
x


# In[71]:


list(x.columns.values)


# In[72]:


a=df.drop(['MSZoning',
 'Street',
 'Alley',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'PoolQC',
 'Fence',
 'MiscFeature',
 'SaleType',
 'SaleCondition'], axis=1)


# In[73]:


a


# In[74]:


a["LotFrontage"].fillna(a["LotFrontage"].mean(), inplace=True)
a


# In[75]:


a = (a - a.mean())/a.std()
a.head()


# In[83]:


def hypothesis(x, theta):
 return np.dot(
  np.transpose(theta),
  x
)

def gradientDescent(x, y, theta, m, alpha, iterations=1500):
 for iteration in range(iterations):
    for j in range(len(theta)):
      gradient = 0
      for i in range(m):
          gradient += (hypothesis(x[i], theta) - y[i]) * x[i][j]
    gradient *= 1/m
    theta[j] = theta[j] -  (alpha * gradient)
    print(theta)
 return theta
print (theta)


# In[90]:


def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

