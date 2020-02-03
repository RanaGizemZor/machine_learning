#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import matplotlib as plt 
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df =pd.read_csv('Desktop/train.csv', nrows=1460)
df


# In[106]:


df.describe()


# In[39]:


import pandas as pd
import numpy as np

df["MiscVal"].fillna(df["MiscVal"].mean(), inplace=True)
df


# In[22]:


df.dtypes


# In[60]:


df.isnull().sum()


# In[108]:


import pandas as pd
import numpy as np

df["LotFrontage"].fillna(df["LotFrontage"].mean(), inplace=True)
df


# In[129]:


df["MasVnrArea"].fillna(df["MasVnrArea"].mean(), inplace=True)
df["LotFrontage"].fillna(df["LotFrontage"].mean(), inplace=True)
df["BsmtFinSF1"].fillna(df["BsmtFinSF1"].mean(), inplace=True)
df["BsmtFinSF2"].fillna(df["BsmtFinSF2"].mean(), inplace=True)
df["BsmtUnfSF"].fillna(df["BsmtUnfSF"].mean(), inplace=True)
df["TotalBsmtSF"].fillna(df["TotalBsmtSF"].mean(), inplace=True)
df["2ndFlrSF"].fillna(df["2ndFlrSF"].mean(), inplace=True)
df["1stFlrSF"].fillna(df["1stFlrSF"].mean(), inplace=True)
df["OpenPorchSF"].fillna(df["OpenPorchSF"].mean(), inplace=True)
df["MiscVal"].fillna(df["MiscVal"].mean(), inplace=True)
df["WoodDeckSF"].fillna(df["WoodDeckSF"].mean(), inplace=True)
df["EnclosedPorch"].fillna(df["EnclosedPorch"].mean(), inplace=True)
df["LotArea"].fillna(df["LotArea"].mean(), inplace=True)
df["MiscVal"].fillna(df["MiscVal"].mean(), inplace=True)

df


# In[49]:



df['MSZoning'].unique()
df['Street'].unique()
df['Alley'].unique()
df['LotShape'].unique()
df['LandContour'].unique()
df['Utilities'].unique()
df['LotConfig'].unique()
df['LandSlope'].unique()
df['Neighborhood'].unique()
df['Condition1'].unique()
df['Condition2'].unique()
df['BldgType'].unique()
df['HouseStyle'].unique()


# In[50]:


df.describe()


# In[115]:


df.isnull().sum().sum()


# In[51]:


import pandas as pd
import matplotlib as plt 
import numpy as np

df["LotFrontage"]= df["LotFrontage"].astype('float64') 
df


# In[52]:


df["LotArea"]= df["LotArea"].astype('float64') 
df


# In[53]:


df["MSSubClass"]= df["MSSubClass"].astype('float64') 
df


# In[54]:


df["OverallQual"]= df["OverallQual"].astype('float64') 
df


# In[55]:


df["OverallCond"]= df["OverallCond"].astype('float64') 
df


# In[56]:


df["MasVnrArea"]= df["MasVnrArea"].astype('float64') 
df


# In[57]:



df["BsmtFinSF1"]= df["BsmtFinSF1"].astype('float64') 
df


# In[65]:


df.fillna(df.mean())


# In[1]:


#gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost


g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)

finalCost = computeCost(X,y,g)
print(finalCost)


# In[ ]:




