#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#to select the train data and test data import the model_selection


advert = pd.read_csv('Desktop/train.csv', usecols=["LotArea","SalePrice"])

train, test= train_test_split(advert,test_size=0.30)
train,test

#split the test data


# In[38]:


import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#sale prediction
sales_pred = model.predict()

# Drawing the plot of the Lotarea and Saleprice in the train.csv
plt.figure(figsize=(16, 12))
plt.plot(advert['LotArea'], advert['SalePrice'], 'o')           # scatter plot showing actual data
plt.plot(advert['LotArea'], sales_pred, 'r', linewidth=3)   # regression line
plt.xlabel('LotArea ')
plt.ylabel('SalePrice')
plt.title('Lotarea/Saleprice')

plt.show()

#drawing the linear regression line 
model = smf.ols('SalePrice ~ LotArea', data=advert)
model = model.fit()
model.params




# In[44]:


#prediciton of actual y and y'

modpred = model.predict(test)
modpred

