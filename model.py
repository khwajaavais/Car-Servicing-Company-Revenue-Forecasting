


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from time import gmtime, strftime 
from pylab import rcParams
import statsmodels.api as sm
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from pandas.tseries.offsets import DateOffset





invoice = pd.read_csv("Final_invoice.csv",low_memory=False)





invoice['Month-Year'] = pd.to_datetime(invoice['Invoice Date']).dt.strftime('20%y-%m')


# In[4]:


revenue_model = invoice.groupby(['Model','Month-Year']).agg({'Total Amt Wtd Tax.':['mean','count']})


# In[5]:





# In[6]:


revenue_model.columns = revenue_model.columns.droplevel()


# In[7]:


revenue_model['Revenue'] = revenue_model['mean'] * revenue_model['count']


# In[8]:





# In[9]:


model_data = revenue_model.round(1).pivot_table(index = ['Month-Year'], columns = ['Model'], values = ['Revenue'], fill_value=0, aggfunc='sum').reset_index()


# In[10]:





# In[11]:


model_data.columns = model_data.columns.droplevel()


# In[12]:


model_data.rename(columns={'':'Month-Year'},inplace=True)


# In[13]:





# In[14]:


model_data['Total_Revenue'] =  model_data.iloc[:,1:].sum(axis=1)


# In[15]:





# In[16]:


#model_data.to_csv('model data.csv')


# In[17]:


Time_Series_data = model_data[['Total_Revenue','Month-Year']].sort_values(by = 'Month-Year')
Time_Series_data['Month-Year'] = pd.to_datetime(Time_Series_data['Month-Year'])
Time_Series_data.set_index('Month-Year',inplace=True)


# In[18]:





# In[19]:


result = sm.tsa.seasonal_decompose(Time_Series_data['Total_Revenue'], model='multiplicative')


# In[20]:





# In[21]:


x =  Time_Series_data['Total_Revenue'].values


# In[27]:


#p=d=q=range(0,9)


# In[28]:


#pdq=list(product(p,d,q))


# In[29]:


#seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]


# In[30]:


sarima = SARIMAX(x,order=(0,0,0),seasonal_order = (0, 7, 0, 12))
sarima_fit = sarima.fit()


# In[ ]:





# In[ ]:


data = int(input(" Enter the Number of Months: "))

dates = [Time_Series_data["Total_Revenue"].index[-1] + DateOffset(months=x) for x in range(0,data) ]
f_date = pd.DataFrame(index=dates[1:],columns = Time_Series_data.columns)
dataset = pd.concat([Time_Series_data,f_date])
prediction_sArima = sarima_fit.predict(start = 55, end = dataset.shape[0])
dataset = dataset.drop(index = dataset.index,inplace = True)

print(prediction_sArima.astype(np.int64))


# In[ ]:
#data = int(input(" Enter the Number of Month: "))
#Year = int(input(" Enter the Number of Year"))
#dates = [Time_Series_data["Total_Revenue"].index[-1] + DateOffset(month=x) for x in range(0,data) ]
#Years = [Time_Series_data["Total_Revenue"].index[-1] + DateOffset(Year=x) for x in range(0,Year)]
#f_date = pd.DataFrame(index=dates[1:],columns = Time_Series_data.columns)
#dataset = pd.concat([Time_Series_data,f_date])
#prediction_sArima = sarima_fit.predict(start = 55, end = dataset.shape[0])
#dataset = dataset.drop(index = dataset.index,inplace = True)

#print(prediction_sArima.astype(np.int64))



import pickle
# Open a file
file = open('model.pkl', 'wb')

# dump information to that file
pickle.dump(prediction_sArima,file)




