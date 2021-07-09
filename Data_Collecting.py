#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from sklearn import datasets


# In[33]:


us_idx = pd.read_csv('US_indices.csv').T
us_idx.columns = us_idx.iloc[0]
us_idx = us_idx.iloc[1:]
us_idx 
get_ipython().run_line_magic('store', 'us_idx')

eu_idx = pd.read_csv('EU_indices.csv').T
eu_idx.columns = eu_idx.iloc[0]
eu_idx = eu_idx.iloc[1:]
eu_idx 
get_ipython().run_line_magic('store', 'eu_idx')


# In[ ]:





# In[34]:


tickers = pd.read_csv('Unique_List_With BBG ticker.csv')
tickers 
get_ipython().run_line_magic('store', 'tickers')


# In[22]:


#CONCATING ALL THE CSV FILES INTO ONE
df = pd.DataFrame()

for yr in range(1999, 2022):
    str_yr = str(yr)

    p_file_path = "Price/" + str_yr + '.csv'
    price_df = pd.read_csv(p_file_path, header = 0, index_col = 0, parse_dates=True, skiprows=4)
    price_df = price_df.T
    
#     v_file_path = "Volume/"  + str_yr + '.csv'
#     volume_df = pd.read_csv(v_file_path, header = 0, index_col = 0, parse_dates=True, skiprows=4)
#     volume_df = volume_df.T

#     pv_df[name + ' Volume'] = volume_df

    df = pd.concat([df, price_df])
df


# In[37]:


all_stocks_df = df
all_stocks_df.to_csv("All_Stocks.csv")


# In[25]:





# In[ ]:





# In[ ]:




