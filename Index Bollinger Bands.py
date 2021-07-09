#!/usr/bin/env python
# coding: utf-8

# In[713]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# In[714]:


us_idx = pd.read_csv('US_indices.csv').T
us_idx.columns = us_idx.iloc[0]
us_idx = us_idx.iloc[1:]
us_idx 

eu_idx = pd.read_csv('EU_indices.csv').T
eu_idx.columns = eu_idx.iloc[0]
eu_idx = eu_idx.iloc[1:]
eu_idx 


# In[715]:


def real_index_price(index_df):
    idx = index_df.copy()
    idx = (index_df + 1).cumprod()
    return idx


# In[716]:


def real_index(index):
    adj_idx = pd.DataFrame()
    for idx in index:
        adj = real_index_price(index[idx])
        adj_idx = pd.concat([adj_idx, adj], axis = 1, join = 'outer')
    return adj_idx
    


# In[717]:


adj_us_idx = real_index(us_idx)
adj_eu_idx = real_index(eu_idx)
all_indices = pd.concat([adj_us_idx, adj_eu_idx], axis = 1, join = 'outer')
all_indices


# In[718]:


all_indices.to_csv("Index_Prices.csv")


# In[719]:



def build_bands_index(data, width = 2, lookback = 20): #builds the bollinger bands for every stock in index
    def build_bands(data, name, width, lookback): #gets bollinger bands value for single stock
        # width = 2
        # lookback = 20
        df = pd.DataFrame()
        df[name] = data[name]
        df[name + ' SMA'] = df[name].rolling(lookback).mean()
        df[name + ' UB'] = df[name + ' SMA'] + (df[name].rolling(lookback).std() * width) #upper band
        df[name + ' LB'] = df[name + ' SMA'] - (df[name].rolling(lookback).std() * width) #lower band
       # df[df[name] > df[name + 'UB']].astype(int) * -1 + df[df[name] < df[name + 'LB']].astype(int) 
        df[name + ' B%'] = (df[name] - df[name + ' LB']) / (df[name + ' UB'] - df[name + ' LB'] )
        return df

    df = pd.DataFrame()
    for name in data:
        index_band = build_bands(data, name, width, lookback)
        df = pd.concat([df, index_band], axis = 1, join = 'outer')
    return df

def get_index(data, index): #returns all columns with specific index
    return data.filter(regex = index)

def get_B_df(data): #B% vals columns only
    return data.filter(regex = "B%").fillna(0.5)

 


# In[720]:


index_bands = build_bands_index(all_indices)
b_percents = get_B_df(index_bands)


# In[ ]:





# In[721]:


def sort_B_pct(b_df): #returns list in ascending order of B% columns
    arr = np.argsort(b_df.values, axis=1)
    df1 = pd.DataFrame(b_df.columns[arr], index=b_df.index)
    return df1

def get_top_bot_five(data):
    b_vals = get_B_df(data)
    sorted_df = sort_B_pct(b_vals)
    
    top_five = sorted_df.iloc[:,-5:]
    bot_five = sorted_df.iloc[:,:5]
    df = pd.concat([bot_five, top_five], axis = 1, join = 'outer')
    df.columns = ['L1', 'L2', 'L3', 'L4', 'L5', 'T5', 'T4', 'T3', 'T2', 'T1']
    return df


# In[ ]:





# In[722]:


top_ten = get_top_bot_five(index_bands)


# In[723]:


def sector_bollinger_day(data, date):
    sectors = np.array(get_top_bot_five(data).loc[date].T)

    top_five_B = get_B_df(data).loc[date].nlargest(5)
    bot_five_B = get_B_df(data).loc[date].nsmallest(5)

    top_bot_B = np.array(pd.concat([bot_five_B, top_five_B.iloc[::-1]]))
    
    allocation = [.17,.13,.10,.07,.03,-.03,-.07,-.10,-.13,-.17]

    long_short = np.where(top_bot_B > 0.5, 'Short', 'Long')
    df = pd.DataFrame({'Sectors': sectors,
                       'B%': top_bot_B,
                       'Long/Short': long_short,
                       'Allocation': allocation})
    return df


# In[724]:


sector_bollinger_day(index_bands, '01/01/2010')


# In[725]:


def get_B_vals(data, arr, date):
    return data[arr].loc[date]

def dic_val(arr, date, long_bool): #array of indexes, date, long or short
    long_idx = arr
    long_B = list(get_B_vals(index_bands, long_idx, date))
    if(long_bool):
        alloc = [.17,.13,.10,.07,.03]
        return list(zip(long_idx, long_B, alloc))
    else:
        alloc = [-.03,-.07,-.10,-.13,-.17]
    return list(zip(long_idx, long_B, alloc))

    


# In[ ]:





# In[726]:


def rebalance_day(data, date):
    overall_long_idx = list(sector_bollinger_day(data, date)['Sectors'].iloc[:5])
    overall_short_idx = list(sector_bollinger_day(data, date)['Sectors'].iloc[-5:])
    return dic_val(overall_long_idx, date, True) + dic_val(overall_short_idx, date, False)

def replace_sectors(init_arr, n_mask, n_replace_vals):#make sure replace vals are ordered in the way we want them to
    arr = init_arr.reset_index(drop=True)
    mask = n_mask.reset_index(drop=True)
    replace_vals = n_replace_vals.reset_index(drop=True)
    no_dups_replace_arr = replace_vals[~replace_vals.isin(arr[~mask])]
    arr[mask] = list(no_dups_replace_arr[:mask[mask].shape[0]])
    return arr


# In[727]:


def construct_port(data, rebalance_freq, start_T):
    b_percents = get_B_df(data)
    b_percents_new = b_percents.loc[start_T:].iloc[1:]

    start_long_idx = list(sector_bollinger_day(data, start_T)['Sectors'].iloc[:5])
    start_short_idx = list(sector_bollinger_day(data, start_T)['Sectors'].iloc[-5:])
    
    port = {start_T: dic_val(start_long_idx, start_T, True) + dic_val(start_short_idx, start_T, False)}
    sorted_sectors = sort_B_pct(b_percents)
    yester_date = start_T
    
    freq_counter = 1
    for date, row in b_percents_new.iterrows():
        if(freq_counter % rebalance_freq == 0):
            port[date] = rebalance_day(data, date)
            
        else:   
            curr_vals = port[yester_date]

            curr_idxs, curr_Bs, curr_allocs = zip(*curr_vals)
            curr_idxs_ls = pd.Series(list(curr_idxs))

            long_idxs = curr_idxs_ls[:5]
            short_idxs = curr_idxs_ls[-5:]
            short_idxs = short_idxs[::-1]

            long_mask = (row[long_idxs] >= 0.5).reset_index(drop=True)
            short_mask = (row[short_idxs] <= 0.5).reset_index(drop=True)
            
            sorted_sectors_date = sorted_sectors.loc[date]
            replace_long_sectors = sorted_sectors_date[:10]
            replace_short_sectors = sorted_sectors_date[-10:]
            replace_short_sectors = replace_short_sectors[::-1]
            #long replacement

            new_long_idxs = replace_sectors(long_idxs, long_mask, replace_long_sectors)
            new_short_idxs = replace_sectors(short_idxs, short_mask, replace_short_sectors)[::-1]

            port[date] = dic_val(new_long_idxs, date, True) + dic_val(new_short_idxs, date, False)

        yester_date = date
        freq_counter = freq_counter + 1
    return port
    
def construct_port_no_reb(data, start_T):
    b_percents = get_B_df(data)
    b_percents_new = b_percents.loc[start_T:].iloc[1:]

    start_long_idx = list(sector_bollinger_day(data, start_T)['Sectors'].iloc[:5])
    start_short_idx = list(sector_bollinger_day(data, start_T)['Sectors'].iloc[-5:])
    
    port = {start_T: dic_val(start_long_idx, start_T, True) + dic_val(start_short_idx, start_T, False)}
    sorted_sectors = sort_B_pct(b_percents)
    yester_date = start_T
    
    freq_counter = 1
    for date, row in b_percents_new.iterrows():

        curr_vals = port[yester_date]

        curr_idxs, curr_Bs, curr_allocs = zip(*curr_vals)
        curr_idxs_ls = pd.Series(list(curr_idxs))

        long_idxs = curr_idxs_ls[:5]
        short_idxs = curr_idxs_ls[-5:]
        short_idxs = short_idxs[::-1]

        long_mask = (row[long_idxs] >= 0.5).reset_index(drop=True)
        short_mask = (row[short_idxs] <= 0.5).reset_index(drop=True)

        sorted_sectors_date = sorted_sectors.loc[date]
        replace_long_sectors = sorted_sectors_date[:10]
        replace_short_sectors = sorted_sectors_date[-10:]
        replace_short_sectors = replace_short_sectors[::-1]
        #long replacement

        new_long_idxs = replace_sectors(long_idxs, long_mask, replace_long_sectors)
        new_short_idxs = replace_sectors(short_idxs, short_mask, replace_short_sectors)[::-1]

        port[date] = dic_val(new_long_idxs, date, True) + dic_val(new_short_idxs, date, False)

        yester_date = date
        freq_counter = freq_counter + 1
    return port
    
get_ipython().run_line_magic('time', '')


# In[728]:


get_ipython().run_cell_magic('time', '', "port = construct_port(index_bands, 7, '01/01/2010') #weekly rebalance\nport")


# In[729]:


port2 = construct_port_no_reb(index_bands, '01/01/2010')
port2


# In[730]:


def get_day_pos_port(port, date):
    df = pd.DataFrame(port[date])
    df.columns = ['Sectors', 'B%', 'Allocation']
    return df


# In[731]:


example = get_day_pos_port(port, '01/01/2021')


# In[732]:


port_df = pd.DataFrame(port).T
port_df.to_csv("Bands_Portfolio_Wkly_Rebalance.csv")


# In[733]:


port2_df = pd.DataFrame(port2).T
port2_df.to_csv("Bands_Portfolio_No_Rebalance.csv")


# In[734]:


new_port = {}
for k,v in port.items():
    new_port[k] = [t[i] for t in v for i in [0,2]]
alloc_df = pd.DataFrame(new_port).T
alloc_df.columns = [x for i in range(1,11) for x in ['Sector ' + str(i), 'Allocation ' + str(i)]]
test1 = alloc_df[['Sector 1', 'Allocation 1']]

pos_df = pd.DataFrame() 

for i in range(1,11):
    sec_name = 'Sector ' + str(i)
    alloc_name = 'Allocation ' + str(i)
    sectori = alloc_df[[sec_name, alloc_name]].pivot(columns = sec_name, values = alloc_name).fillna(0)
    pos_df = sectori.add(pos_df, fill_value = 0)

index_prices = all_indices
pos_df = pd.concat([index_prices, pos_df], axis = 1, join = 'outer').sort_index(axis = 1).fillna(0)
idx_cols = all_indices.columns

for idx in idx_cols:
    pos_df[idx + '|P&L'] = pos_df[idx].diff(periods = -1) * -1 / pos_df[idx] * pos_df[idx + ' B%'] 
    #(tomorrows price - todays price) / todays price * our position allocation today = P&L for the day
    
pos_df['Total P&L'] = pos_df.filter(regex = 'P&L').sum(axis = 1)
pos_df['Cumulative P&L'] = pos_df['Total P&L'].cumsum()


# In[735]:


pos_df.index = pd.to_datetime(pos_df.index)
pos_df


# In[ ]:





# In[736]:


fig = plt.figure(figsize=(50,30)) #overall plot size
ax4 = fig.add_subplot(324)


ax4.plot(pos_df.loc['01/01/2010':]['Total P&L'].cumsum()) #note: we use cumsum here to see *cumulative* profit
ax4.set_title("Cumulative P&L")

plt.xlabel("Time")
plt.show()


# In[ ]:





# In[ ]:




