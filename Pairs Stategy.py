#!/usr/bin/env python
# coding: utf-8

# In[664]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import scipy.stats as stats
from pandas.tseries.offsets import BDay
import random

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# In[6]:



prices_df = pd.read_csv('All_Stocks.csv')


# In[14]:


prices_df.index = prices_df.iloc[:,0]
prices_df = prices_df.iloc[:,1:]


# In[64]:


list_df1 = pd.read_csv('index_list1.csv')
list_df2 = pd.read_csv('index_list2.csv')
list_df = pd.concat([list_df1, list_df2], join = 'inner', axis = 0)
list_df.groupby('index').groups['S5COND']


# In[89]:


list_df


# In[204]:


prices_df


# In[282]:


isBusinessDay = BDay().onOffset
match_series = pd.to_datetime(prices_df.index).map(isBusinessDay)
prices_df = prices_df[match_series]


# In[283]:


prices_df


# In[469]:


get_ipython().run_cell_magic('time', '', 'prices_df.index = pd.to_datetime(prices_df.index)')


# In[780]:


prices_df[['KRXX4N-R', 'P8R3C2-R']]
index_list2 = list_df.groupby('index').groups['S5COND']
prices_df[list_df.iloc[index_list2]['fsym id']]
index_list2


# In[785]:


#create a function, given a layered grouping key, go through the column combinations and 
#pick the most correlated and cointegrated pairs
#should return a list of pairs (tuples)

def create_group(list_df, prices_df, key):
#PARAMETERS
# key = 'S5COND'
# ids_df = list_df
# prices_df = prices_df
##########
    index_list = list_df.groupby('index').groups[key]
    group_df = prices_df[list_df.iloc[index_list]['fsym id']]

    return group_df


# In[798]:


group1 = create_group(list_df, prices_df, 'S5AIRLX')
# group1 = group1.loc['01/01/2007':]
# cutoff = round(0.7*len(group1.index))
# train = group1[:cutoff]
# test = group1[cutoff:]
group1


# In[792]:


def group_all_indexes(list_df, prices_df):
    index_group = list_df.groupby(by = list_df['index'])
    index_group_dict = {}
    for k,v in index_group:
        index_group_dict[k] = create_group(list_df, prices_df, k)
    return index_group_dict


# In[793]:


get_ipython().run_cell_magic('time', '', 'index_df_list = group_all_indexes(list_df, prices_df)')


# In[797]:


index_df_list['S5AIRLX']


# In[744]:


train = train.fillna(method = 'ffill').dropna(axis = 'columns')
train


# In[828]:


#finding pairs within a dataframe
def find_pairs(df):
    n = df.shape[1]
    keys = df.keys()
    pairs = []
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    for i in range(0,n):
        for j in range(i+1,n):
            k1,k2 = keys[i], keys[j]
            s_df = df[[k1, k2]].dropna()
            s_df = s_df.loc[:,~s_df.columns.duplicated()]
            s1 = s_df[k1]
            s2 = s_df[k2]
            try:
                result = coint(s1, s2)
            except:
                continue
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
        


# In[801]:



score_matrix, pval_matrix, pairs = find_pairs(train)


# In[158]:


score_matrix


# In[799]:


pval_matrix.min()


# In[161]:


len(pairs)


# In[829]:


def find_all_pairs(index_dict):
    pairs = []
    for k in index_dict:
        _,_, pairs_v = find_pairs(index_dict[k])
        pairs += pairs_v
    return pairs


# In[ ]:


get_ipython().run_cell_magic('time', '', 'all_pairs = find_all_pairs(index_df_list)\nall_pairs')


# In[162]:


list_df[(list_df['fsym id'] == 'HNZVK6-R')& (list_df['index'] == 'S5COND')]


# In[163]:


def get_pair_names(tup):
    name1 = list_df[(list_df['fsym id'] == tup[0])]['Name'].values[0]
    name2 = list_df[(list_df['fsym id'] == tup[1])]['Name'].values[0]
    return name1 + ' and ' + name2
names = [get_pair_names(x) for x in pairs]
names


# In[164]:


len(pairs)/(114*114 / 2)


# In[165]:


pair = pairs[0]
pair


# In[500]:


def get_spread(S1, S2, lookback):
    hedge_ratio = S1.rolling(lookback).corr(S2) * S1.rolling(lookback).std() / S2.rolling(lookback).std()
    spread = S1 - hedge_ratio * S2
    return spread, hedge_ratio


# In[738]:


def adf_test(timeseries): #[1] is the p-value
    #print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    #dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used',
                                           #  'Number of Observations Used'])
    #for key,value in dftest[4].items():
       #dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    return dftest
    
spread,_ = get_spread(train[pair[0]], train[pair[1]], 60)
adf_test(spread[63:])[1]


# In[516]:


get_ipython().run_cell_magic('time', '', 'adf_test(spread.loc["01/01/2013": "12/31/2016"])')


# In[740]:


def filter_pairs_adf_year(pairs, train, lookback = 60):       
    
    def filter_adf(pair):
        spread, hr = get_spread(prices_df[pair[0]], prices_df[pair[1]], lookback)
        spread_group_pvals = spread.groupby(by = spread.index.year).apply(lambda x: adf_test(x.fillna(0))[1])
        average_pval_year = spread_group.fillna(method = 'bfill').mean()
        return average_pval_year < 0.10
    new_pairs = [pair for pair in pairs if filter_adf(pair)]
        
    return new_pairs
        
adf_pairs = filter_pairs_adf_year(pairs, train, 60)


# In[741]:


len(adf_pairs)


# In[729]:


spread, hr = get_spread(prices_df[pair[0]], prices_df[pair[1]], lookback)
spread_group = spread.groupby(by = spread.index.year)
spread_group.apply(lambda x: adf_test(x.fillna(0))[1])


# In[ ]:


def filter_pairs_year_coint(pairs):
    for pair in pairs:
        


# In[266]:


#def bonferroni_holm_correct(pairs, p_val, n):
#parameters
# pairs = adf_pairs
# p_val = 0.1
# n = 110* (110-1)/2

# new_pairs = []
# count = 0
# for pair in pairs:
#     cint = coint(train[pair[0]],train[pair[1]])
#     pair_p_val = cint[1]
#     if pair_p_val < p_val/(n-count):
#         new_pairs.append((pair, p_val))
#     count += 1
# new_pairs

# #super_coint_pairs = filter_pairs_coint(adf_pairs, train)
# super_coint_pairs


# In[187]:


len(super_coint_pairs)


# In[195]:


pair2 = super_coint_pairs[0]


# In[659]:


def run_pairs_strategy(name1, name2, prices_df, start_date, end_date, lookback, enter_width, exit_width):
#PARAMETERS
# S1 = prices_df[pair2[0]]
# S2 = prices_df[pair2[1]]
# name1 = S1.name
# name2 = S2.name
# start_date = '01/01/2005'
# end_date = '12/31/2012'
# lookback = 90
# width = 2
#############
    S1 = prices_df[name1]
    S2 = prices_df[name2]
    
    hr_lookback = 3
    num_trading_days = 21
    
    s_df = pd.concat([S1, S2], axis = 1, join = 'outer')
   # s_df.index = pd.to_datetime(s_df.index)
    
    s_df['Raw Spread'], s_df['Hedge Ratio'] = get_spread(np.log(s_df[name1]), np.log(s_df[name2]), hr_lookback * num_trading_days)
    
    s_df['Z-Score'] = (s_df['Raw Spread'] - s_df['Raw Spread'].rolling(lookback).mean()) / s_df['Raw Spread'].rolling(lookback).std()
#     rolling_z = s_df['Z-Score'].rolling(lookback).mean()
#     width_adj = s_df['Z-Score'].rolling(lookback).std() * width
#     s_df['Upper Band'] = np.array(rolling_z) + np.array(width_adj)
#     s_df['Lower Band'] = rolling_z - width_adj
    s_df['Upper Band'] = enter_width
    s_df['Lower Band'] = -1 * enter_width
    
    s_df = s_df.loc[start_date:end_date]
#     rolling_z = rolling_z.loc[start_date:end_date]
    
    s_df['Sell Position'] = np.where(s_df['Z-Score'] > enter_width, -1, 
                  np.where(s_df['Z-Score'] < exit_width, 0, np.nan))
    s_df['Sell Position'] = s_df['Sell Position'].ffill().fillna(0)
    
    s_df['Buy Position'] = np.where(s_df['Z-Score'] < -1 * enter_width, 1, 
                      np.where(s_df['Z-Score'] > -1 * exit_width, 0, np.nan))
    s_df['Buy Position'] = s_df['Buy Position'].ffill().fillna(0)
    
    

    s_df[name1 + '|Position'] = 100* (s_df['Buy Position'] + s_df['Sell Position'])
    s_df[name2 + '|Position'] =  -s_df[name1 + '|Position'] * s_df['Hedge Ratio']

    
    
    s_df[name1+"|P&L"] =  s_df[name1 + '|Position'] * s_df[name1].diff().shift(-1)
    s_df[name2+"|P&L"] =  s_df[name2 + '|Position'] * s_df[name2].diff().shift(-1)
    s_df['Total P&L'] = s_df[name1+'|P&L'] + s_df[name2+'|P&L']
    s_df['Cumulative P&L'] = s_df['Total P&L'].cumsum()

    return s_df





# In[660]:


get_ipython().run_cell_magic('time', '', 'pair2_df= run_pairs_strategy(pair2[0], pair2[1], prices_df, "01/01/2010", "12/31/2017", 63, 1.7, 1.5)\npair2_df')


# In[661]:


get_ipython().run_cell_magic('time', '', "pair2_group = pair2_df.groupby(by=[pair2_df.index.year, pair2_df.index.month])\nsharpes_group = pair2_group.apply(lambda x: get_sharpe(x['Total P&L']))\n#{str(k[1]) + '/' + str(k[0]):v for k,v in sharpes_group.items()}\n#pair2_group.get_group((2010,2))")


# In[662]:


def get_sharpe(r, rfr=0.02):
    if r.std():
        return (r.mean()-rfr) / r.std() * np.sqrt(252)
    return 0


# In[663]:


get_sharpe(pair2_df['Total P&L'])


# In[695]:


def train_sharpes(pair, interval):
    sharpes = []
    lookbacks = [x * 5 for x in range(1, 52)]
    widths = [x*0.05 + 0.5 for x in range(1,45)]
    for l in lookbacks:
        for w in widths:
            exit_width = random.uniform(0,w)
            s_df= run_pairs_strategy(pair[0], pair[1], prices_df, interval[0], interval[1], l, w, exit_width)
            sharpe = get_sharpe(s_df['Total P&L'])
            net_return = s_df['Total P&L'].sum()
            volatility = s_df['Total P&L'].std()
            d = {"Lookback": l,
                 "Enter Width": w,
                 "Exit Width": exit_width,
                 "Sharpe": sharpe,
                 "Net Return": net_return,
                 "Net Volatility": volatility}
            
#             dates_dict = {}
            s_df_group = s_df.groupby(by=[s_df.index.year, s_df.index.month])
            sharpes_group = s_df_group.apply(lambda x: get_sharpe(x['Total P&L']))
#             dates_dict = {str(k[1]) + '/' + str(k[0]):v for k,v in sharpes_group.items()}
            
            d['Monthly Sharpes'] = sharpes_group
            sharpes.append(d)
    return sorted(sharpes, key = lambda x: -x['Sharpe'])


# In[669]:


#using random parameter optimization (faster)
def train_sharpes_random(pair, interval):
    sharpes = []
    lookbacks = [random.randint(1,250) for _ in range(500)]
    widths = [random.uniform(0.5,2.0) for _ in range(500)]
    for i in range(500):
        lb = lookbacks[i]
        enter_wth = widths[i]
        exit_wth = random.uniform(0,enter_wth)
        
        s_df= run_pairs_strategy(pair[0], pair[1], prices_df, interval[0], interval[1], lb, enter_wth, exit_wth)
        sharpe = get_sharpe(s_df['Total P&L'])
        net_return = s_df['Total P&L'].sum()
        volatility = s_df['Total P&L'].std()
        d = {"Lookback": lb,
             "Enter Width": enter_wth,
             "Exit Width": exit_wth,
             "Sharpe": sharpe,
             "Net Return": net_return,
             "Net Volatility": volatility}

#             dates_dict = {}
        s_df_group = s_df.groupby(by=[s_df.index.year, s_df.index.month])
        sharpes_group = s_df_group.apply(lambda x: get_sharpe(x['Total P&L']))
#             dates_dict = {str(k[1]) + '/' + str(k[0]):v for k,v in sharpes_group.items()}

        d['Monthly Sharpes'] = sharpes_group
        sharpes.append(d)
    return sorted(sharpes, key = lambda x: -x['Sharpe'])


# In[696]:


get_ipython().run_cell_magic('time', '', 'sharpes2 = train_sharpes(pair2, ("01/01/1999", "12/31/2010"))')


# In[697]:


sharpes2


# In[672]:


max(sharpes2, key = lambda x: x['Sharpe'])['Lookback']


# In[691]:


sharpes_map2 = pd.DataFrame(sharpes2).pivot(index = "Lookback", columns = "Width", values = "Sharpe")

import seaborn as sb

sb.heatmap(sharpes_map2)


# In[698]:


rand_sharpes2 = pd.DataFrame(sharpes2)
rand_sharpes2


# In[707]:


fig = plt.figure(figsize=(40,30))
ax1 = fig.add_subplot(321)

ax1.plot(np.log(rand_sharpes2['Enter Width']) - np.log(rand_sharpes2['Exit Width']), rand_sharpes2['Sharpe'], '.')
plt.show()


# In[529]:


def test_sharpes(pair, interval, sharpes_list):
    training_parameters = sharpes_list
    testing_parameters = {}
    for p in training_parameters:
        lookback = p['Lookback']
        width = p['Width']
        s_df = run_pairs_strategy(pair[0], pair[1], prices_df, interval[0], interval[1] , lookback, width )
        sharpe = get_sharpe(s_df['Total P&L'])
        net_return = s_df['Total P&L'].sum()
        volatility = s_df['Total P&L'].std()
        d = {"Lookback": lookback,
             "Width": width,
             "Sharpe": sharpe,
             "Net Return": net_return,
             "Net Volatility": volatility}

#             dates_dict = {}
        s_df_group = s_df.groupby(by=[s_df.index.year, s_df.index.month])
        sharpes_group = s_df_group.apply(lambda x: get_sharpe(x['Total P&L']))
#             dates_dict = {str(k[1]) + '/' + str(k[0]):v for k,v in sharpes_group.items()}

        d['Monthly Sharpes'] = sharpes_group
        testing_parameters[(lookback, width)] = d
    return testing_parameters


# In[538]:


get_ipython().run_cell_magic('time', '', 'test_sharpes2 = test_sharpes(pair2, ("01/01/2018", "12/31/2020"), sharpes2)')


# In[544]:


[v['Net Volatility'] for k, v in test_sharpes2.items()]


# In[547]:


def rolling_backtest(pair, price_df):
    intervals = [("01/01/" + str(x), "12/31/"+str(x+10), "01/01/"+str(x+11), "12/31/"+str(x+14)) for x in range(1999,2006)]
    testing_info = []
    for interval in intervals:
        trained_sharpes = train_sharpes(pair, (interval[0], interval[1]))
        tested_sharpes = test_sharpes(pair, (interval[2], interval[3]), trained_sharpes)
        
        average_training_volatility = np.array([x['Net Volatility'] for x in trained_sharpes]).mean()
        average_testing_volatility = np.array([v['Net Volatility'] for k,v in tested_sharpes.items()]).mean()

        train_sharpes_only = np.array([x['Sharpe'] for x in trained_sharpes])
        test_sharpes_only = np.array([v['Sharpe'] for k,v in tested_sharpes.items()])

        avg_best_15_train_sharpe = train_sharpes_only[:15].mean()
        avg_best_15_train_returns = np.array([x['Net Return'] for x in trained_sharpes])[:15].mean()
        avg_best_15_test_sharpe = test_sharpes_only[-15:].sum()/15
        avg_best_15_test_returns = np.array([v['Net Return'] for k,v in tested_sharpes.items()])[:15].mean()
        
        best_performance = max(trained_sharpes, key = lambda x: x['Sharpe'])
        best_training_lookback = best_performance['Lookback']
        best_training_width = best_performance['Width']
        best_test_performance = tested_sharpes[(best_training_lookback, best_training_width)]

#         weighted_sharpes = 0.3*train_sharpes_only + 0.7*test_sharpes_only

#         max_weighted_sharpe = max(weighted_sharpes)

        testing_info.append({"Train Start": interval[0],
                             "Train End": interval[1],
                             "Test Start": interval[2],
                             "Test End": interval[3],
                             "Average Training Volatility": average_training_volatility,
                             "Average Testing Volatility": average_testing_volatility,
                             "Average 15 Best Training Sharpe": avg_best_15_train_sharpe,
                             "Average 15 Best Training Returns": avg_best_15_train_returns,
                             "Average 15 Best Testing Sharpe": avg_best_15_test_sharpe,
                             "Average 15 Best Testing Returns": avg_best_15_test_returns,
                             "Mean Squared Error ":((train_sharpes_only - test_sharpes_only)**2).mean(),
                             "Best* Lookback (Sharpe)": best_training_lookback,
                             "Best* Width (Sharpe)": best_training_width,
                             "Best* Training Sharpe (Sharpe)": best_performance['Sharpe'],
                             "Best* Return (Sharpe)":  best_performance['Net Return'],
                             "Best* Volatility (Sharpe)": best_performance['Net Volatility'],
                             "Best* Monthly Sharpes": best_performance['Monthly Sharpes'],
                             "Test Sharpe (Sharpe)": best_test_performance['Sharpe'],
                             "Test Return (Sharpe)": best_test_performance['Net Return'],
                             "Test Volatility (Sharpe)": best_test_performance['Net Volatility'],
                             "Test Monthly Sharpes (Sharpe)": best_test_performance['Monthly Sharpes']})
    return pd.DataFrame(testing_info)
    
    


# In[548]:


get_ipython().run_cell_magic('time', '', 'info = rolling_backtest(pair2, prices_df)\ninfo')


# In[568]:


info.iloc[:, 10:]['Test Monthly Sharpes (Sharpe)'].iloc[6].reset_index(drop = True)


# In[579]:


fig = plt.figure(figsize=(40,30)) #overall plot size
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)

ax1.plot(hedge_ratio)
ax1.set_title("Hedge Ratio")

ax2.set_title("Z-Score w/ Bands")
ax2.plot(pair2_df['Z-Score'], label = 'Spread')

ax2.plot(pair2_df['Upper Band'], label = 'Spread')
ax2.plot(pair2_df['Lower Band'], label = 'Spread')

ax3.set_title("Cumulative P&L")
ax3.plot(pair2_df['Cumulative P&L'])

ax4.set_title("Stock Graphs")
ax4.plot(pair2_df['BRWKF0-R'], label = "S1")
ax4.plot(pair2_df['FH1NKL-R'], label = "S2")





# In[574]:


fig = plt.figure(figsize=(40,30))
ax5 = fig.add_subplot(325)


ax5.set_title("Monthly Sharpe")
ax5.plot(info['Train Monthly Sharpes (Sharpe)'].iloc[6].reset_index(drop = True))

plt.xlabel('Time')
plt.show()

