
# coding: utf-8

# In[13]:


import numpy as np
import talib
import pandas as pd
from dateutil import parser
import os

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()  # <== that's all it takes :-)


# In[14]:


# get_ipython().magic('run C:/Users/alifa/Desktop/FinanceCodes/Variables.py')


# In[3]:


def load_price_df_yahoo_finance(symbol, start, end, file_path):

    price_df = pdr.get_data_yahoo(symbol, start = start, end = end)
    price_df['Adj Close Factor'] = price_df['Adj Close'] / price_df['Close']
    price_df['Adj Open'] = price_df['Open'] * price_df['Adj Close Factor']
    price_df['Adj Low'] = price_df['Low'] * price_df['Adj Close Factor']
    price_df['Adj High'] = price_df['High'] * price_df['Adj Close Factor']
    price_df = price_df[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Volume']]
    price_df = price_df.rename(columns={'Adj Open': 'Open', 'Adj High': 'High', 'Adj Low': 'Low', 'Adj Close': 'Close'})

    folder_path = file_path+ '/{}.csv'
    price_df.to_csv(folder_path.format(symbol))

    return price_df


# In[4]:


def TSE_price_df_extract(symbol, dir_path, type_symbol = 'SH'):
    '''
    :param symbol: the name of the symbol
    :param dir_path: directory that has all the excel file of price df
    :param type_symbol: 'SH' or 'IDX'
    :return: price df
    '''
    if type_symbol == 'SH':
        file_name = symbol + '_D_SH.txt'
        url_file = dir_path + '/' + file_name
        columns_to_use = ['<Ticker>', '<DTYYYYMMDD>', '<TIME>', '<Open>', '<High>', '<Low>']
        x = pd.read_csv(url_file, usecols=columns_to_use)
        x.rename(columns={'<Ticker>': 'Date', '<DTYYYYMMDD>': 'Open',
                          '<TIME>': 'High', '<Open>': 'Low', '<High>': 'Close',
                          '<Low>': 'Volume'}, inplace=True)
    elif type_symbol == 'IDX':
        file_name = symbol + '_D_IDX.txt'
        url_file = dir_path + '/' + file_name
        columns_to_use = ['<Per>', '<TIME>', '<Open>', '<High>', '<Low>', '<Close>']

        x = pd.read_csv(url_file, usecols=columns_to_use)
        x.rename(columns={'<Per>': 'Date', '<TIME>': 'Open',
                          '<Open>': 'High', '<High>': 'Low', '<Low>': 'Close',
                          '<Close>': 'Volume'}, inplace=True)

    x['Date'] = list(map(lambda x: parser.parse(str(x)), x['Date']))
    # x['Date'] = x['Date'].astype('Date')
    x = x[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    x = x.reset_index(drop = True)

    return x


# In[5]:


def price_df_extract_from_filename(filename, price_df_path, market):
    '''
    return price_df from file name
    '''
    if market == 'US':
        symbol = filename.split('.csv')[0]
        file_path = os.path.join(price_df_path, filename)
        price_df = pd.read_csv(file_path)
    elif market == 'TSE':
        symbol = filename.split('_')[0]
        type_symbol = filename.split('_')[2].split('.')[0]

        price_df = TSE_price_df_extract(symbol, price_df_path, type_symbol=type_symbol)

    return symbol, price_df



def load_forex_price_df(currency, price_df_path, timeFrame):
    
    if 'M' in timeFrame:
        fileName = '{a}{b}.csv'.format(a=currency, b = timeFrame.split('M')[1])
        
    file_path = os.path.join(price_df_path, fileName )
    df = pd.read_csv(file_path, names = ['Date', 'Time','Open', 'High', 'Low', 'Close', 'Volume'] )
    df['Year'] = list(map(lambda x: x.split('.')[0], df['Date']))
    df['Month'] = list(map(lambda x: x.split('.')[1], df['Date']))
    df['Day'] = list(map(lambda x: x.split('.')[2], df['Date']))
    df['Hour'] = list(map(lambda x: x.split(':')[0], df['Time']))
    df['Minute'] = list(map(lambda x: x.split(':')[1], df['Time']))

    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df    

