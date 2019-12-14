
# coding: utf-8

# ## Notebook for creating df to be fed into ML algorithms
# ### Include all the fucntion of Technical indicators

# In[46]:


import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## 1. Overlay

# In[50]:


def bollinger(price_df, timeperiod = 5, nbdevup = 2, nbdevdn = 2, trend_period=[1,2,3,4,5]):   
    '''
    attn: work to remove return columns
    '''
    
    close_p = np.array(price_df['Close'])
    price_df['High_Band'], price_df['Med_Band'], price_df['Low_Band'] =     talib.BBANDS(close_p, timeperiod = timeperiod, nbdevup = nbdevup, nbdevdn = nbdevdn)
    
    price_df['P/High_Band'] =  price_df['Close'] / price_df['High_Band']
    price_df['P/Med_Band']  =  price_df['Close'] / price_df['Med_Band']
    price_df['P/Low_Band']  =  price_df['Close'] / price_df['Low_Band']
    
    for i in trend_period:
        price_df ['return_{a}'.format(a=i)] = price_df['Close'].pct_change(i)
        
    # multiply close price change for periods in trend_period to p/ each band.
    # I wanted to see the combination of price trend and the location of price compared to each band
    for i in trend_period:
        price_df['Trend_{a}x P/High_Band'.format(a=i)] = price_df['return_{a}'.format(a=i)] * price_df['P/High_Band']
        price_df['Trend_{a}x P/Med_Band'.format(a=i)] =  price_df['return_{a}'.format(a=i)] * price_df['P/Med_Band']
        price_df['Trend_{a}x P/Low_Band'.format(a=i)] =  price_df['return_{a}'.format(a=i)] * price_df['P/Low_Band']
    
    return price_df


# In[51]:


def moving_average(price_df, periods_list=[2, 3, 4], ave_type='Close', type_ma='sma'):
    '''
    This function just calculates moving average and their relationship with eachother and the close price

    ave_type: Pivot or Close

    periods_list: whatever periods list for getting average except for
    '''

    if 'Pivot' not in price_df.columns:
        price_df['Pivot'] = list(
            map(lambda x, y, z: (x + y + z) / 3, price_df['Close'], price_df['High'], price_df['Low']))

    for period in periods_list:
        label1 = '{a}_{b}_{c}'.format(a=ave_type, b=type_ma, c=period)
        label2 = 'P/' + label1
        label3 = 'Pivot/' + label1
        if type_ma == 'sma':
            price_df[label1] = price_df[ave_type].rolling(window=period).mean()
            price_df[label2] = price_df['Close'] / price_df[label1]
            price_df[label3] = price_df['Pivot'] / price_df[label1]
        else:
            price_df[label1] = price_df[ave_type].ewm(span=period).mean()
            price_df[label2] = price_df['Close'] / price_df[label1]
            price_df[label3] = price_df['Pivot'] / price_df[label1]

    num_periods = len(periods_list)

    for i in range(1, num_periods):
        for j in range(0, i):
            label1 = ave_type + '_' + type_ma + '_' + str(periods_list[i])
            label2 = ave_type + '_' + type_ma + '_' + str(periods_list[j])
            price_df[label1 + '/' + label2] = price_df[label1] / price_df[label2]

    return price_df


# In[52]:


def parabolic_SAR(price_df, acceleration=0.02, maximum=0.2):
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])

    price_df['SAR'] = talib.SAR(high_p, low_p, acceleration=acceleration, maximum=maximum)

    price_df['Close_to_SAR'] = price_df['Close'] / price_df['SAR']

    price_df['SAREXT'] = talib.SAREXT(high_p, low_p)
    price_df['Close_to_SAREXT'] = price_df['Close'] / price_df['SAREXT']

    return price_df


# In[53]:


def triple_exponential_MA(price_df, timeperiod=5, vfactor=0.7):
    close_p = np.array(price_df['Close'])
    price_df['T3'] = talib.T3(close_p, timeperiod=timeperiod, vfactor=vfactor)

    price_df['Triple_Exp_MA'] = talib.TEMA(close_p, timeperiod=timeperiod)

    return price_df


# ## 2. Momentum

# In[54]:


def average_directional_movement_index (price_df, timeperiod=14, trend_periods=[1, 2, 3, 4, 5]):
    '''
    Both Average Directional Movement Index and Average Directional Movement Index Rating
    Minus and plus DI
    Minus and Plus DM
    :param price_df:
    :param timeperiod:
    :param trend_periods:
    :return:
    '''


    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])

    label_ADX = 'ADX_{a}'.format(a=timeperiod)
    label_ADXR = 'ADXR_{a}'.format(a=timeperiod)

    price_df[label_ADX] = talib.ADX(high_p, low_p, close_p, timeperiod=timeperiod)
    price_df[label_ADXR] = talib.ADXR(high_p, low_p, close_p, timeperiod=timeperiod)

    price_df['MINUS_DI_' + str(timeperiod)] = talib.MINUS_DI(high_p, low_p, close_p, timeperiod)
    price_df['PLUS_DI_' + str(timeperiod)] = talib.PLUS_DI(high_p, low_p, close_p, timeperiod)
    price_df['PLUS_DI/MINUS_DI_' + str(timeperiod)] = price_df['PLUS_DI_' + str(timeperiod)] / price_df[
        'MINUS_DI_' + str(timeperiod)]

    price_df['MINUS_DM_' + str(timeperiod)] = talib.MINUS_DM(high_p, low_p, timeperiod)
    price_df['PLUS_DM_' + str(timeperiod)] = talib.PLUS_DM(high_p, low_p, timeperiod)
    price_df['PLUS_DM/MINUS_DM_' + str(timeperiod)] = price_df['PLUS_DM_' + str(timeperiod)] / price_df[
        'MINUS_DM_' + str(timeperiod)]

    for period in trend_periods:
        price_df[label_ADX + '_pct_' + str(period)] = price_df[label_ADX].pct_change(period)
        price_df[label_ADXR + '_pct_' + str(period)] = price_df[label_ADXR].pct_change(period)

    return price_df


# In[55]:


def apo(price_df, fastperiod=12, slowperiod=26, matype=0):
    '''
    Absolute Price Oscillator
    :param price_df:
    :param fastperiod:
    :param slowperiod:
    :param matype:
    :return:
    '''

    close_p = np.array(price_df['Close'])
    price_df['APO'] = talib.APO(close_p, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    return price_df


# In[56]:


def aroon(price_df, timeperiod=14):
    '''
    AROON
    Aroon Oscillator
    :param price_df:
    :param timeperiod:
    :return:
    '''
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])

    price_df['AROON_down'] = talib.AROON(high_p, low_p, timeperiod)[0]
    price_df['AROON_up'] = talib.AROON(high_p, low_p, timeperiod)[1]

    price_df['AROONOSC'] = talib.AROONOSC(high_p, low_p, timeperiod)

    return price_df


# In[57]:


def balance_of_power(price_df):
    '''
    Balance of Power
    :param price_df:
    :return:
    '''
    close_p = np.array(price_df['Close'])
    open_p = np.array(price_df['Open'])
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])

    price_df['BOP'] = talib.BOP(open_p, high_p, low_p, close_p)

    return price_df


# In[58]:


def commodity_channel_index(price_df, timeperiod=14, trend_period=[1,2,3,4,5,6,7]):
    '''
    Check whether CCI has crossed zero above
    Input: price_df
    Output: price_df including CCI
            trend of CCI for periods in trend_period
    '''
    close_p = np.array(price_df['Close'])
    open_p = np.array(price_df['Open'])
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])

    label = 'CCI_' + str(timeperiod)

    price_df[label] = talib.CCI(high_p, low_p, close_p, timeperiod)

    '''
    CCI_Cross_Zero_Above=[0]*(time_period+2)

    for i in range(time_period+2,len(price_df)):
        if  (price_df['CCI'].iloc[i]>0 and price_df['CCI'].iloc[i-1]<0):
            CCI_Cross_Zero_Above.append(1)
        else:
            CCI_Cross_Zero_Above.append(0)

    price_df['CCI_Cross_Zero_Above']=CCI_Cross_Zero_Above

    '''

    for i in trend_period:
        price_df['CCI_' + str(timeperiod) + '_Trend_' + str(i)] = price_df[label].pct_change(i)
        # price_df['CCI_'+str(timeperiod)+'_x Trend_'+str(i)] = list(map(lambda x,y:x*y,price_df[label],price_df['CCI_'+str(timeperiod)+'_Trend_'+str(i)]))

    return price_df


# In[59]:


def chande_momentum_oscillator (price_df, timeperiod=14):
    '''
    Chandle Momentum Oscillator
    :param price_df:
    :param timeperiod:
    :return:
    '''

    close_p = np.array(price_df['Close'])
    price_df['CMO'] = talib.CMO(close_p, timeperiod)

    return price_df


# In[60]:


def directional_movement_index (price_df, timeperiod=14):
    '''
    Directional Movement Index
    :param price_df:
    :param timeperiod:
    :return:
    '''
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])

    price_df['DX'] = talib.DX(high_p, low_p, close_p, timeperiod=14)

    return price_df


# In[61]:


def macd(price_df, fastperiod=12, slowperiod=26, signalperiod=9):
    '''
    MACD
    :param price_df:
    :param fastperiod:
    :param slowperiod:
    :param signalperiod:
    :return:
    '''
    macd_output = talib.MACD(price_df['Close'].values, fastperiod, slowperiod, signalperiod)

    price_df['MACD'] = macd_output[0]
    price_df['MACD_Signal'] = macd_output[1]
    price_df['Hist'] = macd_output[2]

    return price_df


# In[62]:


def macd_hist_sign(hist_chg, hist):
    if ((hist_chg > 0) and (hist > 0)):
        hist_sign = 1
    elif ((hist_chg < 0) and (hist > 0)):
         hist_sign = 0
    elif ((hist_chg > 0) and (hist < 0)):
         hist_sign = -1
    elif ((hist_chg < 0) and (hist < 0)):
         hist_sign = 0
    else: 
         hist_sign = 10        
    return hist_sign


# ## Test- MACD chg of direction

# In[63]:


# file_name = 'AAPL.csv'
# price_df = price_df_extract_from_filename(file_name, price_df_path, market)[1]
# price_df = macd(price_df, fastperiod=12, slowperiod=26, signalperiod=9)
# price_df['Hist(-1)'] = price_df['Hist'].shift(1)
# price_df['Hist_Chg'] = price_df['Hist'] / price_df['Hist(-1)'] -1
# price_df['Hist_Chg_Sign'] = list(map(macd_hist_sign, price_df['Hist_Chg'], price_df['Hist']))

# price_df.tail(10)


# In[64]:


def macd_ext(price_df, fastperiod=12, fastmatype=1, slowperiod=26, slowmatype=1,              signalperiod=9, signalmatype=1):
    close_p = price_df['Close'].values
    macd_output = talib.MACDEXT(close_p, fastperiod=fastperiod, fastmatype=fastmatype,                                 slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod,                                 signalmatype=slowmatype)

    price_df['MACD_ext'] = macd_output[0]
    price_df['MACD_Signal_ext'] = macd_output[1]
    price_df['Hist_ext'] = macd_output[2]

    return price_df


# In[65]:


def macd_fix(price_df, signalperiod=9):
    macdfix_output = talib.MACDFIX(price_df['Close'].values, signalperiod)

    price_df['MACD_FIX'] = macdfix_output[0]
    price_df['MACD_FIX_Signal'] = macdfix_output[1]
    price_df['MACD_FIX_Hist'] = macdfix_output[2]

    return price_df


# In[66]:


def money_flow_index (price_df, time_period=14, trend_periods=[1,2,3,4,5,6,7]):
    '''

    Input: price_df
    Output: price_df including MFI, its trend, trend x MFI, difference between MFI and its previous #days in trend_period
    '''
    close_p = np.array(price_df['Close'])
    open_p = np.array(price_df['Open'])
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    vol_p = np.array(price_df['Volume'], dtype='float')  # remember for talib numbers should be float

    label = 'MFI_' + str(time_period)

    price_df[label] = talib.MFI(high_p, low_p, close_p, vol_p)

    # to find the growth of MFI for the periods in trend_period and then multiply that by MFI
    for i in trend_periods:
        price_df['MFI_' + str(time_period) + '_Trend_' + str(i)] = price_df[label].pct_change(i)
        price_df['MFI_' + str(time_period) + '_x Trend_' + str(i)] =             list(map(lambda x, y: x * y, price_df[label], price_df['MFI_' + str(time_period) + '_Trend_' + str(i)]))

    # find the differnce between today's MFI and n-days before
    for i in trend_periods:
        price_df[label + '-' + label + '|' + str(-i) + '|'] =             price_df[label] - price_df[label].iloc[-(i + 1)]

    return price_df


# In[67]:


def minus_di(price_df, timeperiod=14):
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])

    price_df['MINUS_DI'] = talib.MINUS_DI(high_p, low_p, close_p, timeperiod)

    return price_df


# In[68]:


def momentum(price_df, timeperiod=10):
    close_p = np.array(price_df['Close'])
    label = 'MOM_' + str(timeperiod)

    price_df[label] = talib.MOM(close_p, timeperiod)

    return price_df


# In[69]:


def percentage_price_oscillator(price_df, fastperiod = 12 , slowperiod = 26, matype = 0 ):

    price_df['PPO_'+str(matype)] = talib.PPO(price_df['Close'].values, fastperiod, slowperiod, matype)

    return price_df


# In[70]:


def relative_strength_index(price_df, period=14, trend_period=[1,2,3,4,5,6,7], RSI_MA=14):
    '''
    Input:
        price df: including OHLCV
    '''

    price_df['RSI'] = talib.RSI(price_df['Close'].values, period)

    '''

    RSI_Cross_30_Above=[0]*(period+1)

    for i in range(period+1,len(df)):
        if  (df['RSI'].iloc[i]>30 and df['RSI'].iloc[i-1]<30):
            RSI_Cross_30_Above.append(1)
        elif (df['RSI'].iloc[i]<70 and df['RSI'].iloc[i-1]>70) :
            RSI_Cross_30_Above.append(-1)
        else:
            RSI_Cross_30_Above.append(0)

    df['RSI_Crossover_30_70']=RSI_Cross_30_Above

    '''
    # to find MA of RSI and its distance to RSI
    price_df['RSI_MA'] = price_df['RSI'].ewm(span=RSI_MA).mean()
    price_df['RSI_to_MA'] = price_df['RSI'] / price_df['RSI_MA']

    for i in trend_period:
        price_df['RSI_Trend_' + str(i)] = price_df['RSI'].pct_change(i)
        price_df['RSI x Trend_' + str(i)] = list(
            map(lambda x, y: x * y, price_df['RSI'], price_df['RSI_Trend_' + str(i)]))

    return price_df


# In[71]:


def relative_vigor_index(price_df, period =10, trend_period=[1,2,3,4,5,6,7]):
    price_df['RVI_1'] = (price_df['Close'] - price_df['Open']) / (price_df['High'] - price_df['Low'])
    price_df['RVI_2'] = price_df['RVI_1'].rolling(window = period).mean()
    price_df['RVI_2/RVI1'] = price_df['RVI_2']/price_df['RVI_1']
    
    for i in trend_period:
        price_df['RVI_2_Trend_' + str(i)] = price_df['RVI_2'].pct_change(i)
      
    return price_df  


# In[72]:


def stochastic(price_df, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])

    price_df['STOCH_slowk_' + str(slowk_matype)] =     talib.STOCH(high_p, low_p, close_p, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)[0]
    price_df['STOCH_slowd_' + str(slowk_matype)] =     talib.STOCH(high_p, low_p, close_p, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)[1]

    price_df['STOCH_slowk/slowd_' + str(slowk_matype)] = list(map((lambda x, y: x / y),                                                                   price_df['STOCH_slowk_' + str(slowk_matype)],                                                                   price_df['STOCH_slowd_' + str(slowk_matype)]))

    return price_df


# In[73]:


def stochasticF(price_df, fastk_period=5, fastd_period=3, fastd_matype=0):
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])

    label1 = 'STOCHF_fastk_' + str(fastd_matype)
    label2 = 'STOCHF_fastd_' + str(fastd_matype)

    price_df[label1] = talib.STOCHF(high_p, low_p, close_p, fastk_period, fastd_period, fastd_matype)[0]
    price_df[label2] = talib.STOCHF(high_p, low_p, close_p, fastk_period, fastd_period, fastd_matype)[1]

    price_df['STOCH_fastk/fastd_' + str(fastd_matype)] = list(map((lambda x, y: x / y), price_df[label1],                                                                   price_df[label2]))

    return price_df


# In[74]:


def stochasticRSI(price_df, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    close_p = np.array(price_df['Close'])

    label1 = 'STOCHRSI_fastk_' + str(timeperiod) + '_' + str(fastd_matype)
    label2 = 'STOCHRSI_fastd_' + str(timeperiod) + '_' + str(fastd_matype)

    result = talib.STOCHRSI(close_p, timeperiod, fastk_period, fastd_period, fastd_matype)
    price_df[label1] = result[0]
    price_df[label2] = result[1]

    label3 = 'STOCHRSI_fastd/fastk' + str(timeperiod) + '_' + str(fastd_matype)
    price_df[label3] = list(map((lambda x, y: x / y), price_df[label1],                                 price_df[label2]))

    return price_df


# In[75]:


def trix(price_df, timeperiod=30, trend_periods = [1,2,3,4,5,6,7]):
    close_p = np.array(price_df['Close'])
    label = 'trix_{a}'.format(a=timeperiod)
    price_df[label] = talib.TRIX(close_p, timeperiod)
    
    for i in trend_periods:
        price_df['{a}_Trend_{b}'.format(a=label,b=i)] = price_df[label].pct_change(i)  

    return price_df


# In[76]:


def williamsR(price_df, time_period= 14, trend_periods = [1,3,5,7]):
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])

    price_df['williamsR'] = talib.WILLR(high_p, low_p, close_p, timeperiod= time_period)    

    return price_df


# In[77]:


def ultimate_oscillator(price_df, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])

    price_df['ULTOSC'] = talib.ULTOSC(high_p, low_p, close_p, timeperiod1, timeperiod2, timeperiod3)

    return price_df


# In[78]:


def donchianChannel(price_df, time_period):
    
    # Donchian Channel
    price_df['High_{a}'.format(a = time_period)] = price_df['High'].rolling(window=time_period).max()
    price_df['Low_{a}'.format(a = time_period) ] = price_df['Low'].rolling(window=time_period).min()
    price_df['Close/High_{a}'.format(a = time_period)] = price_df['Close'] / price_df['High_{a}'.format(a = time_period)]
    price_df['Close/Low_{a}'.format(a = time_period)] = price_df['Close'] / price_df['Low_{a}'.format(a = time_period)]
    
    return price_df


# In[110]:


def ichimoku_clouds(price_df):
    

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    price_df['tenkan_sen'] = (price_df['High'].rolling(window=9).max() + price_df['Low'].rolling(window=9).min())/2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))    
    price_df['kijun_sen'] = (price_df['High'].rolling(window=26).max() + price_df['Low'].rolling(window=26).min())/2
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 shifted forward for 26 periods   
    price_df['senkou_span_a_0'] = (price_df['tenkan_sen'] + price_df['kijun_sen'])/2
    price_df['senkou_span_a'] = price_df['senkou_span_a_0'].shift(26)
    
    
    price_df['senkou_span_b_0'] = ((price_df['High'].rolling(window=52).max() + price_df['Low'].rolling(window=52).min()) / 2).shift(26)
    price_df['senkou_span_b'] = price_df['senkou_span_b_0'].shift(26)    
    
    # The most current closing price plotted 22 time periods behind (optional)
    price_df['chikou_span'] = price_df['Close'].shift(26) # 26 according to investopedia
    
    
    price_df['senkou_span_a/senkou_span_b'] = price_df['senkou_span_a'] / price_df['senkou_span_b']
    price_df['Close/senkou_span_b'] = price_df['Close'] / price_df['senkou_span_b']
    price_df['Close/senkou_span_a'] = price_df['Close'] / price_df['senkou_span_a']
    price_df['Close/chikou_span'] = price_df['Close'] / price_df['chikou_span']
    
    price_df['chikou_span/senkou_span_a'] = price_df['chikou_span'] / price_df['senkou_span_a']    
    price_df['chikou_span/senkou_span_b'] = price_df['chikou_span'] / price_df['senkou_span_b']
    
    price_df['tenkan_sen/senkou_span_a'] = price_df['tenkan_sen'] / price_df['senkou_span_a']    
    price_df['tenkan_sen/senkou_span_b'] = price_df['tenkan_sen'] / price_df['senkou_span_b']
    
    price_df['kijun_sen/senkou_span_a'] = price_df['kijun_sen'] / price_df['senkou_span_a']    
    price_df['kijun_sen/senkou_span_b'] = price_df['kijun_sen'] / price_df['senkou_span_b']
    
    return price_df


# In[114]:


# file_name = 'AAPL.csv'
# price_df = price_df_extract_from_filename(file_name, price_df_path, market)[1]
# price_df = ichimoku_clouds(price_df)

# price_df.tail(10)


# ## Volatility

# In[115]:


def average_true_range(price_df, timeperiod=5):

    close_p = np.array(price_df['Close'])
    open_p = np.array(price_df['Open'])
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])

    label = 'ATR_' + str(timeperiod)
    label2 = label + '/ATR(-5)'
    label3 = label + '/ATR(-10)'   
    
    price_df[label] = talib.ATR(high_p, low_p, close_p, timeperiod)
    price_df[label2] = price_df[label]  / price_df[label].shift(5)
    price_df[label3] = price_df[label]  / price_df[label].shift(10)  

    return price_df


# In[116]:


def normalized_average_true_range(price_df, timeperiod = 14):
    '''
    set time period = 14
    '''

    close_p = np.array(price_df['Close'])
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])

    price_df['NATR'] = talib.NATR(high_p, low_p, close_p, timeperiod)

    return price_df


# In[117]:


def true_range(price_df):
    close_p = np.array(price_df['Close'])
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])

    price_df['TRANGE'] = talib.TRANGE(high_p, low_p, close_p)
    price_df['TRANGE(-1)'] = price_df['TRANGE'].shift(1)
    price_df['TRANGE(-2)'] = price_df['TRANGE'].shift(2)
    price_df['TRANGE(-3)'] = price_df['TRANGE'].shift(3)

    return price_df


# In[118]:


def keltner_channel(price_df, period = 14):
    
    price_df['pivot'] = (price_df['High'] + price_df['Low'] + price_df['Close']) / 3
    price_df['KelChM'+str(period)] = price_df['pivot'].rolling(window= period).mean()
    price_df['KelChU'+str(period)] =     ((4 * price_df['High'] - 2 * price_df['Low'] + price_df['Close']) / 3).rolling(window= period).mean()
    
    price_df['KelChD'+str(period)] =     ((-2 * price_df['High'] + 4 * price_df['Low'] + price_df['Close']) / 3).rolling(window= period).mean()
    
    price_df['Close/KelChU'] = price_df['Close'] / price_df['KelChU'+str(period)]
    price_df['Close/KelChD'] = price_df['Close'] / price_df['KelChD'+str(period)]
    
    return price_df


# ## Volume

# In[119]:


def vwap(price_df):

    price_df['Typical_Price'] = list(map((lambda x, y, z: (x + y + z) / 3), price_df['Close'], price_df['Low'], price_df['High']))
    price_df['VWAP'] = list(map((lambda x, y: x * y), price_df['Typical_Price'], price_df['Volume']))

    price_df['VWAP_MA5'] = price_df['VWAP'].ewm(span=5).mean()
    price_df['VWAP_MA10'] = price_df['VWAP'].ewm(span=10).mean()
    price_df['VWAP_MA20'] = price_df['VWAP'].ewm(span=20).mean()

    price_df['VWAP/VWAP5'] = price_df['VWAP'] / price_df['VWAP_MA5']
    price_df['VWAP5/VWAP10'] = price_df['VWAP_MA5'] / price_df['VWAP_MA10']
    price_df['VWAP10/VWAP20'] = price_df['VWAP_MA10'] / price_df['VWAP_MA20']

    price_df.drop('Typical_Price', axis=1, inplace=True)

    return price_df


# In[120]:


def on_balance_volume (price_df, trend_period=[1,2,3,4,5,6,7]):
    '''
    Output: price_df with OBV and growth rate for periods in trend_period
            with OBV x growth rate of each period
    '''

    close_p = np.array(price_df['Close'])
    volume_p = np.array(price_df['Volume'], dtype='float')

    price_df['OBV'] = talib.OBV(close_p, volume_p)

    for i in trend_period:
        price_df['OBV_Trend_' + str(i)] = price_df['OBV'].pct_change(i)
        price_df['OBV x Trend_' + str(i)] =             list(map(lambda x, y: x * y, price_df['OBV'], price_df['OBV_Trend_' + str(i)]))

    return price_df


# In[121]:


def chaikin(price_df, fastperiod=3, slowperiod=10):
    '''
    Chaikin A/D Oscillator and Chaikin A/D line
    :param price_df:
    :param fastperiod:
    :param slowperiod:
    :return:
    '''
    high_p = np.array(price_df['High'])
    low_p = np.array(price_df['Low'])
    close_p = np.array(price_df['Close'])
    volume_p = np.array(price_df['Volume'], dtype='float')

    price_df['Chaikin_AD_Line'] = talib.AD(high_p, low_p, close_p, volume_p)
    price_df['Chaikin_AD_Oscillator'] = talib.ADOSC(high_p, low_p, close_p, volume_p, fastperiod=fastperiod,                                                     slowperiod=slowperiod)

    return price_df


# In[122]:


def ad_line(price_df):
    
    func = (lambda high, low, close: (((close - low) - (high - close)) /(high - low)))
    
    price_df['MF_Multiplier'] = list(map(func, price_df['High'], price_df['Low'], price_df['Close']))
    func = (lambda high, low, close: (((close - low) - (high - close)) /(high - low)))
    price_df = return_price_df('AAPL', market='US')
    
    price_df['MF_Multiplier'] = list(map(func, price_df['High'], price_df['Low'], price_df['Close']))
    price_df['MF_Values'] = price_df['MF_Multiplier'] * price_df['Volume']
    MF_Values = price_df['MF_Values'].values
    adl = []
    adl.append(MF_Values[0])
    for i in range(1, len(price_df)):
        adl_value = adl[i-1] + MF_Values[i]
        adl.append(adl_value)

    price_df['ADL_Line'] = adl


# ## Cycles

# In[123]:


def SineWave(price_df):
    close_p = np.array(price_df['Close'])
    price_df['sine'] = talib.HT_SINE(close_p)[0]
    price_df['lead_sine'] = talib.HT_SINE(close_p)[1]

    return price_df

def Dominant_Cycle_Period(price_df):
    close_p = np.array(price_df['Close'])
    price_df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_p)

    return price_df

def Dominant_Cycle_Phase(price_df):
    close_p = np.array(price_df['Close'])
    price_df['HT_DCPHASE'] = talib.HT_DCPHASE(close_p)

    return price_df

def Phasor_Components(price_df):
    close_p = np.array(price_df['Close'])

    price_df['HT_PHASOR_inphase'] = talib.HT_PHASOR(close_p)[0]
    price_df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close_p)[1]

    return price_df

def Trend_vs_Cycle_Mode(price_df):
    close_p = np.array(price_df['Close'])

    price_df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_p)

    return price_df


def candleStickPatterns(price_df):
    
    price_df['CandleSize'] = price_df['Close'] - price_df['Open']
    price_df['CandleSize(-1)'] = price_df['CandleSize'].shift(1)
    price_df['CandleSize(-2)'] = price_df['CandleSize'].shift(2)
    price_df['CandleSize(-3)'] = price_df['CandleSize'].shift(3)
    
    return price_df