
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import datetime
from dateutil import parser

get_ipython().magic('run C:/Users/alifa/Documents/Investment/Input_Codes/data_loading_functions.py')


# from a02_functions import TSE_price_df_extract

market = 'TSE'
timeFrame = 'D1'   # for forex: M1, M5, M30, H1, H4, D1, W1, MN

print("The current market and timeframe is: {a} and {b}".format(a=market, b= timeFrame))

if market == 'US':
    index_price_df_file_path = r'C:\Users\alifa\Desktop\Screening\US\raw_index_price'
    price_df_path = r'C:\Users\alifa\Desktop\Screening\US\raw_symbols_price'

    stochastic_screen_path = r'C:\Users\alifa\Desktop\Screening\US\stoch_reversal'

    # '---------------------------- Advanced Get path and variables ----------------
    advanced_get_daily_excel_path = r'C:\Users\alifa\Desktop\Screening\US\advanced_get\excel_daily'
    advanced_get_daily_pdf_path = r'C:\Users\alifa\Desktop\Screening\US\advanced_get\pdf_daily'

    advanced_get_weekly_excel_path = r'C:\Users\alifa\Desktop\Screening\US\advanced_get\excel_weekly'
    advanced_get_weekly_pdf_path = r'C:\Users\alifa\Desktop\Screening\US\advanced_get\pdf_weekly'

    advanced_get_screening_result_path = r'C:\Users\alifa\Desktop\Screening\US\advanced_get\screening_result'
    # --------------------------------------------------------------------------------

    candlestick_screen_path = r'C:\Users\alifa\Desktop\screening\us\candlestick_reversal'
    general_screening_result_path = r'C:\Users\alifa\Desktop\screening\us\general_screening'
    # PMA average screening
    ma_screening_path = r'C:\Users\alifa\Desktop\screening\us\moving_average'
    screening_df_all_file_dir = r'C:\Users\alifa\Desktop\screening\us\screening_all'
    symbols_beta_dir = r'C:\Users\alifa\Desktop\screening\us\beta'

    spy_file_name = os.path.join(index_price_df_file_path, 'SPY.csv')
    df_spy = pd.read_csv(spy_file_name)
    latest_excel_file_name_date = parser.parse(df_spy['Date'].values.tolist()[-1])
    latest_excel_file_name = '{a}-{b}-{c}'.format(a=latest_excel_file_name_date.year,
                                                  b='%02d' % latest_excel_file_name_date.month,
                                                  c='%02d' % latest_excel_file_name_date.day)

    print(f"US Latest date of price df loaded: {latest_excel_file_name}")

    '''
    To get today's date  ---> used to import price df
    '''
    end_us_import = datetime.datetime.today() + datetime.timedelta(days=1)
    end_us_import = end_us_import.strftime('%Y-%m-%d')
    print("The end date used in Yahoo Finance function: {a}".format(a=end_us_import))

    '''
    US list of stocks
    '''
    # load list of symbols
    list_stocks_dir = r'C:\Users\alifa\Desktop\Screening\US'
    list_stocks_file_name = 'US_Stock_Exchanges_Lists.xlsx'
    list_stocks_file_path = os.path.join(list_stocks_dir, list_stocks_file_name)

    all_stocks_df = pd.read_excel(list_stocks_file_path, sheetname='us_final_symbols_to_load')
    all_stocks_list = all_stocks_df['Symbol'].values

    symbols_beta_file_path = os.path.join(r'C:\Users\alifa\Desktop\screening\uS\beta', 'us_symbols_beta.csv')
    price_df_tech_ind_dir = r'C:\Users\alifa\Desktop\screening\us\price_df_tech_ind'

elif market == 'TSE':
    index_price_df_file_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\1Raw_Symbols_Price'
    price_df_path_source = r'D:\Investments\TSE\Raw_Symbols_Price_All'
    price_df_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\1Raw_Symbols_Price'



    # ---------------------------- PMA average screening -----------------------------------------------------------------
    ma_screening_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\2Moving_Average'    
    
    
    # '---------------------------- Advanced Get path and variables -------------------------------------------------------
    advanced_get_daily_excel_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\3Advanced_Get\excel_daily'
    advanced_get_daily_pdf_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\3Advanced_Get\pdf_daily'

    advanced_get_weekly_excel_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\3Advanced_Get\excel_weekly'
    advanced_get_weekly_pdf_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\3Advanced_Get\pdf_weekly'

    advanced_get_screening_result_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\3Advanced_Get\screening_result'
    # -----------------------------------------------------------------------------------------------------------------------

    # ---------------------------- Candlestick reversals path and variables -------------------------------------------------
    candlestick_screen_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\4Candlestick_Reversal'
    # -----------------------------------------------------------------------------------------------------------------------
    
    # '------------------------------------------ Stochastic Reversal -------------------------------------------------------    
    stochastic_screen_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\5Stoch_Reversal'
    # ------------------------------------------------------------------------------------------------------------------------    
    
    # ------------------------------------------ General Screening -------------------------------------------------------     
    general_screening_result_path = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\6General_Screening'
    # ------------------------------------------------------------------------------------------------------------------------

    
    screening_df_all_file_dir = r'C:\Users\alifa\Documents\Investment\Stocks\US_TSE_Screening\7Screening_All'
    
    
    
    symbols_beta_file_path = os.path.join(r'C:\Users\alifa\Desktop\screening\TSE\beta', 'tse_symbols_beta.csv')

    '---------------------------- Find the last trading day ----------------'
    tepix_price_df_path = price_df_path_source

    
    symbols = ['FEOLAD', 'FEMELI', 'AKHABER', 'FARS']
    last_date_list = []
    for symbol in symbols:
        tepix_file_name = os.path.join(tepix_price_df_path, '{a}_D_SH.txt'.format(a = symbol))
        columns_to_use = ['<Ticker>', '<DTYYYYMMDD>', '<TIME>', '<Open>', '<High>', '<Low>']
        df = pd.read_csv(tepix_file_name, usecols = columns_to_use)
        df.rename(columns={'<Ticker>': 'Date', '<DTYYYYMMDD>': 'Open',
                              '<TIME>': 'High', '<Open>': 'Low', '<High>': 'Close',
                              '<Low>': 'Volume'}, inplace=True)
        last_date_list.append(df['Date'].values.tolist()[-1])

    last_date = str(max(last_date_list))
    print(last_date)
    latest_excel_file_name = '{a}-{b}-{c}'.format(a = last_date[:4], b= last_date[4:6], c = last_date[-2:]) 
    print(f"TSE Latest date of price df loaded: {latest_excel_file_name}")
    
    symbols_beta_file_path = os.path.join(r'C:\Users\alifa\Desktop\screening\TSE\beta', 'tse_symbols_beta.csv')
    
    price_df_tech_ind_dir = r'C:\Users\alifa\Desktop\screening\TSE\price_df_tech_ind'
    
    # -------------------------------------- 
    port_mgm_file_path = r'C:\Users\alifa\Desktop\portfolio_mgm\tse\tse_port_mgm.xlsx'


elif (market == 'FOREX') & (timeFrame == 'M30'):
    price_df_path = r'C:\Users\alifa\Desktop\screening\Forex\M30'
    
    price_df_tech_ind_dir = r'C:\Users\alifa\Desktop\Screening\Forex\price_df_tech_ind\M30'

