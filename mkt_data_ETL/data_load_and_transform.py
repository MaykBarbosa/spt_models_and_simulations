import pandas as pd
import datetime as dt


import os
import json

from dotenv import load_dotenv
load_dotenv()


from data_extraction import save_raw_data



# Uncomment and execute the function below only if you want to extract the data
# erros= save_raw_data()

def get_calendar(is_wirkingday=True):
    
    # As datas abaixo não possuem preços e serão removidas do banco de dados
    # 2004-06-11 -> Bolsas fechadas devido ao funeral de Ronald Reagan
    # 2012-10-29 e 2012-10-30 -> Bolsas fechadas devido ao furacão Sandy
    # 2020-11-26 -> Thanks giving não mapeado no meu banco de dias úteis

    # Na data 2007-01-02 não há preços para nenhuma ações devido a algum problema não encontrado. Nessa data, os preços do dia anterior serão repetidos.


    calendar =  pd.read_csv(os.path.join(os.environ['FILES_PATH'], 'working_days_calendar.csv'), sep=',')
    if is_wirkingday:
        calendar = calendar[(calendar['code']=='cme') & (calendar['workingday']==1)]
    else:
        calendar = calendar[calendar['code']=='cme']
    
    calendar['date'] = pd.to_datetime(calendar['date'])
    #missing_prices_dates = [ pd.to_datetime(item) for item in ['2004-06-11','2012-10-29', '2012-10-30','2020-11-26']]

    return calendar


def load_and_transform_shares(files_path):
    
    stocks_shares_amount_df= pd.read_csv(os.path.join(files_path, "shares_amout_eodhd.csv"))
    stock_shares_pivot_df = stocks_shares_amount_df.pivot(index='date', columns='stock', values='shares')
    stock_shares_pivot_df.index = pd.to_datetime(stock_shares_pivot_df.index)
    stock_shares_pivot_df = stock_shares_pivot_df.loc[:pd.to_datetime('2025-01-01')]

    # Get all days in cme calendar
    calendar = get_calendar(is_wirkingday=False)
    calendar=calendar[(calendar['date'] >= pd.to_datetime('2001-01-01')) & (calendar['date'] <= pd.to_datetime('2024-12-31'))]

    # Reindex with all dates
    stock_shares_pivot_df = stock_shares_pivot_df.reindex(calendar['date']).ffill()

    return stock_shares_pivot_df

def load_and_transform_prices(files_path):

    stock_prices_df= pd.read_csv(os.path.join(files_path, "prices_df_eodhd.csv"))
    prices_df = prices_df[['date','adjusted_close', 'comp_name']].pivot(index='date', columns='comp_name', values='adjusted_close')
    stock_prices_df.index = pd.to_datetime(stock_prices_df.index)
    stock_prices_df = stock_prices_df.loc[:pd.to_datetime('2025-01-01')]

    sp500_prices_df = pd.read_csv(os.path.join(files_path, "sp_500_prices_eodhd.csv"))
    sp500_prices_df = sp500_prices_df[['date','adjusted_close', 'comp_name']].pivot(index='date', columns='comp_name', values='adjusted_close')
    sp500_prices_df.index = pd.to_datetime(sp500_prices_df.index)
    sp500_prices_df = sp500_prices_df.loc[:pd.to_datetime('2025-01-01')]

    return stock_prices_df


def get_data(start_date = '2004-01-01', end_date = '2024-12-31'):
    FILES_PATH = os.environ['FILES_PATH']

    stock_shares_amount_df            = load_and_transform_shares(files_path=FILES_PATH)
    stock_prices_df, sp500_prices_df  = load_and_transform_prices(files_path=FILES_PATH)

    # Columns interserction
    cols_intersec     = list(set(stock_shares_amount_df.columns).intersection(set(stock_prices_df.columns)))
    removed_companies = list(set(stock_shares_amount_df.columns).union(set(stock_prices_df.columns)) - set(stock_shares_amount_df.columns).intersection(set(stock_prices_df.columns)))

    # Fixing only columns that exists in both dataframes
    stock_prices_df        = stock_prices_df[cols_intersec]
    stock_shares_amount_df = stock_prices_df[cols_intersec]

    # Reindex the shares df with the prices_df to align the dataframe index because prices are our reference
    stock_shares_amount_df = stock_shares_amount_df.reindex(stock_prices_df.index)

    # Creating the dataframe with companies market capitalization
    mkt_cap_df = stock_prices_df*stock_shares_amount_df


    # Set startdate and end date for all dataframes
    stock_prices_df        = stock_prices_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    stock_shares_amount_df = stock_shares_amount_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    mkt_cap_df             = mkt_cap_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

    return stock_prices_df, stock_shares_amount_df, mkt_cap_df, sp500_prices_df, removed_companies

