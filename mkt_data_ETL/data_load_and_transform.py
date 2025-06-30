import pandas as pd
import datetime as dt


import os
import json

from dotenv import load_dotenv
load_dotenv()


from mkt_data_ETL.data_extraction import save_raw_data


def get_calendar(is_workingday=True):

    calendar =  pd.read_csv(os.path.join(os.environ['FILES_PATH'], 'working_days_calendar.csv'), sep=',')
    if is_workingday:
        calendar = calendar[(calendar['code']=='cme') & (calendar['workingday']==1)]
    else:
        calendar = calendar[calendar['code']=='cme']
    
    calendar['date'] = pd.to_datetime(calendar['date'])
    #missing_prices_dates = [ pd.to_datetime(item) for item in ['2004-06-11','2012-10-29', '2012-10-30','2020-11-26']]

    return calendar


def load_and_transform_shares(files_path):
    
    stocks_shares_amount_df= pd.read_csv(os.path.join(files_path, "shares_amout_eodhd.csv"), index_col=0)
    stock_shares_pivot_df = stocks_shares_amount_df.pivot(index='date', columns='stock', values='shares')
    stock_shares_pivot_df.index = pd.to_datetime(stock_shares_pivot_df.index)
    stock_shares_pivot_df = stock_shares_pivot_df.sort_index()
    stock_shares_pivot_df = stock_shares_pivot_df.loc[:pd.to_datetime('2025-01-01')]

    
    # Removing outliers
    prev = stock_shares_pivot_df.shift(1)
    next_ = stock_shares_pivot_df.shift(-1)
    current = stock_shares_pivot_df.copy()

    outlier_up = (current > 2 * prev) & (current > 2 * next_)
    outlier_down = (2*current < prev) & (2*current < next_)
    outliers_bool = outlier_up | outlier_down
    stock_shares_pivot_df[outliers_bool] = pd.NA

    
    # Get all days in cme calendar
    calendar = get_calendar(is_workingday=False)
    calendar=calendar[(calendar['date'] >= pd.to_datetime('2001-01-01')) & (calendar['date'] <= pd.to_datetime('2024-12-31'))]

    # Reindex with all dates
    stock_shares_pivot_df = stock_shares_pivot_df.reindex(calendar['date']).ffill()

    return stock_shares_pivot_df

def load_and_transform_prices(files_path):

    stock_prices_df= pd.read_csv(os.path.join(files_path, "prices_df_eodhd.csv"), index_col=0)
    stock_prices_df = stock_prices_df[['date','adjusted_close', 'comp_name']].pivot(index='date', columns='comp_name', values='adjusted_close')
    stock_prices_df.index = pd.to_datetime(stock_prices_df.index)
    stock_prices_df = stock_prices_df.sort_index()
    stock_prices_df = stock_prices_df.loc[:pd.to_datetime('2025-01-01')]

    sp500_prices_df = pd.read_csv(os.path.join(files_path, "sp_500_prices_eodhd.csv"))
    sp500_prices_df = sp500_prices_df[['date','adjusted_close', 'comp_name']].pivot(index='date', columns='comp_name', values='adjusted_close')
    sp500_prices_df.index = pd.to_datetime(sp500_prices_df.index)
    sp500_prices_df = sp500_prices_df.sort_index()
    sp500_prices_df = sp500_prices_df.loc[:pd.to_datetime('2025-01-01')]

    return stock_prices_df, sp500_prices_df


def get_data(start_date = '2004-01-01', end_date = '2024-12-31'):
    FILES_PATH = os.environ['FILES_PATH']

    # Uncomment and execute the function below only if you want to extract the data
    # erros=save_raw_data()

    stock_shares_amount_df            = load_and_transform_shares(files_path=FILES_PATH)
    stock_prices_df, sp500_prices_df  = load_and_transform_prices(files_path=FILES_PATH)

    # Columns interserction
    cols_intersec     = list(set(stock_shares_amount_df.columns).intersection(set(stock_prices_df.columns)))
    removed_companies = list(set(stock_shares_amount_df.columns).union(set(stock_prices_df.columns)) - set(stock_shares_amount_df.columns).intersection(set(stock_prices_df.columns)))

    # Fixing only columns that exists in both dataframes
    stock_prices_df        = stock_prices_df[cols_intersec]
    stock_shares_amount_df = stock_shares_amount_df[cols_intersec]

    # Reindex the shares df with the prices_df to align the dataframe index because prices are our reference
    stock_shares_amount_df = stock_shares_amount_df.reindex(stock_prices_df.index)

    # Creating the dataframe with companies market capitalization
    mkt_cap_df = stock_prices_df*stock_shares_amount_df


    # Set startdate and end date for all dataframes
    stock_prices_df        = stock_prices_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    stock_shares_amount_df = stock_shares_amount_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    mkt_cap_df             = mkt_cap_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    sp500_prices_df        = sp500_prices_df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]

    return stock_prices_df, stock_shares_amount_df, mkt_cap_df, sp500_prices_df, removed_companies

def get_top_mkt_cap_stocks(stock_prices_df, stock_mkt_cap_df):
    """
        Selects the top 100 stocks by market capitalization at the start of the period and returns two dataframes: 
        one with market capitalizations and the other with the prices of those stocks.
    """
    
    # Select only stocks that are present in the whole period
    complete_stocks = stock_prices_df.columns[stock_prices_df.isna().sum() == 0].tolist()
    sp_prices_complete_df = stock_prices_df[complete_stocks]
    sp_mkt_cap_complete_df = stock_mkt_cap_df[complete_stocks]

    # Select the top 100 stocks by market capitalization at the start of the period
    top_100_stocks_mkt_cap_list = sp_mkt_cap_complete_df.iloc[0].nlargest(100).index.to_list()
    top_100_mkt_cap_prices_df = sp_prices_complete_df[top_100_stocks_mkt_cap_list]
    top_100_mkt_cap_df        = sp_mkt_cap_complete_df[top_100_stocks_mkt_cap_list]

    # Align the columns of the dataframes
    top_100_mkt_cap_df = top_100_mkt_cap_df[top_100_mkt_cap_prices_df.columns]

    return top_100_mkt_cap_df, top_100_mkt_cap_prices_df
