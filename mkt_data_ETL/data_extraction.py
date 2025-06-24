import pandas as pd 
import numpy as np


import requests
import json
import os


from dotenv import load_dotenv
load_dotenv()


def get_non_delisted_sp_const(token, index='GSPC.INDX', start_date="2001-01-01", end_date="2024-12-31"):

    url = f'https://eodhd.com/api/fundamentals/{index}?historical=1&from={start_date}&to={end_date}&api_token={token}&fmt=json' 
    data = requests.get(url).json() 
    
    constituints_data_all = pd.DataFrame(data['HistoricalTickerComponents']).T
    constituints_data_all = constituints_data_all[constituints_data_all['IsDelisted'] != 1]
    constituints_data_all['Code'].values

    return list(constituints_data_all['Code'].values)


def get_stock_shares_json(stock, api_key):
    url = f"https://eodhd.com/api/fundamentals/{stock}"
    params = {
        "api_token": f"{api_key}",
        "fmt": "json",
        "filter": "outstandingShares::quarterly"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
               
    return data


def build_stock_shares_df(stock, data):
    stock_shares = []
    for indx in data.keys():
        stock_shares.append({'stock':f'{stock}',
                             'date': data[indx]['dateFormatted'],
                             'shares': np.round(data[indx]['shares'],0)})
    df_stock = pd.DataFrame(stock_shares)
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock.sort_values(by='date',inplace=True)
    df_stock = df_stock[df_stock['date'] >= '2001-01-01']
    
    return df_stock


def build_all_stock_shares_data(ticker_list, api_key):
    all_stocks_shares = pd.DataFrame(columns=['stock', 'date', 'shares'])
    error_tickers = []

    for company_name in ticker_list:
        try:
            stock_json_data = get_stock_shares_json(company_name, api_key)
            stock_shares_df = build_stock_shares_df(company_name, stock_json_data)
            all_stocks_shares = pd.concat([all_stocks_shares, stock_shares_df], axis=0)
        except Exception as e:
            error_tickers.append(company_name)
            print(f"Fail processing {company_name}: {e}")

    return all_stocks_shares, error_tickers


def get_prices_eodhd(non_delisted_stocks_lst, start_date, end_date,API_KEY):
    
    prices_df = pd.DataFrame()
    for stock in non_delisted_stocks_lst:
        url = f'https://eodhd.com/api/eod/{stock}?from={start_date}&to={end_date}&period=d&api_token={API_KEY}&fmt=json'
        data = requests.get(url).json()
        
        prices_df_tmp = pd.DataFrame(data)
        prices_df_tmp['comp_name'] = stock
        
        prices_df = pd.concat([ prices_df, prices_df_tmp],axis=0)
    return prices_df


def get_SP500_eodhd(start_date, end_date,API_KEY):
    url = f'https://eodhd.com/api/eod/GSPC.INDX?from={start_date}&to={end_date}&period=d&api_token={API_KEY}&fmt=json'
    data = requests.get(url).json()
    sp500_prices_df = pd.DataFrame(data)
    sp500_prices_df['comp_name'] = 'SP500'

    return sp500_prices_df


def save_raw_data(start_date='2001-01-01',end_date='2024-12-31'):
    API_KEY    = os.environ['API_KEY']
    FILES_PATH = os.environ['FILES_PATH']

    non_delisted_stocks_lst = get_non_delisted_sp_const(API_KEY)
    stocks_shares, errors   = build_all_stock_shares_data(ticker_list=non_delisted_stocks_lst, api_key=API_KEY)
    prices_df               = get_prices_eodhd(non_delisted_stocks_lst, start_date, end_date,API_KEY)
    sp500_prices            =  get_SP500_eodhd(start_date, end_date,API_KEY)
    
    # Save List of non delisted Stocks
    with open(os.path.join(FILES_PATH, 'non_delisted_stocks_eodhd.txt'), 'w') as f:
        json.dump(non_delisted_stocks_lst, f)
    
    # Save historical stocks shares amount and prices
    stocks_shares.to_csv(os.path.join(FILES_PATH, 'shares_amout_eodhd.csv')) 
    prices_df.to_csv(os.path.join(FILES_PATH, 'prices_df_eodhd.csv'))
    sp500_prices.to_csv(os.path.join(FILES_PATH, 'sp_500_prices_eodhd.csv'))
    
    return errors

