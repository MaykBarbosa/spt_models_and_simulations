import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datetime import datetime


from dotenv import load_dotenv
import os
load_dotenv()

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


def get_calendar():
    
    # As datas abaixo não possuem preços e serão removidas do banco de dados
    # 2004-06-11 -> Bolsas fechadas devido ao funeral de Ronal Reagan
    # 2012-10-29 e 2012-10-30 -> Bolsas fechadas devido ao furacão Sandy
    # 2020-11-26 -> Thanks giving não mapeado no meu banco de dias úteis

    # Na data 2007-01-02 não há preços para nenhuma ações devido a algum problema não encontrado. Nessa data, os preços do dia anterior serão repetidos.


    calendario =  pd.read_csv(os.path.join(os.environ['FILES_PATH'], 'sp500\working_days_calendar.csv'), sep=',')
    calendario = calendario[(calendario['code']=='cme') & (calendario['workingday']==1)]
    calendario['date'] = pd.to_datetime(calendario['date'])
    missing_prices_dates = [ pd.to_datetime(item) for item in ['2004-06-11','2012-10-29', '2012-10-30','2020-11-26']]

    return calendario, missing_prices_dates

def get_sp500_composition(calendario, missing_prices_dates):
    #S&P composition 2014-2024
    sp500_comp = pd.read_csv(os.path.join(os.environ['FILES_PATH'], 'sp500\sP_500_comp.csv'), sep=';')
    sp500_comp['ticker'] = sp500_comp['ticker'].str.strip()
    sp500_comp['ticker'] = sp500_comp['ticker'] + ' US Equity'
    sp500_comp['date'] = pd.to_datetime(sp500_comp['date'])
    sp500_comp['is_SP500'] = True
    sp500_comp = sp500_comp[sp500_comp['date'].isin(calendario['date'])]
    sp500_comp = sp500_comp[~sp500_comp['date'].isin(missing_prices_dates)]
    
    return sp500_comp

def get_sp500_stock_data(missing_prices_dates,calendario ):
    sp500_stock_data= pd.read_csv(os.path.join(os.environ['FILES_PATH'], "sp500\SP_500_prices.csv"),index_col=0) 
    sp500_stock_data['date'] =  pd.to_datetime(sp500_stock_data['date'])
    sp500_stock_data = sp500_stock_data[sp500_stock_data['date'].isin(calendario['date'])] 
    sp500_stock_data = sp500_stock_data[~sp500_stock_data['date'].isin(missing_prices_dates) ] #Removendo datas
    
    return sp500_stock_data

def get_sp500_spx_index(missing_prices_dates,calendario ):
    spx_index  = pd.read_csv(os.path.join(os.environ['FILES_PATH'], "sp500\SP_500_index.csv"), index_col=0)
    spx_index.columns = ['date', 'ticker','PX_LAST','PX_VOLUME','PX_OPEN','PX_HIGH','PX_LOW','CUR_MKT_CAP']


    spx_index = spx_index[['date', 'PX_LAST']]
    spx_index['date'] = pd.to_datetime(spx_index['date']) 


    spx_index = spx_index[spx_index['date'].isin(calendario['date'])] 
    spx_index = spx_index[~spx_index['date'].isin(missing_prices_dates)]  #removendo datas


    spx_index.set_index("date",inplace=True)
    spx_index.rename(columns={'PX_LAST':'spx_index'},inplace=True)
    spx_index.index = pd.to_datetime(spx_index.index) 

    return spx_index


def get_selected_stocks(sp500_stock_data):
    """
        Seleciona as 100 ações com maior capitalização de mercado no inicio do periodo e 
        retorno 2 dataframes: 1 com as capitalizações de mercado e outro com os preços dessas ações.

    """
    sp_prices_df  = sp500_stock_data.pivot(columns='ticker', index='date', values='PX_LAST')
    sp_mkt_cap_df = sp500_stock_data.pivot(columns='ticker', index='date', values='CUR_MKT_CAP')

    #Ajustando valores da data 2007-01-02 e igualando aos preços do dia anterior
    sp_mkt_cap_df.loc[pd.to_datetime('2007-01-02')] = sp_mkt_cap_df.loc[pd.to_datetime('2006-12-29')] 
    sp_prices_df.loc[pd.to_datetime('2007-01-02')]  = sp_prices_df.loc[pd.to_datetime('2006-12-29')] 


    # Removendo companias problemáticas
    sp_prices_df.drop(columns=['MOS US Equity', 'NRG US Equity'],inplace=True)
    sp_mkt_cap_df.drop(columns=['MOS US Equity', 'NRG US Equity'],inplace=True)

    sp_mkt_cap_df = sp_mkt_cap_df.ffill()
    sp_prices_df  = sp_prices_df.ffill()

    # Seleciona somente ações que estão presentes em todo o período
    complete_stocks = sp_prices_df.columns[sp_prices_df.isna().sum() == 0].tolist()
    sp_prices_complete_df = sp_prices_df[complete_stocks]
    sp_mkt_cap_complete_df = sp_mkt_cap_df[complete_stocks]

    #Seleciona top 100 ações no início do período
    top_100_stocks_mkt_cap_list = sp_mkt_cap_complete_df.iloc[0].nlargest(100).index.to_list()

    top_100_mkt_cap_prices_df = sp_prices_complete_df[top_100_stocks_mkt_cap_list]
    top_100_mkt_cap_df        = sp_mkt_cap_complete_df[top_100_stocks_mkt_cap_list]

    # Alinha as colunas dos dataframes
    top_100_mkt_cap_df = top_100_mkt_cap_df[top_100_mkt_cap_prices_df.columns]

    return top_100_mkt_cap_df, top_100_mkt_cap_prices_df


def get_raw_data(): 

    calendario, missing_prices_dates = get_calendar()
    sp500_comp                       = get_sp500_composition(calendario, missing_prices_dates)
    spx_index                        = get_sp500_spx_index(missing_prices_dates, calendario)
    sp500_stock_data                 = get_sp500_stock_data(missing_prices_dates, calendario)

    return sp500_comp, spx_index, sp500_stock_data




# BBG
#     BBG_HIST_FIELDS = {'PX_LAST':'px_close',
#                 'PX_VOLUME':'volume',
#                 'PX_OPEN':'px_open',
#                 'PX_HIGH':'px_high',
#                 'PX_LOW':'px_low',
#                 'CUR_MKT_CAP':'market_cap',
#                 'EQY_FLOAT':'free_float',
#                 'EQY_FREE_FLOAT_PCT':'free_float_pct',
#                 'EQY_SH_OUT':'shares_outstanding'}

# # bbg_data = blp.bdh(sp500_ticker_list, list(BBG_HIST_FIELDS.keys()),'2004-01-01', '2024-12-31', CshAdjNormal=True, CshAdjAbnormal=True, CapChg=True, Currency='USD',timeout=100000)
# # # transform data
# # data_transformed = bbg_data.copy()
# # data_transformed = data_transformed.stack(level=0).reset_index()
# # data_transformed.columns = ['date', 'ticker','PX_LAST','PX_VOLUME','PX_OPEN','PX_HIGH','PX_LOW','CUR_MKT_CAP','EQY_FLOAT','EQY_FREE_FLOAT_PCT','EQY_SH_OUT']
# # data_transformed.sort_values(by='date', ascending=True, inplace=True)
# # data_transformed.to_csv(os.path.join(os.environ['FILES_PATH'], 'sp500\SP_500_prices.csv'))

# # spx_index = blp.bdh(['SPX Index'], list(BBG_HIST_FIELDS.keys()),'2004-01-01', '2024-12-31', CshAdjNormal=True, CshAdjAbnormal=True, CapChg=True, Currency='USD',timeout=100000)
# # spx_index = spx_index.stack(level=0).reset_index()
# # spx_index.to_csv(os.path.join(os.environ['FILES_PATH'], "sp500\SP_500_index.csv"))