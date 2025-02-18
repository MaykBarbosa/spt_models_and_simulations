import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

#Neural Nets
import tensorflow as tf


#Bloomberg python library. Is necessary to install bloomberg BLP API and have access to a bloomberg terminal
from xbbg import blp

from dotenv import load_dotenv
import os
load_dotenv()

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")




#S&P composition 2014-2024
sp500_comp = pd.read_csv(os.path.join(os.environ['FILES_PATH'], 'sp500\sP_500_comp.csv'), sep=';')
sp500_comp['ticker'] = sp500_comp['ticker'].str.strip()
sp500_comp['ticker'] = sp500_comp['ticker'] + ' US Equity'
sp500_comp['date'] = pd.to_datetime(sp500_comp['date'])
sp500_comp['is_SP500'] = True
sp500_ticker_list = sp500_comp['ticker'].unique()
sp500_comp.head(3)


BBG_HIST_FIELDS = {'PX_LAST':'px_close',
                'PX_VOLUME':'volume',
                'PX_OPEN':'px_open',
                'PX_HIGH':'px_high',
                'PX_LOW':'px_low',
                'CUR_MKT_CAP':'market_cap',
                'EQY_FLOAT':'free_float',
                'EQY_FREE_FLOAT_PCT':'free_float_pct',
                'EQY_SH_OUT':'shares_outstanding'}




sp500_prices= pd.read_csv(os.path.join(os.environ['FILES_PATH'], "sp500\SP_500_prices.csv"),index_col=0) 
sp500_prices['date'] =  pd.to_datetime(sp500_prices['date'])
sp500_prices.head(3)


spx_index  = pd.read_csv(os.path.join(os.environ['FILES_PATH'], "sp500\SP_500_index.csv"), index_col=0)
spx_index.columns = ['date', 'ticker','PX_LAST','PX_VOLUME','PX_OPEN','PX_HIGH','PX_LOW','CUR_MKT_CAP']


spx_index = spx_index[['date', 'PX_LAST']]
spx_index.set_index("date",inplace=True)
spx_index.rename(columns={'PX_LAST':'spx_index'},inplace=True)
spx_index.index = pd.to_datetime(spx_index.index) 


#Ações consideradas para trabalhar com banco de dados balanceado
stocks = ['CCL US Equity','KO US Equity','ITW US Equity','WAT US Equity','PWR US Equity','MS US Equity','HAL US Equity','HSY US Equity','BF/B US Equity',
          'HIG US Equity','GRMN US Equity','LLY US Equity','TRV US Equity','MDLZ US Equity','SWK US Equity','ACN US Equity','BRK/B US Equity','UHS US Equity',
          'AN US Equity','CB US Equity','COST US Equity','ETN US Equity','HON US Equity','IT US Equity','MSI US Equity','AMGN US Equity','BLK US Equity',
          'CSCO US Equity','CVX US Equity','FIS US Equity','GEN US Equity','JNJ US Equity','K US Equity','MTB US Equity','TAP US Equity','WAB US Equity',
          'XRX US Equity','A US Equity','AA US Equity','AAP US Equity','AAPL US Equity','ABT US Equity','ACGL US Equity','ADBE US Equity','ADI US Equity',
          'ADM US Equity','ADP US Equity','ADSK US Equity','AEE US Equity','AEP US Equity','AES US Equity','AFL US Equity','AIG US Equity','AIV US Equity',
          'AJG US Equity','AKAM US Equity','ALB US Equity','ALGN US Equity','ALK US Equity','ALL US Equity','AMAT US Equity','AMD US Equity','AME US Equity',
          'AMG US Equity','AMT US Equity','AMZN US Equity','ANSS US Equity','AON US Equity','AOS US Equity','APA US Equity','APD US Equity','APH US Equity',
          'ARE US Equity','ATI US Equity','ATO US Equity','AVB US Equity','AVY US Equity','AXON US Equity','AXP US Equity','AYI US Equity','AZO US Equity',
          'BA US Equity','BAC US Equity','BALL US Equity','BAX US Equity','BBWI US Equity','BBY US Equity','BDX US Equity','BEN US Equity','BFH US Equity',
          'BG US Equity','BIIB US Equity','BIO US Equity','BK US Equity','BKNG US Equity','BKR US Equity','BMY US Equity','BRO US Equity','BSX US Equity',
          'BWA US Equity','BXP US Equity','C US Equity','CAG US Equity','CAH US Equity','CAT US Equity','CCEP US Equity','CCI US Equity','CDNS US Equity',
          'CHD US Equity','CHRW US Equity','CI US Equity','CINF US Equity','CL US Equity','CLF US Equity','CLX US Equity','CMA US Equity','CMCSA US Equity',
          'CME US Equity','CMI US Equity','CMS US Equity','CNC US Equity','CNP US Equity','CNX US Equity','COF US Equity','COO US Equity',
          'COP US Equity','COR US Equity','CPB US Equity','CPRT US Equity','CPT US Equity','CRL US Equity','CSGP US Equity','CSX US Equity','CTAS US Equity',
          'CTRA US Equity','CTSH US Equity','CVS US Equity','D US Equity','DD US Equity','DE US Equity','DECK US Equity','DGX US Equity',
          'DHI US Equity','DHR US Equity','DIS US Equity','DLTR US Equity','DOC US Equity','DOV US Equity','DRI US Equity','DTE US Equity',
          'DUK US Equity','DVA US Equity','DVN US Equity','EA US Equity','EBAY US Equity','ECL US Equity','ED US Equity','EFX US Equity','EG US Equity',
          'EIX US Equity','EL US Equity','ELV US Equity','EMN US Equity','EMR US Equity','EOG US Equity','EQIX US Equity','EQR US Equity','EQT US Equity','ES US Equity',
          'ESS US Equity','ETR US Equity','EVRG US Equity','EW US Equity','EXC US Equity','EXPD US Equity','F US Equity','FAST US Equity','FCX US Equity',
          'FDS US Equity','FDX US Equity','FE US Equity','FFIV US Equity','FI US Equity','FICO US Equity','FITB US Equity','FL US Equity','FLR US Equity',
          'FLS US Equity','FMC US Equity','FOSL US Equity','FRT US Equity','FTI US Equity','GAP US Equity','GD US Equity','GE US Equity','GHC US Equity',
          'GILD US Equity','GIS US Equity','GL US Equity','GLW US Equity','GME US Equity','GPC US Equity','GPN US Equity','GS US Equity','GT US Equity',
          'GWW US Equity','HAS US Equity','HBAN US Equity','HD US Equity','HES US Equity','HOG US Equity','HOLX US Equity','HP US Equity','HPQ US Equity',
          'HRB US Equity','HRL US Equity','HSIC US Equity','HST US Equity','HUBB US Equity','HUM US Equity','IBM US Equity','IDXX US Equity','IEX US Equity',
          'IFF US Equity','ILMN US Equity','INCY US Equity','INTC US Equity','INTU US Equity','IP US Equity','IPG US Equity','IRM US Equity','ISRG US Equity',
          'IVZ US Equity','J US Equity','JBHT US Equity','JBL US Equity','JCI US Equity','JEF US Equity','JKHY US Equity','JNPR US Equity','JPM US Equity',
          'JWN US Equity','KEY US Equity','KIM US Equity','KLAC US Equity','KMB US Equity','KMX US Equity','KR US Equity','KSS US Equity','L US Equity',
          'LEG US Equity','LEN US Equity','LHX US Equity','LIN US Equity','LKQ US Equity','LMT US Equity','LNC US Equity','LNT US Equity','LOW US Equity',
          'LRCX US Equity','LUMN US Equity','LUV US Equity','M US Equity','MAA US Equity','MAC US Equity','MAR US Equity','MAS US Equity','MAT US Equity','MCD US Equity',
          'MCHP US Equity','MCK US Equity','MCO US Equity','MDT US Equity','MET US Equity','MGM US Equity','MHK US Equity','MKC US Equity','MLM US Equity','MMC US Equity',
          'MMM US Equity','MNST US Equity','MO US Equity','MOH US Equity','MRK US Equity','MSFT US Equity','MTD US Equity','MU US Equity','MUR US Equity','NBR US Equity',
          'NDAQ US Equity','NDSN US Equity','NEE US Equity','NEM US Equity','NFLX US Equity','NI US Equity','NKE US Equity','NKTR US Equity','NOC US Equity','NOV US Equity',
          'NRG US Equity','NSC US Equity','NTAP US Equity','NTRS US Equity','NUE US Equity','NVDA US Equity','NVR US Equity','NWL US Equity','O US Equity','ODFL US Equity',
          'OI US Equity','OKE US Equity','OMC US Equity','ON US Equity','ORCL US Equity','ORLY US Equity','OXY US Equity','PARA US Equity','PAYX US Equity','PBI US Equity',
          'PCAR US Equity','PCG US Equity','PDCO US Equity','PEG US Equity','PENN US Equity','PEP US Equity','PFE US Equity','PFG US Equity','PG US Equity','PGR US Equity',
          'PH US Equity','PHM US Equity','PKG US Equity','PLD US Equity','PNC US Equity','PNR US Equity','PNW US Equity','POOL US Equity','PPG US Equity','PPL US Equity',
          'PRGO US Equity','PRU US Equity','PSA US Equity','PTC US Equity','PVH US Equity','QCOM US Equity','R US Equity','RCL US Equity','REG US Equity','REGN US Equity',
          'RF US Equity','RHI US Equity','RIG US Equity','RJF US Equity','RL US Equity','RMD US Equity','ROK US Equity','ROL US Equity','ROP US Equity','ROST US Equity',
          'RRC US Equity','RSG US Equity','RTX US Equity','RVTY US Equity','SBAC US Equity','SBUX US Equity','SCHW US Equity','SEE US Equity','SHW US Equity','SIG US Equity',
          'SJM US Equity','SLB US Equity','SLG US Equity','SLM US Equity','SNA US Equity','SNPS US Equity','SO US Equity','SPG US Equity','SPGI US Equity','SRE US Equity',
          'STE US Equity','STLD US Equity','STT US Equity','STX US Equity','STZ US Equity','SWKS US Equity','SYK US Equity','SYY US Equity','T US Equity','TDY US Equity',
          'TECH US Equity','TER US Equity','TFC US Equity','TFX US Equity','TGNA US Equity','TGT US Equity','THC US Equity','TJX US Equity',
          'TMO US Equity','TPR US Equity','TRMB US Equity','TROW US Equity','TSCO US Equity','TSN US Equity','TT US Equity','TTWO US Equity','TXN US Equity','TXT US Equity',
          'TYL US Equity','UDR US Equity','UNH US Equity','UNM US Equity','UNP US Equity','UPS US Equity','URBN US Equity','URI US Equity','USB US Equity','VFC US Equity',
          'VLO US Equity','VMC US Equity','VNO US Equity','VRSN US Equity','VRTX US Equity','VTR US Equity','VTRS US Equity','VZ US Equity','WBA US Equity','WDC US Equity',
          'WEC US Equity','WELL US Equity','WFC US Equity','WHR US Equity','WM US Equity','WMB US Equity','WMT US Equity','WRB US Equity','WST US Equity','WTW US Equity',
          'WY US Equity','WYNN US Equity','X US Equity','XEL US Equity','XOM US Equity','XRAY US Equity','YUM US Equity','ZBH US Equity','ZBRA US Equity','ZION US Equity',
          'MOS US Equity']





# Assuming sp500_prices is your DataFrame with S&P 500 prices
sp_prices_df = sp500_prices.pivot(columns='ticker', index='date', values='PX_LAST')
balanced_sp_prices_df = sp_prices_df[stocks].copy()
balanced_sp_prices_df = balanced_sp_prices_df.ffill()

p=0.5
# Farei o treinamento da rede utilizando pesos diários para ter uma quantidade maior de pontos.


mu_t_df = balanced_sp_prices_df.div(balanced_sp_prices_df.sum(axis=1), axis=0)



# Calculate daily returns
balanced_stocks_returns = np.log(balanced_sp_prices_df / balanced_sp_prices_df.shift(1))
sp500_idx_returns =  np.log(spx_index / spx_index.shift(1))

mu_t_df = mu_t_df.reindex(balanced_stocks_returns.index, method='ffill')

# Calculate the monthly rebalanced market portfolio returns
mkt_return_t_df = (mu_t_df * balanced_stocks_returns.shift(1)).sum(axis=1)






class PortfolioModel(tf.keras.Model):
    def __init__(self, network, stock_returns):
        super().__init__()
        self.network = network
        self.stock_returns = tf.convert_to_tensor(stock_returns, dtype=tf.float32)
        
    def train_step(self, data):
        X_input, y_true = data
        
        with tf.GradientTape(persistent=True) as tape:
            G = self.network(X_input)
            
            # Compute gradient of log(G) with respect to INPUTS (X)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(X_input)
                log_G = tf.math.log(self.network(X_input))
                
            grad_log_G_X = inner_tape.gradient(log_G, X_input)
            
            # Compute portfolio weights (ensure gradient flow)
            sum_term = tf.reduce_sum(X_input * grad_log_G_X, axis=1, keepdims=True)
            pi_weights = X_input * (grad_log_G_X + 1 - sum_term)
            
            # Calculate portfolio returns
            batch_returns = tf.gather(self.stock_returns, tf.range(tf.shape(X_input)[0]))
            port_returns = tf.reduce_sum(pi_weights * batch_returns, axis=1)
            
            # Custom loss calculation
            loss = -tf.reduce_mean(port_returns - tf.squeeze(y_true))
            

            
        # Compute gradients and update weights
        trainable_vars = self.network.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        tf.print("X_input:", X_input)
        tf.print("grad_log_G_X:", grad_log_G_X)
        tf.print("pi_weights:", pi_weights)
        tf.print("batch_returns:", batch_returns)
        tf.print("port_returns:", port_returns)
        
        print("breakpoint")

        return {'loss': loss}



Y = tf.convert_to_tensor(mkt_return_t_df.iloc[2:].values, dtype=tf.float32)
X = tf.convert_to_tensor(mu_t_df.iloc[2:].values, dtype=tf.float32)

# Usage example:
# 1. First build the base network
input_dim = X.shape[1] # Number of features per sample
base_network = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 2. Prepare stock returns data (should align with training samples)
stock_returns = tf.convert_to_tensor(balanced_stocks_returns.iloc[2:].values, dtype=tf.float32)

# 3. Create custom model
model = PortfolioModel(base_network, stock_returns)
model.compile(optimizer=tf.keras.optimizers.Adam())

# 4. Prepare data (X should be indexed to match stock_returns)
train_dataset =tf.data.Dataset.from_tensor_slices((X, Y)).batch(32)

# 5. Train using fit() - note we pass [X_train, y_train] as x argument
model.fit(train_dataset, epochs=50)



