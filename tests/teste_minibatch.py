import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

#Neural Nets
import tensorflow as tf


#Bloomberg python library. Is necessary to install bloomberg BLP API and have access to a bloomberg terminal
#from xbbg import blp

from dotenv import load_dotenv
import os
load_dotenv()

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

BBG_HIST_FIELDS = {'PX_LAST':'px_close',
                'PX_VOLUME':'volume',
                'PX_OPEN':'px_open',
                'PX_HIGH':'px_high',
                'PX_LOW':'px_low',
                'CUR_MKT_CAP':'market_cap',
                'EQY_FLOAT':'free_float',
                'EQY_FREE_FLOAT_PCT':'free_float_pct',
                'EQY_SH_OUT':'shares_outstanding'}

#Calendário
calendario =  pd.read_csv(os.path.join(os.environ['FILES_PATH'], 'sp500\working_days_calendar.csv'), sep=',')
calendario = calendario[(calendario['code']=='cme') & (calendario['workingday']==1)]
calendario['date'] = pd.to_datetime(calendario['date'])


missing_prices_dates = [ pd.to_datetime(item) for item in ['2004-06-11','2012-10-29', '2012-10-30','2020-11-26']]


#S&P composition 2014-2024
sp500_comp = pd.read_csv(os.path.join(os.environ['FILES_PATH'], 'sp500\sP_500_comp.csv'), sep=';')
sp500_comp['ticker'] = sp500_comp['ticker'].str.strip()
sp500_comp['ticker'] = sp500_comp['ticker'] + ' US Equity'
sp500_comp['date'] = pd.to_datetime(sp500_comp['date'])
sp500_comp['is_SP500'] = True
sp500_ticker_list = sp500_comp['ticker'].unique()
sp500_comp = sp500_comp[sp500_comp['date'].isin(calendario['date'])]
sp500_comp = sp500_comp[~sp500_comp['date'].isin(missing_prices_dates)]


sp500_stock_data= pd.read_csv(os.path.join(os.environ['FILES_PATH'], "sp500\SP_500_prices.csv"),index_col=0) 
sp500_stock_data['date'] =  pd.to_datetime(sp500_stock_data['date'])
sp500_stock_data = sp500_stock_data[sp500_stock_data['date'].isin(calendario['date'])] 
sp500_stock_data = sp500_stock_data[~sp500_stock_data['date'].isin(missing_prices_dates) ] #Removendo datas


spx_index  = pd.read_csv(os.path.join(os.environ['FILES_PATH'], "sp500\SP_500_index.csv"), index_col=0)
spx_index.columns = ['date', 'ticker','PX_LAST','PX_VOLUME','PX_OPEN','PX_HIGH','PX_LOW','CUR_MKT_CAP']


spx_index = spx_index[['date', 'PX_LAST']]
spx_index['date'] = pd.to_datetime(spx_index['date']) 


spx_index = spx_index[spx_index['date'].isin(calendario['date'])] 
spx_index = spx_index[~spx_index['date'].isin(missing_prices_dates)]  #removendo datas


spx_index.set_index("date",inplace=True)
spx_index.rename(columns={'PX_LAST':'spx_index'},inplace=True)
spx_index.index = pd.to_datetime(spx_index.index) 


sp_prices_df  = sp500_stock_data.pivot(columns='ticker', index='date', values='PX_LAST')
sp_mkt_cap_df = sp500_stock_data.pivot(columns='ticker', index='date', values='CUR_MKT_CAP')

#Ajustando valores da data 2007-01-02 e igualando aos preços do dia anterior
sp_mkt_cap_df.loc[pd.to_datetime('2007-01-02')] = sp_mkt_cap_df.loc[pd.to_datetime('2006-12-29')] 
sp_prices_df.loc[pd.to_datetime('2007-01-02')]  = sp_prices_df.loc[pd.to_datetime('2006-12-29')] 

# Mercado Fechado
complete_stocks_prices = sp_prices_df.columns[sp_prices_df.isna().sum() == 0].tolist()
complete_stocks_mkt_cap = sp_mkt_cap_df.columns[sp_mkt_cap_df.isna().sum() == 0].tolist()

diff_prices_and_mkt_cap = list(set(complete_stocks_prices) - set(complete_stocks_mkt_cap))
print("Diferenças entre os dois conjuntos")
print(diff_prices_and_mkt_cap)

sp_prices_df.drop(columns=['MOS US Equity', 'NRG US Equity'],inplace=True)
sp_mkt_cap_df.drop(columns=['MOS US Equity', 'NRG US Equity'],inplace=True)

sp_mkt_cap_df = sp_mkt_cap_df.ffill()
sp_prices_df  = sp_prices_df.ffill()

#Seleciona somente ações que stão presentes em todo o período
complete_stocks = sp_prices_df.columns[sp_prices_df.isna().sum() == 0].tolist()
sp_prices_complete_df = sp_prices_df[complete_stocks]
sp_mkt_cap_complete_df = sp_mkt_cap_df[complete_stocks]

#Seleciona top 100 ações no início do período
top_100_stocks_mkt_cap_list = sp_mkt_cap_complete_df.iloc[0].nlargest(100).index.to_list()

top_100_mkt_cap_prices_df = sp_prices_complete_df[top_100_stocks_mkt_cap_list]
top_100_mkt_cap_df        = sp_mkt_cap_complete_df[top_100_stocks_mkt_cap_list]

# Alinha as colunas dos dataframes
top_100_mkt_cap_df = top_100_mkt_cap_df[top_100_mkt_cap_prices_df.columns]


#Cálculo dos pesos dos portfolios 

# Calculate daily returns
stocks_returns    = np.log(top_100_mkt_cap_prices_df / top_100_mkt_cap_prices_df.shift(1))
sp500_idx_returns =  np.log(spx_index / spx_index.shift(1))

def periodic_rebalance(df, p=None, rebalance_fred = 'w'):

    # Resample to monthly frequency, taking the last observation of each wperiod
    resampled_df = df.resample(rebalance_fred).last()
    
    if p is not None:
        # For DWP: apply power transformation
        weights = resampled_df.apply(lambda x: x ** p, axis=0)
    else:
        # For market portfolio: no transformation
        weights = resampled_df.copy()

    # Normalize weights to sum to 1
    weights = weights.div(weights.sum(axis=1), axis=0)
    
    weights_daily = weights.resample('D').ffill()
    
    return weights_daily

mkt_portfolio_weights = top_100_mkt_cap_df.div(top_100_mkt_cap_df.sum(axis=1),axis=0)


#DWP portfolio
p = 0.5
power_p_transform = mkt_portfolio_weights.apply(lambda mu: mu**p,axis=0 )
dwp_portfolio_weights = power_p_transform.div(power_p_transform.sum(axis=1),axis=0 )


# Calculo dos retornos dos portfolios
# O shift nos pesos considera que as compras das ações com os pesos calculados em D0 
# sejam efetuadas no final do dia, recebendo assim o retorno das ações no fechamento do dia seguinte

#Retorno do portfólio de mercado
mkt_return = (mkt_portfolio_weights.shift(1) * stocks_returns).sum(axis=1)

# Retorno do Portfólio DWP
dwp_return = (dwp_portfolio_weights.shift(1) * stocks_returns).sum(axis=1)


##################################################
# # MONTAGEM DAS AMOSTRAS DE TREINO E DE TESTE # #
##################################################


# Copiando os dados gerados anteriormente 
# Pesos do Portfolio de mercado
mu_t_df = mkt_portfolio_weights.copy()

# Retorno das ações
R_t_df =  stocks_returns.copy()

#Retorno do portfólio de mercado
#Iloc[1:] para iniciar as series em 2004-01-05 e eliminar os valores NaN decorrentes do calculo do retorno. 
Y = tf.convert_to_tensor(mkt_return.iloc[1:], dtype=tf.float32)              # Retorno do portfólio de mercado
X = tf.convert_to_tensor(mu_t_df.shift(1).iloc[1:].values, dtype=tf.float32) # Pesos de mercado

# Prepare stock returns data (should align with training samples)
stock_returns = tf.convert_to_tensor(R_t_df.iloc[1:].values, dtype=tf.float32)


from sklearn.model_selection import train_test_split

indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle=False)

# Split datasets
X_train = tf.convert_to_tensor(X.numpy()[train_idx], dtype=tf.float32)
X_test  = tf.convert_to_tensor(X.numpy()[test_idx], dtype=tf.float32)

Y_train = tf.convert_to_tensor(Y.numpy()[train_idx] , dtype=tf.float32)
Y_test  = tf.convert_to_tensor(Y.numpy()[test_idx], dtype=tf.float32)

stock_returns_train = tf.gather(stock_returns, train_idx)
stock_returns_test = tf.gather(stock_returns, test_idx)

 #100


training_input = tf.concat([X_train, stock_returns_train], axis=1)
test_input     = tf.concat([X_test, stock_returns_test], axis=1)

input_dim =  training_input.shape[1] # 200

##################################################
############### DESIGN DA REDE ###################
##################################################

class PINN(tf.keras.Model):
    def __init__(self, input_dim):
        super(PINN, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal()
        
        self.hidden_layers = [
            tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer, input_dim=input_dim),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(90, activation='tanh', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(80, activation='tanh', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(60, activation='tanh', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2)
        ]
        
        self.output_layer = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs):
        # Split input into portfolio weights (x) and stock returns (r_t)
        x = inputs[:, 0:100]
        r_t = inputs[:, 100:200]
        
        # Concatenate and process through hidden layers
        z = tf.concat([x, r_t], axis=1)
        for layer in self.hidden_layers:
            z = layer(z)
            
        return self.output_layer(z)

# Define the loss function (physics-informed loss)
def custom_loss(model, x, ret, ret_mkt):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(ret)
        
        G_x_pred = model(tf.concat([x, ret], axis=1))
        log_G_x_pred = tf.math.log(G_x_pred + 1e-7)
        
        grad_log_G_X = tape.gradient(log_G_x_pred, x)
        inner_prod = tf.reduce_sum(x * grad_log_G_X, axis=1)
        gra_log_G_plus_1 = grad_log_G_X + 1
        sum_terms = gra_log_G_plus_1 - tf.expand_dims(inner_prod, axis=1)
        pi_t = sum_terms * x
        
        # Portfolio returns with safeguards
        port_ret = tf.clip_by_value(
            tf.reduce_sum(pi_t * ret, axis=1),
            -0.95, 10.0
        )
        
        # Cumulative returns calculation
        gross_returns = 1 + port_ret
        cumprod = tf.math.cumprod(gross_returns)
        ret_acumulado = tf.clip_by_value(cumprod - 1, -0.99, 1e3)

        # Market returns calculation
        gross_mkt_returns = 1 + tf.clip_by_value(ret_mkt, -0.95, 10.0)
        cumprod_mkt = tf.math.cumprod(gross_mkt_returns)
        ret_acumulado_mkt = tf.clip_by_value(cumprod_mkt - 1, -0.99, 1e3)

    # Final loss calculation
    safe_ret = tf.maximum(1 + ret_acumulado, 1e-7)
    safe_ret_mkt = tf.maximum(1 + ret_acumulado_mkt, 1e-7)
    log_excess_return = tf.math.log(safe_ret) - tf.math.log(safe_ret_mkt)

    return -tf.reduce_mean(log_excess_return)

# Prepare dataset (NO SHUFFLING)
batch_size = 63
dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, stock_returns_train, Y_train)
).batch(batch_size)

# Model Compilation
model = PINN(input_dim=200)  # 100 (x) + 100 (r_t) = 200
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    global_clipnorm=1.0
)

# Training loop (batches processed in sequence)
num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for x_batch, ret_batch, ret_mkt_batch in dataset:
        with tf.GradientTape() as tape:
            physics_loss_value = custom_loss(model, x_batch, ret_batch, ret_mkt_batch)
            total_loss = physics_loss_value

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        epoch_loss += total_loss.numpy()
        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
 
    print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")

print("fim")