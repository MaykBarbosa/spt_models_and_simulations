import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from datetime import datetime

#Neural Nets
import tensorflow as tf

#Bloomberg python library. Is necessary to install bloomberg BLP API and have access to a bloomberg terminal
#from xbbg import blp

from data_eng import get_raw_data, get_selected_stocks

from dotenv import load_dotenv
import os
load_dotenv()

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


# Get data 
sp500_comp, spx_index, sp500_stock_data = get_raw_data()
top_100_mkt_cap_df, top_100_mkt_cap_prices_df = get_selected_stocks(sp500_stock_data)

#Cálculo dos pesos dos portfolios 

# Calculate daily returns
stocks_returns    = np.log(top_100_mkt_cap_prices_df / top_100_mkt_cap_prices_df.shift(1))
sp500_idx_returns =  np.log(spx_index / spx_index.shift(1))


mkt_portfolio_weights = top_100_mkt_cap_df.div(top_100_mkt_cap_df.sum(axis=1),axis=0)

#Retorno do portfólio de mercado
mkt_return = (mkt_portfolio_weights.shift(1) * stocks_returns).sum(axis=1)


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
mu                   = tf.convert_to_tensor(mu_t_df.iloc[1:].values, dtype=tf.float32) # Pesos de mercado
mkt_ret              = tf.convert_to_tensor(mkt_return.iloc[1:], dtype=tf.float32)              # Retorno do portfólio de mercado
R_t_df               = tf.convert_to_tensor(R_t_df.iloc[1:].values, dtype=tf.float32)

indices = np.arange(len(mu))
train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle=False)

# Split datasets

#Treino
mu_train            = tf.convert_to_tensor(mu.numpy()[train_idx], dtype=tf.float32)
mkt_ret_train       = tf.convert_to_tensor(mkt_ret.numpy()[train_idx] , dtype=tf.float32)
stock_returns_train = tf.gather(R_t_df, train_idx)

# Teste
mu_test            = tf.convert_to_tensor(mu.numpy()[test_idx], dtype=tf.float32)
mkt_ret_test       = tf.convert_to_tensor(mkt_ret.numpy()[test_idx], dtype=tf.float32)
stock_returns_test = tf.gather(R_t_df, test_idx)


# Input da rede
training_input = tf.concat([mu_train, stock_returns_train], axis=1)
test_input     = tf.concat([mu_test,  stock_returns_test], axis=1)

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
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(80, activation='tanh', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(60, activation='tanh', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=initializer, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.1)
        ]
        
        self.output_layer = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs):
        # Separa pesos do portfólio de mercado (x) e retorno das ações (r_t)
        x = inputs[:, 0:100]
        r_t = inputs[:, 100:200]
        
        # Concatena novamente para a primeira camada e processa as camadas escondidas
        z = tf.concat([x, r_t], axis=1)
        for layer in self.hidden_layers:
            z = layer(z)
        return self.output_layer(z)
    


def compute_cumulative_returns(x, ret, ret_mkt, grad_log_G_X):
    """_summary_

    Args:
        x (_type_): Market portfolio weights
        ret (_type_): Stocks Returns
        ret_mkt (_type_): Market Return
        grad_log_G_X (_type_): Gradient of Log of the network output 

    Returns:
        Network generated portfolio cumulative return, Market cumulative return : Cumulative returns 
    """
    inner_prod = tf.reduce_sum(x * grad_log_G_X, axis=1)
    pi_t = ((grad_log_G_X + 1) - tf.expand_dims(inner_prod, axis=1))*x

    # Shift nos pesos para considerar trade date no final do dia
    pi_t_shifted = tf.concat([tf.zeros_like(pi_t[:1]), pi_t[:-1]], axis=0)

    # Retorno do portfólio
    port_ret = tf.reduce_sum(pi_t_shifted * ret, axis=1)
    gross_returns = 1 + port_ret
    ret_acumulado = tf.math.cumprod(gross_returns)

    # Retorno do mercado
    gross_mkt_returns = 1 + ret_mkt
    ret_acumulado_mkt = tf.math.cumprod(gross_mkt_returns)

    return ret_acumulado, ret_acumulado_mkt
    


    # Função de perda customizada (physics-informed)
def custom_loss(model, x, ret, ret_mkt):
    """_summary_

    Args:
        model (_type_): Keras Model
        x (_type_): Market Weights
        ret (_type_): Stocks Returns
        ret_mkt (_type_): Market portfolio return

    Returns:
        _type_: loss
    """

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        
        G_x_pred = model(tf.concat([x, ret], axis=1))
        log_G_x_pred = tf.math.log(G_x_pred + 1e-7)

    # Derivada de log(G) em relação a x
    grad_log_G_X = tape.gradient(log_G_x_pred, x)

    # Modularização da lógica de retorno
    ret_acumulado, ret_acumulado_mkt = compute_cumulative_returns(x, ret, ret_mkt, grad_log_G_X)

    # Perda baseada no retorno em excesso
    excess_return = ret_acumulado - ret_acumulado_mkt
    
    # A função de perda
    averave_excess_return= tf.reduce_mean(excess_return)

    # Garantir que o valor de excess_return seja numericamente seguro para o log    
    # transformando em logaritmo adicionando 1 para que valores positivos de excesso de retorno não tenha log negativo 
    epsilon = 1e-7
    log_avg_excess = tf.math.log(1+ tf.maximum(averave_excess_return, epsilon))


    return -log_avg_excess, {
            'excess_return_series': excess_return,
            'grad_log_G_X_norm': tf.norm(grad_log_G_X, ord='euclidean', axis=1),
         }

# Prepara o banco de dados  (sem embaralhamento para manter a dependencia tempral das series)
batch_size = 63
dataset = tf.data.Dataset.from_tensor_slices(
    (mu_train, stock_returns_train, mkt_ret_train)
).batch(batch_size)

# Model Compilation
model = PINN(input_dim=200)  # 100 (x) + 100 (r_t) = 200
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001)


###########################################################
## Vetores para armazenamento de métricas de Treinamento ##
###########################################################
avg_epoch_loss_vect = []

grad_norms_per_epoch = {}
excess_of_ret_per_epoch = {}

grad_norms_batch_vect     = []
excess_ret_per_batch_vect = []


# Training loop (batches processed in sequence)
num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for x_batch, ret_batch, ret_mkt_batch in dataset:
        with tf.GradientTape() as tape:
            loss_value, metrics_dics = custom_loss(model, x_batch, ret_batch, ret_mkt_batch)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        grad_norms_batch_vect = grad_norms_batch_vect + metrics_dics['grad_log_G_X_norm'].numpy().tolist()
        excess_ret_per_batch_vect = excess_ret_per_batch_vect + metrics_dics['excess_return_series'].numpy().tolist()

        epoch_loss += loss_value.numpy()
        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    
    avg_epoch_loss_vect.append(avg_epoch_loss)
    grad_norms_per_epoch[str(epoch)]  = grad_norms_batch_vect
    excess_of_ret_per_epoch[str(epoch)]  = excess_ret_per_batch_vect

    # Printa a perda a cada 50 épocas
    if epoch % 10==0:
        print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")


