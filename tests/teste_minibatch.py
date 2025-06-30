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
stocks_returns    = (top_100_mkt_cap_prices_df / top_100_mkt_cap_prices_df.shift(1))-1
sp500_idx_returns = (spx_index / spx_index.shift(1))-1


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



import tensorflow as tf

##################################################
############### DESIGN DA REDE ###################
##################################################

class PINN(tf.keras.Model):
    def __init__(self, input_dim=200):
        """
        Inicializa o modelo PINN com uma arquitetura otimizada para aprender
        funções suaves (C^2) e côncavas.

        Args:
            input_dim (int): A dimensionalidade da entrada da rede (padrão: 200).
        """
        super(PINN, self).__init__()
        
        initializer = tf.keras.initializers.GlorotNormal()
        l2_regularizer = tf.keras.regularizers.l2(0.001)

        # O corpo principal da rede é definido usando tf.keras.Sequential
        # para maior clareza e manutenibilidade.
        self.hidden_layers = tf.keras.Sequential([
            # Camada de Entrada
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            
            # Bloco Oculto 1
            # Usamos 'swish' por ser suave (C^∞) e ter ótimo desempenho.
            # A arquitetura de largura constante (128 neurônios) é robusta.
            tf.keras.layers.Dense(128, activation='swish', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            
            # Bloco Oculto 2
            tf.keras.layers.Dense(128, activation='swish', kernel_initializer=initializer, kernel_regularizer=l2_regularizer),
            tf.keras.layers.BatchNormalization(),
            
            # Bloco Oculto 3
            tf.keras.layers.Dense(128, activation='swish', kernel_initializer=initializer, kernel_regularizer=l2_regularizer),
            # Nota: Dropout foi removido para garantir derivadas mais estáveis.
        ])
        
        # A camada de saída permanece a mesma, pois 'softplus' garante G > 0.
        self.output_layer = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs):
        """
        Executa a passagem para a frente (forward pass).
        A lógica de separação e concatenação foi removida por ser redundante.
        """
        z = self.hidden_layers(inputs)
        return self.output_layer(z)
    



def custom_loss_2(model, x, ret, ret_mkt):

    ###############################
    # # Derivatives Calculation # #  
    ###############################
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            G_x_pred = model(tf.concat([x, ret], axis=1))  # G(x, r)
            log_G_x_pred = tf.math.log(G_x_pred + 1e-7)
        
        grad_log_G_X = tape1.gradient(log_G_x_pred, x)
        grad_G       = tape1.gradient(G_x_pred, x)  # ∇G(x)

    H_G = tape2.batch_jacobian(grad_G, x)  # ∇²G(x) - Hessian

    ########################################
    # # Functionally Generated Portfolio # #
    ########################################
    inner_prod = tf.reduce_sum(x * grad_log_G_X, axis=1)
    pi_t = ((grad_log_G_X + 1) - tf.expand_dims(inner_prod, axis=1))*x

    # Shift nos pesos para considerar trade date no final do dia
    pi_t_shifted = tf.concat([tf.zeros_like(pi_t[:1]), pi_t[:-1]], axis=0)

    # Retorno do portfólio
    port_ret = tf.reduce_sum(pi_t_shifted * ret, axis=1)

    #############################
    # # Matriz de Covariância # #
    #############################
    ret_mean = tf.reduce_mean(ret, axis=0, keepdims=True)  
    ret_centered = ret - ret_mean  
    n_samples = tf.cast(tf.shape(ret)[0], tf.float32)
    Sigma = tf.matmul(ret_centered, ret_centered, transpose_a=True) / (n_samples - 1.0)

    ######################################################
    # # Covariância relativa ao portfolio gerado (tau) # #
    ######################################################
    T = pi_t.shape[0]

    drift_array = tf.TensorArray(dtype=tf.float32, size=T)
    for t in range(T):
        row_t = pi_t[t, :]  # Get row t (shape [100])
        mu_t  = x[t,:]
        
        sigma_market  = tf.matmul(tf.expand_dims(row_t, 0), Sigma)  # [1,100] * [100,100] = [1,100]
      
        sigma_mm = tf.matmul(
        tf.matmul(tf.expand_dims(row_t, 0), Sigma),  # [1,100] @ [100,100] → [1,100]
        tf.expand_dims(row_t, 1)                     # [1,100] @ [100,1] → [1,1]
         )
        sigma_mm = tf.squeeze(sigma_mm)

        tau_matrix_t = Sigma -  tf.transpose(tf.tile(tf.transpose(sigma_market), [1, sigma_market.shape[1]])) - tf.tile(tf.transpose(sigma_market), [1, sigma_market.shape[1]]) + sigma_mm

        mu_outer = tf.einsum('i,j->ij', mu_t, mu_t)
        contraction = tf.einsum('ij,ij,ij->', H_G[t], tau_matrix_t, mu_outer)
        drift_dt = -0.5 / G_x_pred[t] * contraction
        drift_array = drift_array.write(t, drift_dt)

    drift_dt_tensor = drift_array.stack()

    #########################
    # # Integral do Drift # #
    #########################

    T = tf.shape(drift_dt_tensor)[0]
    delta_t = 1.0  # ou outro valor de passo de tempo, se não for unitário

    # Soma intermediária para regra do trapézio
    sum_trap = tf.reduce_sum(drift_dt_tensor[1:T-1])
    I_trap = delta_t * (0.5 * drift_dt_tensor[0] + sum_trap + 0.5 * drift_dt_tensor[T - 1])

    # Soma intermediária para regra de Simpson
    even_sum = tf.reduce_sum(drift_dt_tensor[2:T-1:2])  # f2, f4, ...
    odd_sum  = tf.reduce_sum(drift_dt_tensor[1:T:2])    # f1, f3, ...
    I_simp = (delta_t / 3.0) * (drift_dt_tensor[0] + 2.0 * even_sum + 4.0 * odd_sum + drift_dt_tensor[T - 1])

    # Se T ímpar (i.e., número de pontos par), aplica Simpson; senão Trapézio
    integral_dg = tf.where(tf.equal(T % 2, 1), I_simp, I_trap)

    #######################
    # # Master Eq Error # #
    #######################
    loss = (tf.math.log(tf.reduce_sum(port_ret)/tf.reduce_sum(mkt_ret) )) - (tf.math.log((G_x_pred[-1]/G_x_pred[0])) + integral_dg)

    mse_loss = tf.reduce_mean(tf.square(loss))


    return -(mse_loss), {
            'excess_return_series': port_ret,
            'grad_log_G_X_norm': tf.norm(grad_log_G_X, ord='euclidean', axis=1),
         }


# Prepara o banco de dados  (sem embaralhamento para manter a dependencia tempral das series)
batch_size = 252
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
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for x_batch, ret_batch, ret_mkt_batch in dataset:
        with tf.GradientTape() as tape:
            loss_value, metrics_dics = custom_loss_2(model, x_batch, ret_batch, ret_mkt_batch)

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

    grad_norms_batch_vect     = []
    excess_ret_per_batch_vect = []

    # Printa a perda a cada 10 épocas
    if epoch % 10==0:
        print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")

print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")