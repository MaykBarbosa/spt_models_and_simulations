import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from datetime import datetime

#Neural Nets
import tensorflow as tf


from data_eng import get_raw_data, get_selected_stocks
from dotenv import load_dotenv

import os
import warnings

load_dotenv()
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
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            G_x_pred = model(tf.concat([x, ret], axis=1))
            D_p = tf.square(tf.reduce_sum(tf.sqrt(x), axis=1))

        # Gradientes 
        grad_G_pred = tape.gradient(G_x_pred, x)
        grad_D_p    = tape.gradient(D_p, x)
        del tape 
    # Hessianos
    H_G_pred = tape2.batch_jacobian(grad_G_pred, x)
    H_D_p    = tape2.batch_jacobian(grad_D_p, x)
    del tape2
    
    #############################################
    # #          PERDA MSE DAS FUNÇÕES        # #
    #############################################
    
    loss_func  = G_x_pred - tf.expand_dims(D_p, axis=1)
    mse =  tf.reduce_mean(tf.square(loss_func))
    
    #############################################
    # #          PERDAS DAS HESSIANAS         # #
    #############################################

    H_diff = H_G_pred - H_D_p 
    hessian_loss =  tf.reduce_mean(tf.square(tf.norm(H_diff, ord='fro', axis=[1, 2])))

    #############################################
    # #          PERDAS DOS GRADIENTES        # #
    #############################################

    grad_diff = grad_G_pred - grad_D_p
    gradient_loss =  tf.reduce_mean(tf.square(tf.norm(grad_diff, ord='euclidean', axis=1, keepdims=True)))

    #############################################
    # #            PERDA TOTAL                # #
    #############################################
    
    total_loss = (mse +  0.5*gradient_loss + 0.25*hessian_loss)

    return total_loss, {
            'excess_return_series': loss_func,
            'grad_log_G_X_norm': tf.norm(G_x_pred, ord='euclidean', axis=1),
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
num_epochs = 5
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

    grad_norms_batch_vect     = []
    excess_ret_per_batch_vect = []

    # Print average Epoch loss
    if epoch % 1==0:
        print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")

print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")