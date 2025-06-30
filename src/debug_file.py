import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from datetime import datetime

#Neural Nets
import tensorflow as tf

# Add the parent directory to the Python path to allow imports from tests/
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '...'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dotenv import load_dotenv

from mkt_data_ETL.data_load_and_transform import get_data, get_top_mkt_cap_stocks

import warnings
import random

load_dotenv()
# Suppress all warnings
warnings.filterwarnings("ignore")


# =============================================================================
# REPRODUCIBILITY SETUP - Set seeds for deterministic results
# =============================================================================
def set_seeds(seed=42):
    """
    Set seeds for reproducible results across all random number generators.
    
    Args:
        seed (int): The seed value to use for all random number generators
    """
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set TensorFlow seeds
    tf.random.set_seed(seed)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Configure TensorFlow for deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    print(f"All seeds set to: {seed}")
    print("Deterministic operations enabled for reproducible results.")

# Set the seed for reproducible results
SEED = 42
set_seeds(SEED)

def verify_seed_settings():
    """
    Verify that all seed settings are properly configured.
    This function can be called to check reproducibility setup.
    """
    print("=== Reproducibility Verification ===")
    print(f"SEED value: {SEED}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    print(f"TF_DETERMINISTIC_OPS: {os.environ.get('TF_DETERMINISTIC_OPS', 'Not set')}")
    print(f"TF_CUDNN_DETERMINISTIC: {os.environ.get('TF_CUDNN_DETERMINISTIC', 'Not set')}")
    print("=====================================")

# Uncomment the line below to verify seed settings
# verify_seed_settings()
# =============================================================================

# Get data 
stock_prices_df, stock_shares_amount_df, mkt_cap_df, spx_index, removed_companies = get_data()
top_100_mkt_cap_df, top_100_mkt_cap_prices_df=get_top_mkt_cap_stocks(stock_prices_df=stock_prices_df, 
stock_mkt_cap_df=mkt_cap_df)


# Calculate daily returns
stocks_returns    = np.log(top_100_mkt_cap_prices_df / top_100_mkt_cap_prices_df.shift(1))
sp500_idx_returns =  np.log(spx_index / spx_index.shift(1))

# Rolling volatility
window_size = 252*2 # 2 years
Sigma_df= stocks_returns.rolling(window=window_size).cov(pairwise=True)
Sigma_df = Sigma_df.dropna()

# Get the first date in the cleaned DataFrame
START_DATE = Sigma_df.index.get_level_values(0)[0]

# Filter dataframes to start from START_DATE
top_100_mkt_cap_df = top_100_mkt_cap_df.loc[START_DATE:]
stocks_returns     = stocks_returns.loc[START_DATE:]
stock_prices_df    = stock_prices_df.loc[START_DATE:]

assets = stocks_returns.columns
dates  = stocks_returns.index

n = len(assets)
T = len(dates)
Sigma_t = np.empty((T, n, n))

# Fill array with each rolling covariance matrix
for i, t in enumerate(dates):
    Sigma = Sigma_df.loc[t].reindex(index=assets, columns=assets).values
    Sigma_t[i] = Sigma


# Mkt Weights
mkt_portfolio_weights = top_100_mkt_cap_df.div(top_100_mkt_cap_df.sum(axis=1),axis=0)

#Retorno do portfólio de mercado
mkt_return = (mkt_portfolio_weights.shift(1) * stocks_returns).sum(axis=1)


##################################################
# # MONTAGEM DAS AMOSTRAS DE TREINO E DE TESTE # #
##################################################


def compute_relative_covariance(sigma: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de covariância relativa τ_ij^π(t) com base na covariância σ_ij(t)
    e no vetor de pesos do portfólio π(t).
    
    Args:
        sigma (np.ndarray): Matriz de covariância dos ativos (n x n).
        pi (np.ndarray): Vetor de pesos do portfólio (n,).

    Returns:
        np.ndarray: Matriz de covariância relativa τ^π (n x n).
    """
    # Covariância ativo i com portfólio: sigma_iπ = sigma @ pi
    sigma_i_pi = sigma @ pi        # (n,)
    sigma_pi_pi = pi.T @ sigma @ pi  # escalar

    # Matriz τ_ij^π = σ_ij - σ_iπ - σ_jπ + σ_ππ
    n = len(pi)
    tau = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            tau[i, j] = sigma[i, j] - sigma_i_pi[i] - sigma_i_pi[j] + sigma_pi_pi
    return tau

print("initializing tau_t calculation")
start_time = time.time()

tau_t = np.empty((T, n, n))
for t in range(T):
    tau_t[t] = compute_relative_covariance(Sigma_t[t], mkt_portfolio_weights.iloc[t].values)

end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Elapsed time: {elapsed_seconds:.2f} seconds")


# Copying data generated previously 
mu_t_df = mkt_portfolio_weights.copy()
R_t_df =  stocks_returns.copy()
 
mu_tf                   = tf.convert_to_tensor(mu_t_df.values, dtype=tf.float32) # Pesos de mercado
mkt_ret_tf              = tf.convert_to_tensor(mkt_return.values, dtype=tf.float32)              # Retorno do portfólio de mercado
R_t_tf                  = tf.convert_to_tensor(R_t_df.values, dtype=tf.float32)
tau_t_tf                = tf.convert_to_tensor(tau_t, dtype=tf.float32)


indices = np.arange(len(mu_tf))
train_idx, test_idx = train_test_split(indices, test_size=0.1, shuffle=False, random_state=SEED)

# Split datasets

#Treino
mu_train            = tf.gather(mu_tf, train_idx)
mkt_ret_train       = tf.gather(mkt_ret_tf, train_idx)
stock_returns_train = tf.gather(R_t_tf, train_idx)
tau_train           = tf.gather(tau_t_tf, train_idx)

# Teste
mu_test            = tf.gather(mu_tf, test_idx)
mkt_ret_test       = tf.gather(mkt_ret_tf, test_idx)
stock_returns_test = tf.gather(R_t_tf,  test_idx)
tau_test           = tf.gather(tau_t_tf, test_idx)



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

        self.s1 = tf.Variable(initial_value=-5, trainable=True, dtype=tf.float32)
        self.s2 = tf.Variable(initial_value=-5, trainable=True, dtype=tf.float32)

        
        initializer = tf.keras.initializers.GlorotNormal(seed=SEED)
        l2_regularizer = tf.keras.regularizers.l2(0.001)

        # O corpo principal da rede é definido usando tf.keras.Sequential
        # para maior clareza e manutenibilidade.
        self.hidden_layers = tf.keras.Sequential([
            # Camada de Entrada
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            
            # Bloco Oculto 1
            # Usamos 'swish' por ser suave (C^∞) e ter ótimo desempenho.
            # A arquitetura de largura constante (128 neurônios) é robusta.
            tf.keras.layers.Dense(100, activation='swish', kernel_initializer=initializer),
            
            # Bloco Oculto 2
            tf.keras.layers.Dense(100, activation='swish', kernel_initializer=initializer, kernel_regularizer=l2_regularizer),
            
            # Bloco Oculto 3
            tf.keras.layers.Dense(100, activation='swish', kernel_initializer=initializer, kernel_regularizer=l2_regularizer),
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
    

def custom_loss(model, x, ret, ret_mkt, tau):
    """
    Custom loss function for the PINN.

    Args:
        model (PINN): The PINN model.
        x (tf.Tensor): Market weights.
        ret (tf.Tensor): Stock returns.
        ret_mkt (tf.Tensor): The market return.
        tau (tf.Tensor): Relative covariance matrix.

    Returns:
        _type_: Loss value and metrics.
    """

    ###############################
    # # Derivatives Calculation # #  
    ###############################
    try: 
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                G_x_pred = model(tf.concat([x, ret], axis=1))  # G(x, r)
                log_G_x_pred = tf.math.log(G_x_pred + 1e-7)
            
            grad_log_G_X = tape1.gradient(log_G_x_pred, x)
            grad_G       = tape1.gradient(G_x_pred, x)  # ∇G(x)

        H_G = tape2.batch_jacobian(grad_G, x)  # ∇²G(x) - Hessian

        ###################################################################################
        # # Functionally Generated Portfolio - Generated by the Neural Network function # #
        ###################################################################################

        inner_prod = tf.reduce_sum(x * grad_log_G_X, axis=1)
        pi_t = ((grad_log_G_X + 1) - tf.expand_dims(inner_prod, axis=1))*x

        # Shift weights to consider trade date at the end of the day
        pi_t_shifted = tf.concat([tf.zeros_like(pi_t[:1]), pi_t[:-1]], axis=0)

        # Portfolio return
        port_ret = tf.reduce_sum(pi_t_shifted * ret, axis=1)


        #########################
        # # Drift Calculation # #
        #########################
        T = tf.shape(pi_t)[0]

        mu_i = tf.expand_dims(x, axis=2)     # (T, n, 1)
        mu_j = tf.expand_dims(x, axis=1)     # (T, 1, n)

        # internal product μ_i μ_j: shape (T, n, n)
        mu_outer = mu_i * mu_j

        # elementwise: H * μ_i * μ_j * τ
        elementwise = H_G * mu_outer * tau  # shape (T, n, n)

        # sum over i and j (last two dimensions)
        summed = tf.reduce_sum(elementwise, axis=[1, 2])  # shape (T,)

        # Drift final: -1 / (2G) * sum
        dg_t = -0.5 * summed / G_x_pred


        # #########################
        # # # Drift integration # #
        # #########################
        base = tf.range(T)

        # Conditional: trapezoid if T % 2 == 0, Simpson otherwise
        g_t = tf.cond(
            tf.equal(T % 2, 0),
            lambda: tf.reduce_sum(
                tf.cast(tf.where((base == 0) | (base == T - 1), 0.5, 1.0), dg_t.dtype) * dg_t
            ),
            lambda: (1.0 / 3.0) * tf.reduce_sum(
                tf.cast(tf.where(base == 0, 1.0,
                    tf.where(base == T - 1, 1.0,
                    tf.where(base % 2 == 0, 2.0, 4.0))), dg_t.dtype) * dg_t
            )
        )

        # #######################
        # # # Master Eq Error # #
        # #######################

        eps = 1e-6

        G0 = tf.clip_by_value(G_x_pred[0], clip_value_min=eps, clip_value_max=1e6)
        GT = tf.clip_by_value(G_x_pred[-1], clip_value_min=eps, clip_value_max=1e6)
        right_hand_side = tf.math.log(G0[0]) - tf.math.log(GT[0]) + g_t

        # Normalizind the returns
        # port_ret_normalized = (port_ret - tf.reduce_mean(port_ret)) / (tf.math.reduce_std(port_ret) + 1e-6)
        # ret_mkr_normalized  = (ret_mkt - tf.reduce_mean(ret_mkt)) / (tf.math.reduce_std(ret_mkt) + 1e-6)

        # Cumulative log return
        port_cumulative_return = tf.exp(tf.reduce_sum(port_ret))
        mkt_cumulative_return = tf.exp(tf.reduce_sum(ret_mkt))
        left_hand_side = tf.math.log(port_cumulative_return) - tf.math.log(mkt_cumulative_return)

        equation_error = tf.square(left_hand_side - right_hand_side)  # Smooth, differentiable
        positivity_penalty = tf.square(tf.nn.relu(-left_hand_side))  # Only penalizes when negative
        
        # gradient_penalty = tf.reduce_mean(tf.square(grad_log_G_X))
        # lambda1 = tf.exp(-model.s1)
        # lambda2 = tf.exp(-model.s2)


        # loss = lambda1*equation_error +  lambda2*positivity_penalty + 0.1*(model.s1 + model.s2) #+ 0.001 * gradient_penalty

        loss = equation_error +  positivity_penalty 

        return loss, {
                'grad_log_G_X_norm': tf.norm(grad_G, ord='euclidean', axis=1),
                'portfolio_weights': pi_t,
                'integrated_drift': g_t,
            }
    finally:
        if tape1 is not None:
            del tape1
        if tape2 is not None:
            del tape2

batch_size = 63
dataset = tf.data.Dataset.from_tensor_slices(
    (mu_train, stock_returns_train, mkt_ret_train, tau_train)
).batch(batch_size)

# Model Compilation
# Note: Seeds have been set for reproducible results
model = PINN(input_dim=200)  # 100 (x) + 100 (r_t) = 200
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001)


###########################################################
## Vetores para armazenamento de métricas de Treinamento ##
###########################################################
avg_epoch_loss_vect = []

grad_norms_per_epoch = {}
mkt_weights_per_epoch = {}
drift_per_epoch = {}

grad_norms_batch_vect     = []
mkt_weights_per_batch_vect = []
drift_per_batch_vect = []

# Training loop (batches processed in sequence)
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    for x_batch, ret_batch, ret_mkt_batch, tau_batch in dataset:
        with tf.GradientTape() as tape:
            loss_value, metrics_dics = custom_loss(model, x_batch, ret_batch, ret_mkt_batch, tau_batch)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        grad_norms_batch_vect = grad_norms_batch_vect + metrics_dics['grad_log_G_X_norm'].numpy().tolist()
        mkt_weights_per_batch_vect = mkt_weights_per_batch_vect + metrics_dics['portfolio_weights'].numpy().tolist()
        drift_per_batch_vect.append(metrics_dics['integrated_drift'].numpy())

        epoch_loss += loss_value.numpy()
        num_batches += 1

    avg_epoch_loss = epoch_loss / num_batches
    
    avg_epoch_loss_vect.append(avg_epoch_loss)
    grad_norms_per_epoch[str(epoch)]  = grad_norms_batch_vect
    mkt_weights_per_epoch[str(epoch)]  = mkt_weights_per_batch_vect
    drift_per_epoch[str(epoch)]  = drift_per_batch_vect

    grad_norms_batch_vect     = []
    mkt_weights_per_batch_vect = []
    drift_per_batch_vect = []

    # Print average Epoch loss
    if epoch % 1==0:
        print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")

print(f"Epoch {epoch}/{num_epochs}, Avg Loss: {avg_epoch_loss}")




