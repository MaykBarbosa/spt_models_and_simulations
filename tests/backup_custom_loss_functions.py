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