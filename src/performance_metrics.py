import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_G_nn_with_test_highlight(model, mu_tf, R_t_tf, test_idx, dates, ax=None):
    """
    Plots the estimated function G_p_NN over the whole sample and highlights the test sample region.
    Args:
        model: Trained PINN model.
        mu_tf: Tensor of market weights for the whole sample.
        R_t_tf: Tensor of returns for the whole sample.
        test_idx: Indices of the test sample (list or array).
        dates: Array-like of dates (same length as mu_tf and R_t_tf), used for the x-axis.
        ax: Optional matplotlib axis to plot on. If None, creates a new figure.
    """


    # Compute G_p_NN for the whole sample
    G_p_nn_all = model(mu_tf, training=False).numpy().flatten()

    if ax is None:
        plt.figure(figsize=(12, 5))
        ax = plt.gca()

    ax.plot(dates, G_p_nn_all, label="$G_{NN} ( \mu(t) )$", color='blue')

    # Highlight test region
    test_start = min(test_idx)
    test_end = max(test_idx)
    
    plt.grid(linestyle=':')
    ax.margins(x=0)

    ax.set_title("Estimated Function $G_{NN}(\mu(t))$ Over the Whole Sample")
    ax.set_ylabel("$G_{NN}(\mu(t))$")
    ax.axvspan(dates[test_start], dates[test_end], color='orange', alpha=0.3, label="Test Sample Region")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if ax is None:
        plt.show()

def plot_avg_epoch_loss(avg_epoch_loss_vect, ax=None):
    """
    Plots the average epoch loss over training epochs.
    Args:
        avg_epoch_loss_vect: List or array of average loss values per epoch.
        ax: Optional matplotlib axis to plot on. If None, creates a new figure.
    """
    if ax is None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
    ax.plot(avg_epoch_loss_vect, marker='o', color='tab:blue', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Average Loss per Epoch")
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    if ax is None:
        plt.show()

def plot_cumulative_portfolios(market_return, model_return, dwp_return, p_value, target_vol=0.1, ax=None):
    """
    Plot cumulative returns for market, model, and DWP portfolios, normalized to target volatility.

    Parameters:
    -----------
    market_return : pandas Series
        Market portfolio returns (aligned index)
    model_return : pandas Series
        Model portfolio returns (aligned index)
    dwp_return : pandas Series
        DWP portfolio returns (aligned index)
    p_value : float
        DWP portfolio parameter p (for title)
    target_vol : float, optional
        Target annualized volatility (default: 0.1 for 10%)
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates a new figure.
    """
 

    # Prepare series dict
    series_dict = {
        'Market': market_return,
        'Model': model_return,
        'DWP': dwp_return
    }
    colors = {'Market': 'darkred', 'Model': 'blue', 'DWP': 'green'}
    normalized_series = {}
    sharpe_ratios = {}

    for name, series in series_dict.items():
        # Calculate volatility scaler
        vol_scaler = target_vol / series.std() / np.sqrt(252)
        normalized_series[name] = series * vol_scaler
        # Calculate Sharpe ratio
        sharpe_ratios[name] = np.sqrt(252) * normalized_series[name].mean() / normalized_series[name].std()

    if ax is None:
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

    # Plot each series
    for name, series in normalized_series.items():
        label = f'{name} [Sharpe: {sharpe_ratios[name]:.2f}]'
        ((series + 1).cumprod() - 1).plot(label=label, linewidth=1.5, color=colors.get(name, None), ax=ax)

    # Format plot
    plt.grid(linestyle=':')
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(lambda x, y: f'{x:.0%}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), fancybox=False, shadow=False, ncol=3)
    plt.title(f'Cumulative Portfolio Performance [Vol = {target_vol:.0%} ann] | p={p_value}',
              loc='left', fontsize=15, color=(29/255, 87/255, 165/255))
    plt.tight_layout()
    if ax is None:
        plt.show()


def calculate_model_portfolio_return(model, mu, R_t_df):
    """
    Calculate the model portfolio log-returns using the trained model, market weights, and returns.
    Args:
        model: Trained PINN model.
        mu: tf.Tensor of market weights (shape: [T, n_assets]).
        R_t_df: tf.Tensor of returns (shape: [T, n_assets]).
    Returns:
        np.ndarray: Model portfolio log-returns (shape: [T,]).
    """
    
    # Compute gradients with respect to mu
    with tf.GradientTape() as tape:
        tape.watch(mu)
        G = model(mu, training=False)
        log_G = tf.math.log(G)
    grad_log_G_mu = tape.gradient(log_G, mu)


    inner_prod = tf.reduce_sum(mu * grad_log_G_mu, axis=1)
    pi_t = ((grad_log_G_mu + 1) - tf.expand_dims(inner_prod, axis=1))*mu


    pi_t_shifted = tf.concat([pi_t[:1], pi_t[:-1]], axis=0)
    port_ret = tf.reduce_sum(pi_t_shifted * R_t_df, axis=1)
    return port_ret

def calculate_dwp_portfolio_return(mkt_weights, stocks_returns, p_value):


    power_p_transform = mkt_weights.apply(lambda mu: mu**p_value,axis=0 )
    dwp_portfolio_weights = power_p_transform.div(power_p_transform.sum(axis=1),axis=0 )
    dwp_return = (dwp_portfolio_weights.shift(1) * stocks_returns).sum(axis=1)

    return dwp_return

def plot_cumulative_portfolios_with_test_highlight(
    market_return, model_return, dwp_return, dates, test_idx, p_value, target_vol=0.1, ax=None
):
    """
    Plot cumulative returns for market, model, and DWP portfolios, normalized to target volatility,
    using dates as x-axis and highlighting the test period.

    Parameters:
    -----------
    market_return : pandas Series or np.ndarray
        Market portfolio returns (aligned index)
    model_return : pandas Series or np.ndarray
        Model portfolio returns (aligned index)
    dwp_return : pandas Series or np.ndarray
        DWP portfolio returns (aligned index)
    dates : pandas DatetimeIndex or array-like
        Dates for the x-axis (same length as returns)
    test_idx : list or array
        Indices of the test sample (to highlight)
    p_value : float
        DWP portfolio parameter p (for title)
    target_vol : float, optional
        Target annualized volatility (default: 0.1 for 10%)
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates a new figure.
    """

    series_dict = {
        'Market': market_return,
        'Model': model_return,
        'DWP': dwp_return
    }
    colors = {'Market': 'darkred', 'Model': 'blue', 'DWP': 'green'}
    normalized_series = {}
    sharpe_ratios = {}

    for name, series in series_dict.items():
        # Calculate volatility scaler
        vol_scaler = target_vol / np.std(series) / np.sqrt(252)
        normalized_series[name] = series * vol_scaler
        # Calculate Sharpe ratio
        sharpe_ratios[name] = np.sqrt(252) * np.mean(normalized_series[name]) / np.std(normalized_series[name])

    if ax is None:
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1, 1, 1)

    # Plot each series with dates as x-axis (using log-returns: cumulative = cumsum)
    for name, series in normalized_series.items():
        label = f'{name} [Sharpe: {sharpe_ratios[name]:.2f}]'
        ax.plot(dates, np.exp(np.cumsum(series)), label=label, linewidth=1.5, color=colors.get(name, None))

    # Highlight test region
    test_start = min(test_idx)
    test_end = max(test_idx)
    ax.axvspan(dates[test_start], dates[test_end], color='orange', alpha=0.3, label="Test Sample Region")

    # Format plot
    plt.grid(linestyle=':')
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(lambda x, y: f'{x:.0%}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), fancybox=False, shadow=False, ncol=3)
    plt.title(f'Cumulative Portfolio Performance [Vol = {target_vol:.0%} ann] | p={p_value}',
              loc='left', fontsize=15, color=(29/255, 87/255, 165/255))
    plt.tight_layout()
    if ax is None:
        plt.show()


def plot_cumulative_portfolios_performance(series_dict, dates, test_idx, target_vol=0.1):
    """
    Plots cumulative portfolio performance for given return series.

    Args:
        series_dict (dict): Dictionary with keys as series names and values as return series (np.ndarray or pd.Series).
        dates (array-like): Dates corresponding to the return series.
        target_vol (float, optional): Target annualized volatility for normalization. Defaults to 0.1.
        colors (dict, optional): Optional color mapping for each series. Defaults to None.
    """
        
    colors = {'Market': 'darkred', 'Model': 'tab:blue', 'DWP': 'green'}

    normalized_series = {}
    sharpe_ratios = {}

    for name, series in series_dict.items():
        # Calculate volatility scaler
        vol_scaler = target_vol / np.std(series) / np.sqrt(252)
        normalized_series[name] = series * vol_scaler
        # Calculate Sharpe ratio
        sharpe_ratios[name] = np.sqrt(252) * np.mean(normalized_series[name]) / np.std(normalized_series[name])

    plt.figure(figsize=(12,5))
    ax = plt.subplot(1, 1, 1)

    # Plot each series
    for name, series in normalized_series.items():
        label = f'{name} [Sharpe: {sharpe_ratios[name]:.2f}]'
        plt.plot(dates, np.cumsum(series), label=label, linewidth=1.5, color=colors.get(name, None))

    # Optionally highlight test region if test_idx is available in globals
    if 'test_idx' in globals() and len(globals()['test_idx']) > 0:
        test_start_date = dates[globals()['test_idx'][0]]
        test_end_date = dates[globals()['test_idx'][-1]]
        plt.axvspan(test_start_date, test_end_date, color='orange', alpha=0.3, label='Test Sample Region')

    test_start = min(test_idx)
    test_end = max(test_idx)
    ax.axvspan(dates[test_start], dates[test_end], color='orange', alpha=0.3, label="Test Sample Region")
    
    # Format plot
    plt.grid(linestyle=':')
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(lambda x, y: f'{x:.0%}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    plt.legend()
    plt.title(f'Cumulative Portfolio Performance [Vol = {target_vol:.0%} ann]')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
