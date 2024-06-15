import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def optimize_portfolio(selected_stocks, risk_free_rate):
    returns_df = pd.read_csv('returns_filled.csv')
    selected_returns = returns_df[selected_stocks]

    num_ports = 5000
    all_weights = np.zeros((num_ports, len(selected_stocks)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for i in range(num_ports):
        weights = np.random.random(len(selected_stocks))
        weights /= np.sum(weights)
        all_weights[i, :] = weights

        portfolio_return = np.sum(selected_returns.mean() * 250 * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(selected_returns.cov() * 250, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        ret_arr[i] = portfolio_return
        vol_arr[i] = portfolio_volatility
        sharpe_arr[i] = sharpe_ratio

    max_sr_idx = sharpe_arr.argmax()
    max_sr_ret = ret_arr[max_sr_idx]
    max_sr_vol = vol_arr[max_sr_idx]

    optimized_weights = all_weights[max_sr_idx]

    plot_efficient_frontier(vol_arr, ret_arr, max_sr_vol, max_sr_ret, risk_free_rate)
    plot_weights_pie_chart(selected_stocks, optimized_weights)

    result = {
        'return': max_sr_ret,
        'volatility': max_sr_vol,
        'sharpe_ratio': sharpe_arr[max_sr_idx],
        'weights': dict(zip(selected_stocks, optimized_weights * 100))  # Convert weights to percentage
    }

    return result

def plot_efficient_frontier(vol_arr, ret_arr, max_sr_vol, max_sr_ret, risk_free_rate):
    plt.figure(figsize=(10, 6))
    plt.scatter(vol_arr, ret_arr, c=(ret_arr - risk_free_rate) / vol_arr, marker='o')
    plt.colorbar(label='Sharpe ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(max_sr_vol, max_sr_ret, c='red', marker='x', s=200)  # red dot
    plt.title('Efficient Frontier')
    plt.savefig('efficient_frontier.png')

def plot_weights_pie_chart(stocks, weights):
    plt.figure(figsize=(8, 8))
    plt.pie(weights, labels=stocks, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.title('Optimized Portfolio Weights')
    plt.savefig('weights_pie_chart.png')
