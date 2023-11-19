import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# -----------------------------------------------------------------------------------
# Author: Hunter Gould
# Date: 11/01/23
# Description: backtester.py is part of a portfolio analysis suite.
#              It offers backtesting capabilities, allowing users to test investment
#              strategies against historical market data. This helps in understanding
#              potential performance and risks associated with different investment
#              approaches.
#
# Note: Remember that historical data does not guarantee future results. This backtesting
#       tool is crucial for evaluating the robustness of investment strategies under
#       various market conditions.
# -----------------------------------------------------------------------------------


# Constants for portfolio analysis
STOCK_TICKERS = ['AAPL', 'JNJ', 'PG', 'JPM', 'XOM', 'MMM', 'SO', 'VZ', 'NKE', 'DD']  # Stock tickers representing a diverse portfolio
INITIAL_WEIGHTS = np.array([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])  # Initial weights assigned equally to each stock
ANALYSIS_START_DATE = '2013-11-18'
ANALYSIS_END_DATE = '2018-11-18'
TESTING_END_DATE = '2023-11-18'
BENCHMARK_INDEX = 'SPY'  # Using S&P 500 as the benchmark index
RISK_FREE_RATE = 4.611 / 100  # Using 3-Year T-Bill Returns as the risk-free rate
NUMBER_OF_PORTFOLIO_WEIGHTS = 10_000  # Number of random portfolio weights for Monte Carlo simulation
TRADING_DAYS_PER_YEAR = 252
NUMBER_OF_MONTE_CARLO_RUNS = 1_000

# Data Collection
print("\n================== Starting: Data Collection ==================\n")
stock_data = yf.download(STOCK_TICKERS, start=ANALYSIS_START_DATE, end=ANALYSIS_END_DATE)['Adj Close']
benchmark_data = yf.download(BENCHMARK_INDEX, start=ANALYSIS_START_DATE, end=ANALYSIS_END_DATE)['Adj Close']
print("\n================== Completed: Data Collection ==================\n")

# Optimal Portfolio Weighting Simulation
print("\n================== Starting: Optimal Portfolio Weighting Simulation ==================\n")
# Calculating daily returns for stocks and the benchmark index
stock_daily_returns = stock_data.pct_change().dropna()
benchmark_daily_returns = benchmark_data.pct_change().dropna()
print("Successfully Calculated Daily Returns.")

# Generating the covariance matrix for the stock returns
covariance_matrix = stock_daily_returns.cov()
print("Successfully Calculated Covariance Matrix.")

# Performing Monte Carlo Simulation to find optimal portfolio weights
simulation_results = np.zeros((4, NUMBER_OF_PORTFOLIO_WEIGHTS))
recorded_weights = np.zeros((len(STOCK_TICKERS), NUMBER_OF_PORTFOLIO_WEIGHTS))

print("\nRunning Monte Carlo Simulation for Optimal Portfolio Weights...")
for i in range(NUMBER_OF_PORTFOLIO_WEIGHTS):
    random_weights = np.random.random(len(STOCK_TICKERS))
    normalized_weights = random_weights / np.sum(random_weights)
    recorded_weights[:, i] = normalized_weights
    annualized_return = np.sum(normalized_weights * stock_daily_returns.mean()) * TRADING_DAYS_PER_YEAR
    annualized_stddev = np.sqrt(np.dot(normalized_weights.T, np.dot(covariance_matrix, normalized_weights))) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_stddev
    simulation_results[:, i] = [annualized_return, annualized_stddev, sharpe_ratio, i]

columns = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Simulation Index']
simulated_portfolios = pd.DataFrame(simulation_results.T, columns=columns)
print("Monte Carlo Simulation Completed.")

# Analysis of Simulated Portfolios
sorted_by_volatility = simulated_portfolios.sort_values(by='Annualized Volatility').reset_index()
optimal_sharpe_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
median_volatility_idx = sorted_by_volatility.iloc[len(sorted_by_volatility) // 2]['Simulation Index']
print(f"Achieved Maximum Sharpe Ratio of: {simulated_portfolios['Sharpe Ratio'][optimal_sharpe_idx]:.2f}")

# Extracting and displaying weights of optimal and median volatility portfolios
optimal_weights = recorded_weights[:, optimal_sharpe_idx]
optimal_weights_percent = optimal_weights * 100  # Converting to percentage
optimal_weights_percent_str = ', '.join([f"{weight:.2f}%" for weight in optimal_weights_percent])
median_volatility_weights = recorded_weights[:, int(median_volatility_idx)]
print(f"Optimal Weights for Maximum Sharpe Ratio: {optimal_weights_percent_str}")

print("\n================== Completed: Optimal Portfolio Weighting Simulation ==================\n")

# Probability Distribution Generation
print("\n================== Starting: Probability Distribution Generation ==================\n")
# Calculating mean and volatility of daily returns for each asset and benchmark
daily_mean_returns = stock_daily_returns.mean()
daily_volatility = stock_daily_returns.std()
benchmark_mean_return = benchmark_daily_returns.mean()
benchmark_volatility = benchmark_daily_returns.std()

portfolio_weights = {'Optimized Portfolio': optimal_weights, 'Current Portfolio': INITIAL_WEIGHTS, 'Median Portfolio': median_volatility_weights}

portfolio_results = {name: [] for name in portfolio_weights.keys()}
market_final_values = []

# Defining a function for running Monte Carlo simulations
def run_simulation(weights, length):
    """Runs a Monte Carlo simulation for a given set of weights and time period."""
    fund_value = [10000]
    for _ in range(length):
        individual_asset_returns = np.random.normal(daily_mean_returns, daily_volatility)
        portfolio_return = np.dot(weights, individual_asset_returns)
        fund_value.append(fund_value[-1] * (1 + portfolio_return))
    return fund_value

# Running simulations for each portfolio and the market
print("Running Portfolio Simulations...")
portfolio_metrics = {}
for portfolio_name, weights in portfolio_weights.items():
    final_values = []
    returns = []
    for _ in range(NUMBER_OF_MONTE_CARLO_RUNS):
        simulated_fund_values = run_simulation(weights, TRADING_DAYS_PER_YEAR)
        final_value = simulated_fund_values[-1]
        final_values.append(final_value)
        simulation_return = (final_value / 10000) - 1
        returns.append(simulation_return)
    portfolio_results[portfolio_name] = final_values
    expected_return = np.mean(returns)
    volatility = np.std(returns)
    portfolio_metrics[portfolio_name] = (expected_return, volatility)
    print(f"Completed simulations for {portfolio_name} portfolio.")

# Simulating market performance
print("Simulating Market Performance...")
for _ in range(NUMBER_OF_MONTE_CARLO_RUNS):
    market_fund_value = [10000]
    for _ in range(TRADING_DAYS_PER_YEAR):
        market_return = np.random.normal(benchmark_mean_return, benchmark_volatility)
        market_fund_value.append(market_fund_value[-1] * (1 + market_return))
    market_final_values.append(market_fund_value[-1])

# Calculating market performance statistics
market_final_values_percent = [(value / 10000 - 1) * 100 for value in market_final_values]
market_expected_return = np.mean(market_final_values) / 10000 - 1
market_volatility = np.std(market_final_values) / 10000
market_sharpe_ratio = (market_expected_return - RISK_FREE_RATE) / market_volatility
print("Market Performance Simulation Completed.")

print("\n================== Completed: Probability Distribution Generation ==================\n")


print("\n================== Starting: Backtesting ==================\n")
subsequent_data = yf.download(STOCK_TICKERS, start=ANALYSIS_END_DATE, end=TESTING_END_DATE)['Adj Close']

# Download the new period data for the market (SPY)
subsequent_market_data = yf.download(BENCHMARK_INDEX, start=ANALYSIS_END_DATE, end=TESTING_END_DATE)['Adj Close']

# Calculate the portfolio returns using the optimal weights
subsequent_daily_returns = subsequent_data.pct_change().dropna()
portfolio_subsequent_return = np.sum(optimal_weights * subsequent_daily_returns.mean()) * 252

# Calculate the market returns for the same period
subsequent_market_daily_returns = subsequent_market_data.pct_change().dropna()
market_subsequent_return = subsequent_market_daily_returns.mean() * 252

print("\n================== Completed: Backtesting ==================\n")

# Plotting Results
print("\n================== Starting: Plotting ==================\n")
# Setting up the plot environment and plotting probability distributions for portfolios and market
plt.figure(figsize=(16, 9), constrained_layout=True)
ax = plt.gca()

# Configuring plot aesthetics
plt.gcf().set_facecolor('black')
ax.set_facecolor('black')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100 * y:.2f}%'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}%'))

for spine in ax.spines.values():
    spine.set_edgecolor('white')

palette = sns.color_palette("hsv", len(portfolio_weights) + 1)

plt.axvline(x=market_subsequent_return * 100, color='Purple',)
plt.axvline(x=portfolio_subsequent_return * 100, color='Yellow',)
# Plotting the distributions for each portfolio
for i, (portfolio_name, final_values) in enumerate(portfolio_results.items()):
    color = palette[i]
    final_values_percent = [(value / 10000 - 1) * 100 for value in final_values]
    sns.kdeplot(final_values_percent, label=portfolio_name, color=color, ax=ax)
    expected_return, volatility = portfolio_metrics[portfolio_name]
    plt.axvline(x=expected_return * 100, color=color, linestyle='--')
    sharpe_ratio = (expected_return - RISK_FREE_RATE) / volatility

    # Adding text annotations for portfolio metrics
    plt.text(0.01, .98 - 0.1 * i,
             f'{portfolio_name}\n  Mean: {expected_return * 100:.2f}%\n  Volatility: {volatility * 100:.2f}%\n  Sharpe Ratio: {sharpe_ratio:.2f}',
             fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor='black'))

# Plotting and annotating market performance distribution
market_color = palette[-1]
sns.kdeplot(market_final_values_percent, label=BENCHMARK_INDEX, color=market_color, ax=ax)
plt.axvline(x=market_expected_return * 100, color=market_color, linestyle='--')
plt.text(0.01, 0.98 - 0.1 * (len(portfolio_weights)),
         f'{BENCHMARK_INDEX}\n  Mean: {market_expected_return * 100:.2f}%\n  Volatility: {market_volatility * 100:.2f}%\n  Sharpe Ratio: {market_sharpe_ratio:.2f}',
         fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor=market_color, facecolor='black'))

# Probability comparison between optimal and other portfolios
optimal_beats_initial = sum(np.array(portfolio_results['Optimized Portfolio']) > np.array(portfolio_results['Current Portfolio'])) / NUMBER_OF_MONTE_CARLO_RUNS
optimal_beats_market = sum(np.array(portfolio_results['Optimized Portfolio']) > market_final_values) / NUMBER_OF_MONTE_CARLO_RUNS

# Displaying the probability comparison as text
prob_text = f"Probability Optimal > Current: {optimal_beats_initial:.2%}\nProbability Optimal > Market: {optimal_beats_market:.2%}"
plt.text(0.67, 0.92, prob_text, fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='black'))

# Displaying the optimal weights as text
optimal_weights_text = "Optimal Weights:\n" + "\n".join([f"{STOCK_TICKERS[i]}: {weight:.2f}%" for i, weight in enumerate(optimal_weights_percent)])
plt.text(0.115, .98, optimal_weights_text, fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='black'))

# Displaying actual results from time period selected
plt.text(0.67, 0.98, f"Actual Market Return: {market_subsequent_return:.2%}\nActual Optimized Portfolio Return: {portfolio_subsequent_return:.2%}", fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='black'))

# Setting labels, title and legend
plt.xlabel('Final Fund % Returns')
plt.ylabel('Density')
plt.title('Probability Distributions of Final Fund Returns for Different Portfolios', color='white')
plt.legend(loc='best')

print("\n================== Completed: Plotting ==================\n")
plt.show()
