import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# -----------------------------------------------------------------------------------
# Author: Hunter Gould
# Date: 11/26/23
# Description: This script is a part of a portfolio analysis suite designed for
#              backtesting investment strategies using historical market data. It
#              fetches data for user-chosen stock tickers and a benchmark index,
#              then performs optimal portfolio weighting simulations using Monte Carlo
#              methods. The script calculates daily returns, covariance matrices, and
#              uses these to find optimal weights based on Sharpe ratio. It includes
#              detailed analysis of simulated portfolios, comparing against the market
#              benchmark. The script also extends to backtesting, evaluating the actual
#              performance of the optimized portfolio in a subsequent time period.
#              Visualization features include probability distribution plots for portfolio
#              and market performance, showcasing potential and actual returns.
#
# Note: Remember that historical data does not guarantee future results. This backtesting
#       tool is crucial for evaluating the robustness of investment strategies under
#       various market conditions.
# -----------------------------------------------------------------------------------

# Constants for analysis
# List of stock tickers and their respective holdings
STOCKS = {
    'AAPL': 1500.0,  # Apple Inc.
    'JNJ': 1200.0,   # Johnson & Johnson
    'PG': 800.0,     # Procter & Gamble Co.
    'JPM': 1300.0,   # JPMorgan Chase & Co.
    'XOM': 700.0,    # Exxon Mobil Corporation
    'MMM': 600.0,    # 3M Company
    'SO': 500.0,     # Southern Company
    'VZ': 600.0,     # Verizon Communications Inc.
    'NKE': 1000.0,   # NIKE, Inc.
    'DD': 800.0      # DuPont de Nemours, Inc.
}

ANALYSIS_START_DATE = '2013-11-26'
ANALYSIS_END_DATE = '2018-11-26'
TESTING_END_DATE = '2023-11-26'
BENCHMARK_INDEX = 'SPY'  # S&P 500 as the benchmark index
RISK_FREE_RATE = 4.611 / 100  # Risk-free rate using 3-Year T-Bill Returns
NUMBER_OF_PORTFOLIO_WEIGHTS = 10_000  # Monte Carlo simulation sample size
TRADING_DAYS_PER_YEAR = 252  # Number of trading days in a year
NUMBER_OF_MONTE_CARLO_RUNS = 1_000  # Number of Monte Carlo runs for simulations


print("\n================== Starting: Data Collection ==================\n")
# Calculate stocks weights and tickers from dictionary
def calculate_weights(stock_dict):
    total_investment = sum(stock_dict.values())
    weights = np.array([amount / total_investment for amount in stock_dict.values()])
    tickers = list(stock_dict.keys())
    return tickers, weights


stock_tickers, initial_weights = calculate_weights(STOCKS)

# Download adjusted close prices for the stocks and benchmark index
stock_data = yf.download(stock_tickers, start=ANALYSIS_START_DATE, end=ANALYSIS_END_DATE)['Adj Close']
benchmark_data = yf.download(BENCHMARK_INDEX, start=ANALYSIS_START_DATE, end=ANALYSIS_END_DATE)['Adj Close']
print("\n================== Completed: Data Collection ==================\n")


print("\n================== Starting: Optimal Portfolio Weighting Simulation ==================\n")
# Calculate daily percentage returns for stocks and the benchmark
stock_daily_returns = stock_data.pct_change().dropna()
benchmark_daily_returns = benchmark_data.pct_change().dropna()
print("Successfully Calculated Daily Returns.")

# Generate covariance matrix for the stock returns
covariance_matrix = stock_daily_returns.cov()
print("Successfully Calculated Covariance Matrix.")

# Initialize arrays for simulation results and recorded weights
simulation_results = np.zeros((4, NUMBER_OF_PORTFOLIO_WEIGHTS))
recorded_weights = np.zeros((len(stock_tickers), NUMBER_OF_PORTFOLIO_WEIGHTS))

# Monte Carlo Simulation for finding optimal portfolio weights
print("\nRunning Monte Carlo Simulation for Optimal Portfolio Weights...")
for i in range(NUMBER_OF_PORTFOLIO_WEIGHTS):
    # Generate random weights and normalize them
    random_weights = np.random.random(len(stock_tickers))
    normalized_weights = random_weights / np.sum(random_weights)
    recorded_weights[:, i] = normalized_weights

    # Calculate annualized return, standard deviation, and Sharpe ratio
    annualized_return = np.sum(normalized_weights * stock_daily_returns.mean()) * TRADING_DAYS_PER_YEAR
    annualized_stddev = np.sqrt(np.dot(normalized_weights.T, np.dot(covariance_matrix, normalized_weights))) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_stddev

    # Store simulation results
    simulation_results[:, i] = [annualized_return, annualized_stddev, sharpe_ratio, i]

# Create DataFrame for simulated portfolios
columns = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Simulation Index']
simulated_portfolios = pd.DataFrame(simulation_results.T, columns=columns)
print("Monte Carlo Simulation Completed.")

# Sort portfolios by volatility and identify portfolios with optimal Sharpe ratio and median volatility
sorted_by_volatility = simulated_portfolios.sort_values(by='Annualized Volatility').reset_index()
optimal_sharpe_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
median_volatility_idx = sorted_by_volatility.iloc[len(sorted_by_volatility) // 2]['Simulation Index']
print(f"Achieved Maximum Sharpe Ratio of: {simulated_portfolios['Sharpe Ratio'][optimal_sharpe_idx]:.2f}")

# Display weights for optimal and median volatility portfolios
optimal_weights = recorded_weights[:, optimal_sharpe_idx]
optimal_weights_percent = optimal_weights * 100
optimal_weights_percent_str = ', '.join([f"{weight:.2f}%" for weight in optimal_weights_percent])
median_volatility_weights = recorded_weights[:, int(median_volatility_idx)]
print(f"Optimal Weights for Maximum Sharpe Ratio: {optimal_weights_percent_str}")

print("\n================== Completed: Optimal Portfolio Weighting Simulation ==================\n")


print("\n================== Starting: Probability Distribution Generation ==================\n")
# Calculate mean and volatility of daily returns for each asset and benchmark
daily_mean_returns = stock_daily_returns.mean()
daily_volatility = stock_daily_returns.std()
benchmark_mean_return = benchmark_daily_returns.mean()
benchmark_volatility = benchmark_daily_returns.std()

# Define portfolio configurations for simulation
portfolio_weights = {'Optimized Portfolio': optimal_weights, 'Current Portfolio': initial_weights, 'Median Portfolio': median_volatility_weights}
portfolio_results = {name: [] for name in portfolio_weights.keys()}
market_final_values = []

# Defining a function for running Monte Carlo simulations
def run_simulation(weights, length, covariance_matrix):
    """Runs a Monte Carlo simulation for a given set of weights and time period, considering asset correlation."""
    fund_value = [10000]
    chol_matrix = np.linalg.cholesky(covariance_matrix)

    for _ in range(length):
        correlated_random_returns = np.dot(chol_matrix, np.random.normal(size=(len(stock_tickers),)))
        individual_asset_returns = daily_mean_returns + correlated_random_returns
        portfolio_return = np.dot(weights, individual_asset_returns)
        fund_value.append(fund_value[-1] * (1 + portfolio_return))
    return fund_value

# Run simulations for each portfolio and the market
print("Running Portfolio Simulations...")
portfolio_metrics = {}
for portfolio_name, weights in portfolio_weights.items():
    final_values = []
    returns = []
    for _ in range(NUMBER_OF_MONTE_CARLO_RUNS):
        simulated_fund_values = run_simulation(weights, TRADING_DAYS_PER_YEAR, covariance_matrix)
        final_value = simulated_fund_values[-1]
        final_values.append(final_value)
        simulation_return = (final_value / 10000) - 1
        returns.append(simulation_return)
    portfolio_results[portfolio_name] = final_values
    expected_return = np.mean(returns)
    volatility = np.std(returns)
    portfolio_metrics[portfolio_name] = (expected_return, volatility)
    print(f"Completed simulations for {portfolio_name} portfolio.")

# Simulate market performance
print("Simulating Market Performance...")
for _ in range(NUMBER_OF_MONTE_CARLO_RUNS):
    market_fund_value = [10000]
    for _ in range(TRADING_DAYS_PER_YEAR):
        market_return = np.random.normal(benchmark_mean_return, benchmark_volatility)
        market_fund_value.append(market_fund_value[-1] * (1 + market_return))
    market_final_values.append(market_fund_value[-1])

# Calculate market performance statistics
market_final_values_percent = [(value / 10000 - 1) * 100 for value in market_final_values]
market_expected_return = np.mean(market_final_values) / 10000 - 1
market_volatility = np.std(market_final_values) / 10000
market_sharpe_ratio = (market_expected_return - RISK_FREE_RATE) / market_volatility
print("Market Performance Simulation Completed.")

print("\n================== Completed: Probability Distribution Generation ==================\n")


print("\n================== Starting: Backtesting ==================\n")
# Download the new period data for the portfolio assets
subsequent_data = yf.download(stock_tickers, start=ANALYSIS_END_DATE, end=TESTING_END_DATE)['Adj Close']

# Download the new period data for the market (SPY)
subsequent_market_data = yf.download(BENCHMARK_INDEX, start=ANALYSIS_END_DATE, end=TESTING_END_DATE)['Adj Close']

# Calculate the optimal portfolio returns using the optimal weights
optimal_subsequent_daily_returns = subsequent_data.pct_change().dropna()
optimal_portfolio_subsequent_return = np.sum(optimal_weights * optimal_subsequent_daily_returns.mean()) * 252

# Calculate the current portfolio returns using the optimal weights
current_subsequent_daily_returns = subsequent_data.pct_change().dropna()
current_portfolio_subsequent_return = np.sum(initial_weights * current_subsequent_daily_returns.mean()) * 252

# Calculate the market returns for the same period
subsequent_market_daily_returns = subsequent_market_data.pct_change().dropna()
market_subsequent_return = subsequent_market_daily_returns.mean() * 252

print("\n================== Completed: Backtesting ==================\n")


print("\n================== Starting: Plotting ==================\n")
# Setup and configuration for probability distribution plots
plt.figure(figsize=(16, 9), constrained_layout=True)
ax = plt.gca()

# Set plot aesthetics for readability
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

# Mark real world results
plt.axvline(x=market_subsequent_return * 100, color=palette[-1],)
plt.axvline(x=optimal_portfolio_subsequent_return * 100, color=palette[0],)
plt.axvline(x=current_portfolio_subsequent_return * 100, color=palette[1],)

# Plot probability distributions for each portfolio
for i, (portfolio_name, final_values) in enumerate(portfolio_results.items()):
    color = palette[i]
    final_values_percent = [(value / 10000 - 1) * 100 for value in final_values]
    sns.kdeplot(final_values_percent, label=portfolio_name, color=color, ax=ax)
    # Plotting performance metrics for each portfolio
    expected_return, volatility = portfolio_metrics[portfolio_name]
    plt.axvline(x=expected_return * 100, color=color, linestyle='--')
    sharpe_ratio = (expected_return - RISK_FREE_RATE) / volatility

    # Adding text annotations for portfolio metrics
    plt.text(0.01, .98 - 0.1 * i,
             f'{portfolio_name}\n  Mean: {expected_return * 100:.2f}%\n  Volatility: {volatility * 100:.2f}%\n  Sharpe Ratio: {sharpe_ratio:.2f}',
             fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor='black'))

# Plotting market performance distribution
market_color = palette[-1]
sns.kdeplot(market_final_values_percent, label=BENCHMARK_INDEX, color=market_color, ax=ax)
plt.axvline(x=market_expected_return * 100, color=market_color, linestyle='--')

# Adding market performance annotations
plt.text(0.01, 0.98 - 0.1 * (len(portfolio_weights)),
         f'{BENCHMARK_INDEX}\n  Mean: {market_expected_return * 100:.2f}%\n  Volatility: {market_volatility * 100:.2f}%\n  Sharpe Ratio: {market_sharpe_ratio:.2f}',
         fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor=market_color, facecolor='black'))

# Probability comparison between optimal and other portfolios
optimal_beats_initial = sum(np.array(portfolio_results['Optimized Portfolio']) > np.array(portfolio_results['Current Portfolio'])) / NUMBER_OF_MONTE_CARLO_RUNS
optimal_beats_market = sum(np.array(portfolio_results['Optimized Portfolio']) > market_final_values) / NUMBER_OF_MONTE_CARLO_RUNS

# Displaying the probability comparison as text
prob_text = f"Probability Optimal > Current: {optimal_beats_initial:.2%}\nProbability Optimal > Market: {optimal_beats_market:.2%}"
plt.text(0.67, 0.90, prob_text, fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='black'))

# Displaying the optimal weights as text
optimal_weights_text = "Optimal Weights:\n" + "\n".join([f"{stock_tickers[i]}: {weight:.2f}%" for i, weight in enumerate(optimal_weights_percent)])
plt.text(0.115, .98, optimal_weights_text, fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='black'))

# Displaying actual results from time period selected
plt.text(0.67, 0.98, f"Actual Optimized Portfolio Return: {optimal_portfolio_subsequent_return:.2%}\nActual Original Portfolio Return: {current_portfolio_subsequent_return:.2%}\nActual Market Return: {market_subsequent_return:.2%}", fontsize=10, verticalalignment='top', ha='left', color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor='grey', facecolor='black'))

# Finalizing plot settings
plt.xlabel('Final Fund % Returns')
plt.ylabel('Density')
plt.title('Probability Distributions of Final Fund Returns for Different Portfolios', color='white')
plt.legend(loc='best')

# Display the plot
print("\n================== Completed: Plotting ==================\n")
plt.show()
