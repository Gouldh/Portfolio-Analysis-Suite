import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Define constants for the analysis
ASSETS = ['AMD', 'NVDA', 'NEM', 'REGN', 'AAPL']
START_DATE = '2013-01-01'
END_DATE = '2023-01-01'
MARKET_REPRESENTATION = 'SPY'
NUM_WEIGHTS = 10_000
SIMULATION_LENGTH = 252  # Number of trading days in a year
NUM_SIMULATIONS = 1000  # Number of simulations to run

# Downloading historical stock data
print("Starting: Data Collection\n")
data = yf.download(ASSETS, start=START_DATE, end=END_DATE)['Adj Close']
market_data = yf.download(MARKET_REPRESENTATION, start=START_DATE, end=END_DATE)['Adj Close']
print("\nCompleted: Data Collection\n")

print("\nStarting: Optimal Weighting Simulation\n")
# Calculate daily returns and covariance matrix for asset returns
daily_returns = data.pct_change().dropna()
cov_matrix = daily_returns.cov()
market_daily_returns = market_data.pct_change().dropna()

# Perform Monte Carlo Simulation to find optimal portfolio weights
results = np.zeros((4, NUM_WEIGHTS))
weights_record = np.zeros((len(ASSETS), NUM_WEIGHTS))

for i in range(NUM_WEIGHTS):
    weights = np.random.random(len(ASSETS))
    weights /= np.sum(weights)
    weights_record[:, i] = weights
    portfolio_return = np.sum(weights * daily_returns.mean()) * SIMULATION_LENGTH
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(SIMULATION_LENGTH)
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = portfolio_return / portfolio_stddev  # Sharpe Ratio calculation
    results[3, i] = i  # Index of the simulation

columns = ['Return', 'Volatility', 'Sharpe Ratio', 'Simulation']
simulated_portfolios = pd.DataFrame(results.T, columns=columns)

# Identifying portfolios with specific characteristics
optimal_idx = simulated_portfolios['Sharpe Ratio'].idxmax()

# Sort portfolios by volatility
sorted_portfolios = simulated_portfolios.sort_values(by='Volatility').reset_index()

# Use the sorted dataframe to find the portfolios with the minimum and maximum volatility
least_variance_idx = sorted_portfolios.iloc[0]['Simulation']
most_variance_idx = sorted_portfolios.iloc[-1]['Simulation']

# Find the median variance portfolio
median_idx = len(sorted_portfolios) // 2
median_variance_idx = sorted_portfolios.iloc[median_idx]['Simulation']

# Extracting the weights of identified portfolios
optimal_weights = weights_record[:, optimal_idx]
least_variance_weights = weights_record[:, int(least_variance_idx)]
most_variance_weights = weights_record[:, int(most_variance_idx)]
median_variance_weights = weights_record[:, int(median_variance_idx)]

print("Completed: Optimal Weighting Simulation\n")

print("\nStarting: Probability Distribution Generation\n")
# Calculate mean and volatility of daily returns for each asset
daily_mean_returns = daily_returns.mean()
daily_volatility = daily_returns.std()
market_mean_return = market_daily_returns.mean()
market_volatility = market_daily_returns.std()

def run_simulation(weights, length):
    """Runs a Monte Carlo simulation for a given set of weights and time period."""
    fund_value = [10000]
    for _ in range(length):
        individual_asset_returns = np.random.normal(daily_mean_returns, daily_volatility)
        portfolio_return = np.dot(weights, individual_asset_returns)
        fund_value.append(fund_value[-1] * (1 + portfolio_return))
    return fund_value

portfolio_weights = {
    'Optimal Sharpe Ratio': optimal_weights,
    'Least Variance': least_variance_weights,
    'Median Variance': median_variance_weights,
    'Most Variance': most_variance_weights
}

portfolio_results = {name: [] for name in portfolio_weights.keys()}
market_final_values = []

portfolio_metrics = {}
for portfolio_name, weights in portfolio_weights.items():
    final_values = []
    returns = []  # To store the returns of each simulation
    for _ in range(NUM_SIMULATIONS):
        simulated_fund_values = run_simulation(weights, SIMULATION_LENGTH)
        final_value = simulated_fund_values[-1]
        final_values.append(final_value)
        # Calculate return for this simulation and append to returns list
        simulation_return = (final_value / 10000) - 1
        returns.append(simulation_return)
    portfolio_results[portfolio_name] = final_values
    # Calculate the expected return and volatility based on returns
    expected_return = np.mean(returns)
    volatility = np.std(returns)  # Adjust this if annualization is needed
    portfolio_metrics[portfolio_name] = (expected_return, volatility)

# Simulate market performance for comparison
for _ in range(NUM_SIMULATIONS):
    market_fund_value = [10000]
    for _ in range(SIMULATION_LENGTH):
        market_return = np.random.normal(market_mean_return, market_volatility)
        market_fund_value.append(market_fund_value[-1] * (1 + market_return))
    market_final_values.append(market_fund_value[-1])

market_final_values_percent = [(value / 10000 - 1) * 100 for value in market_final_values]
market_expected_return = np.mean(market_final_values) / 10000 - 1
market_volatility = np.std(market_final_values) / 10000

print("Completed: Probability Distribution Generation\n")

print("\nStarting: Plotting\n")
# Visualization setup and plotting probability distributions
plt.figure(figsize=(16, 9))
ax = plt.gca()

plt.gcf().set_facecolor('black')
ax.set_facecolor('black')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100 * y:.0f}%'))

for spine in ax.spines.values():
    spine.set_edgecolor('white')

palette = sns.color_palette("hsv", len(portfolio_weights) + 1)

for i, (portfolio_name, final_values) in enumerate(portfolio_results.items()):
    color = palette[i]
    final_values_percent = [(value / 10000 - 1) * 100 for value in final_values]
    sns.kdeplot(final_values_percent, label=portfolio_name, color=color, ax=ax)
    expected_return, volatility = portfolio_metrics[portfolio_name]
    plt.axvline(x=expected_return * 100, color=color, linestyle='--')
    plt.text(0.02, 0.91 - 0.1 * i,
             f'{portfolio_name}\n  Mean: {expected_return * 100:.2f}%\n  Volatility: {volatility * 100:.2f}%',
             color='white', transform=ax.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor=color, facecolor='black'))

market_color = palette[-1]
sns.kdeplot(market_final_values_percent, label=MARKET_REPRESENTATION, color=market_color, ax=ax)
plt.axvline(x=market_expected_return * 100, color=market_color, linestyle='--')
plt.text(0.02, 0.91 - 0.1 * len(portfolio_weights),
         f'{MARKET_REPRESENTATION}\n  Mean: {market_expected_return * 100:.2f}%\n  Volatility: {market_volatility * 100:.2f}%',
         color='white', transform=ax.transAxes,
         bbox=dict(boxstyle="round,pad=0.3", edgecolor=market_color, facecolor='black'))

plt.xlabel('Final Fund Value (%)')
plt.ylabel('Density')
plt.title('Probability Distributions of Final Fund Values for Different Portfolios', color='white')
plt.legend()

print("Completed: Plotting\n")
plt.show()
