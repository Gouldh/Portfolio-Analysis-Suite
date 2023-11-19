# Portfolio Optimization Using Monte Carlo Simulation with Analysis

## Project Overview
This repository hosts a portfolio analysis tool that leverages Monte Carlo simulations to optimize investment strategies. The tool analyzes a set of stocks over a ten-year period, comparing portfolio performance against the S&P 500 benchmark. Key features include optimal portfolio weighting simulation, probability distribution generation, and  results plotting. Implemented in Python, the project utilizes libraries such as `yfinance` for data retrieval, `numpy` for mathematical operations, `pandas` for data management, `matplotlib` for visualizations, and `seaborn` for enhanced plotting capabilities.

## Features
- **Monte Carlo Simulation**: Conducts 10,000 portfolio weight simulations to determine optimal asset allocation.
- **Performance Projection**: Conducts 1,000 portfolio performance simulations to generate probability distribution for the portfolio.
- **Historical Data Analysis**: Analyzes ten years of stock data for a given portfolio, using stocks like AAPL, JNJ, PG, etc example assets.
- **Benchmarking**: Compares optimized portfolio performance with the S&P 500 index and given portfolio, using both as benchmarks.
- **Visualization**: Utilizes `matplotlib` and `seaborn` for generating plots of probability distributions, portfolio metrics, and comparisons.
- **Portfolio Metrics Calculation**: Calculates annualized returns, volatility, and Sharpe ratios for various portfolio scenarios.
- **Comparisons**: Provides visualizations to compare the performance of optimized, current, and median volatility portfolios.

## Installation
To use this advanced portfolio analysis tool, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gouldh/Portfolio-Optimization-With-Projection.git
   ```
2. Navigate to the repository's directory
   ```bash
   cd Portfolio-Optimization-With-Projection
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the `main.py` script to initiate the analysis:

```bash
python main.py
```

Make sure your Python environment is set up with necessary libraries like `yfinance`, `numpy`, `pandas`, `matplotlib`, and `seaborn`.

## License
This project is made available under the MIT License. Refer to the `LICENSE` file for more details.
