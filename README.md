# Comprehensive Portfolio Analysis Suite

## Project Overview
This repository hosts a small portfolio analysis suite, designed to provide deep insights into investment strategies. The suite consists of two main components: a portfolio optimizerand a backtesting tool. The optimizer uses Monte Carlo simulations for asset allocation, while the backtester assesses strategies against historical data.

## Features
- **Portfolio Optimization**: The optimizer conducts simulations to find the ideal asset allocation, maximizing returns and minimizing risks.
- **Strategy Backtesting**: The backtester evaluates investment strategies against historical market data, providing a realistic performance assessment.
- **Visualization Tools**: Both tools use `matplotlib` and `seaborn` for generating insightful plots, such as probability distributions, asset allocations, and performance metrics.
- **Comprehensive Data Analysis**: Analyzes a wide range of stock data over extensive periods, employing tools like `numpy` and `pandas` for robust data handling.
- **Interactive Analysis**: Users can customize analysis parameters to suit different investment scenarios and strategies.
- **Historical Benchmarking**: Compares portfolio performance against major benchmarks like the S&P 500, offering a relative performance perspective.

## Libraries
- `yfinance`: Primary source for downloading stock and market data. It is used across both modules to fetch historical stock prices and other financial data for analysis.
- `numpy`: Utilized for array and numerical computations. It is particularly used in the optimization process for handling arrays and performing mathematical operations.
- `matplotlib` and `seaborn`: Both crucial for data visualization. While matplotlib is used for plotting graphs and charts, seaborn enhances these visualizations with more attractive and informative statistical graphics.
- `pandas`: Essential for data manipulation and transformation. It is used to handle dataframes, perform data cleaning, and prepare datasets for analysis and visualization.


## Installation
To use the Portfolio Analysis Suite, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gouldh/Portfolio-Analysis-Suite.git
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
To utilize the Portfolio Analysis Suite, you need to run the respective scripts for portfolio optimization and backtesting.

### Portfolio Optimization
Run the `optimizer.py` script to initiate the portfolio optimization process:
```bash
python optimizer.py
```
This script performs Monte Carlo simulations to determine the optimal asset allocation, aiming to maximize returns and minimize risks.

### Strategy Backtesting
Execute the backtester.py script to conduct backtesting of your investment strategies:
```bash
python backtester.py
```
This tool assesses investment strategies against historical market data, providing a realistic assessment of potential performance and risk.

## Sample Output
Below is an example of the output produced by running the code with sample input parameters. The first chart shows the result of running `optimizer.py`, and the second chart shows the result of running `backtester.py`.
![Optimizer Sample Output](https://github.com/Gouldh/Portfolio-Analysis-Suite/blob/main/Portfolio%20Analysis%20Suite%20Optimizer%20Sample%20Output.png)
![Backtester Sample Output](https://github.com/Gouldh/Portfolio-Analysis-Suite/blob/main/Portfolio%20Analysis%20Suite%20Backtester%20Sample%20Output.png)

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

**Author**: Hunter Gould         
**Date**: 11/26/2023
