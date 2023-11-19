# Comprehensive Portfolio Analysis Suite

## Project Overview
This repository hosts a comprehensive portfolio analysis suite, designed to provide deep insights into investment strategies. The suite consists of two main components: a portfolio optimizer (`optimizer.py`) and a backtesting tool (`backtester.py`). The optimizer uses Monte Carlo simulations for asset allocation, while the backtester assesses strategies against historical data. Built with Python, the suite leverages libraries such as `yfinance` for financial data, `numpy` and `pandas` for numerical analysis and data management, and `matplotlib` and `seaborn` for advanced visualizations.

## Features
- **Portfolio Optimization**: The optimizer conducts simulations to find the ideal asset allocation, maximizing returns and minimizing risks.
- **Strategy Backtesting**: The backtester evaluates investment strategies against historical market data, providing a realistic performance assessment.
- **Visualization Tools**: Both tools use `matplotlib` and `seaborn` for generating insightful plots, such as probability distributions, asset allocations, and performance metrics.
- **Comprehensive Data Analysis**: Analyzes a wide range of stock data over extensive periods, employing tools like `numpy` and `pandas` for robust data handling.
- **Interactive Analysis**: Users can customize analysis parameters to suit different investment scenarios and strategies.
- **Historical Benchmarking**: Compares portfolio performance against major benchmarks like the S&P 500, offering a relative performance perspective.

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

Ensure your Python environment is set up with necessary libraries such as `yfinance`, `numpy`, `pandas`, `matplotlib`, and `seaborn`.

## Contributing
Contributions to the Portfolio Analysis Suite are welcome.

## Support
If you encounter any issues or have questions, feel free to open an issue on the GitHub repository.

## Authors
- Hunter Gould 

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
