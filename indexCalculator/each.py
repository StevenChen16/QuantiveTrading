import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Load the portfolio data
file_path = 'portfolio.xlsx'
portfolio_data = pd.read_excel(file_path)

# Extract stock tickers
tickers = portfolio_data['Stock Ticker'].dropna().unique().tolist()

# Download historical data from Yahoo Finance
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns for each stock
daily_returns = data.pct_change()

# Download and calculate daily returns for the benchmark (S&P 500)
benchmark_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
benchmark_returns = benchmark_data.pct_change()

# Assume a risk-free rate (you may update it as per your requirement)
risk_free_rate = 0.02

# Calculate annual average returns for each stock
annual_returns = daily_returns.mean() * 252

# Calculate Sharpe Ratio, Sortino Ratio, Beta, Alpha, Treynor Ratio, and VaR for each stock
metrics = pd.DataFrame(index=tickers)
for ticker in tickers:
    # Sharpe Ratio
    std_dev = daily_returns[ticker].std() * np.sqrt(252)
    metrics.loc[ticker, 'Sharpe Ratio'] = (annual_returns[ticker] - risk_free_rate) / std_dev
    
    # Sortino Ratio
    negative_std_dev = daily_returns[daily_returns[ticker] < 0][ticker].std() * np.sqrt(252)
    metrics.loc[ticker, 'Sortino Ratio'] = (annual_returns[ticker] - risk_free_rate) / negative_std_dev
    
    # Beta
    cov = daily_returns[ticker].cov(benchmark_returns)
    var = benchmark_returns.var()
    metrics.loc[ticker, 'Beta'] = cov / var
    
    # Alpha
    expected_return = risk_free_rate + metrics.loc[ticker, 'Beta'] * (benchmark_returns.mean() * 252 - risk_free_rate)
    metrics.loc[ticker, 'Alpha'] = annual_returns[ticker] - expected_return
    
    # Treynor Ratio
    metrics.loc[ticker, 'Treynor Ratio'] = (annual_returns[ticker] - risk_free_rate) / metrics.loc[ticker, 'Beta']
    
    # VaR (95% confidence)
    metrics.loc[ticker, 'VaR'] = np.percentile(daily_returns[ticker].dropna(), 5)

metrics.to_csv("index.csv", index=True)
# Output the calculated metrics
print(metrics)
