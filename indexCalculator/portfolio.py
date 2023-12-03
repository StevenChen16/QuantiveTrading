import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 加载投资组合数据
file_path = 'portfolio.xlsx'
portfolio_data = pd.read_excel(file_path)

# 提取股票代码和权重
tickers = portfolio_data['Stock Ticker'].dropna().unique().tolist()
weights = portfolio_data.set_index('Stock Ticker')['weight (in whole portfolio)']

# 从 Yahoo Finance 下载历史数据
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# 计算每日回报率
daily_returns = data.pct_change()

# 计算投资组合的加权日回报率
portfolio_returns = (daily_returns * weights).sum(axis=1)

# 下载和计算基准（S&P 500）的日回报率
benchmark_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
benchmark_returns = benchmark_data.pct_change()

# 假设无风险利率（根据您的需求进行更新）
risk_free_rate = 0.02

# 计算年化平均回报率
annual_returns = portfolio_returns.mean() * 252

# 计算整个投资组合的指标
# 夏普比率
std_dev = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (annual_returns - risk_free_rate) / std_dev

# 索诺提比率
negative_std_dev = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
sortino_ratio = (annual_returns - risk_free_rate) / negative_std_dev

# 贝塔系数
portfolio_beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()

# 阿尔法系数
expected_return = risk_free_rate + portfolio_beta * (benchmark_returns.mean() * 252 - risk_free_rate)
alpha = annual_returns - expected_return

# Treynor比率
treynor_ratio = (annual_returns - risk_free_rate) / portfolio_beta

# 风险价值（VaR，95% 置信度）
var_95 = np.percentile(portfolio_returns.dropna(), 5)

# 输出计算结果
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Sortino Ratio: {sortino_ratio}")
print(f"Beta: {portfolio_beta}")
print(f"Alpha: {alpha}")
print(f"Treynor Ratio: {treynor_ratio}")
print(f"VaR (95% Confidence): {var_95}")
