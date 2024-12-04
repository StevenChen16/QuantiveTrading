import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize

def get_stock_data(symbols, start_date, end_date):
   data = pd.DataFrame()
   for symbol in symbols:
       if symbol == 'B-T-6.250-15052030':
           continue
       ticker = yf.Ticker(symbol.replace('.L', ''))
       hist = ticker.history(start=start_date, end=end_date)['Close']
       if not hist.empty:
           data[symbol] = hist
   return data

def calculate_portfolio_risk(weights, cov_matrix):
   portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
   return np.sqrt(portfolio_variance)

def calculate_marginal_risk_contribution(weights, cov_matrix):
   portfolio_risk = calculate_portfolio_risk(weights, cov_matrix)
   marginal_contrib = np.dot(cov_matrix, weights) / portfolio_risk
   return marginal_contrib

def calculate_expected_returns(price_data):
   returns = price_data.pct_change(fill_method=None)
   return returns.mean() * 252

def calculate_gradient(weights, cov_matrix, expected_returns, target_return):
   n = len(weights)
   # 一阶导数
   first_derivatives = np.zeros(n + 2)  # n个权重 + 2个lambda
   
   # ∂L/∂w_i
   for i in range(n):
       sum_term = 0
       for j in range(n):
           sum_term += weights[j] * cov_matrix[i,j]
       first_derivatives[i] = 2 * sum_term
       
   # ∂L/∂λ₁
   first_derivatives[n] = np.sum(weights) - 1
   
   # ∂L/∂λ₂ 
   first_derivatives[n+1] = np.sum(weights * expected_returns) - target_return
   
   # 二阶导数矩阵 (Hessian)
   second_derivatives = np.zeros((n+2, n+2))
   
   # ∂²L/∂w_i∂w_j = 2σ_ij
   second_derivatives[:n,:n] = 2 * cov_matrix
   
   # ∂²L/∂w_i∂λ₁ = 1
   second_derivatives[:n,n] = 1
   second_derivatives[n,:n] = 1
   
   # ∂²L/∂w_i∂λ₂ = E(R_i)
   second_derivatives[:n,n+1] = expected_returns
   second_derivatives[n+1,:n] = expected_returns
   
   return first_derivatives, second_derivatives

def portfolio_objective(weights, cov_matrix, expected_returns, target_return):
   portfolio_risk = calculate_portfolio_risk(weights, cov_matrix)
   portfolio_return = np.sum(weights * expected_returns)
   return portfolio_risk - 0.1 * (portfolio_return - target_return)**2

def optimize_portfolio(expected_returns, cov_matrix, target_return):
   n_assets = len(expected_returns)
   
   def lagrangian(x, lambda1, lambda2):
       return (portfolio_objective(x, cov_matrix, expected_returns, target_return) + 
               lambda1 * (np.sum(x) - 1) + 
               lambda2 * (np.sum(x * expected_returns) - target_return))
   
   constraints = [
       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
       {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}
   ]
   bounds = tuple((0, 1) for _ in range(n_assets))
   
   initial_weights = np.array([1/n_assets] * n_assets)
   result = minimize(
       portfolio_objective,
       initial_weights,
       args=(cov_matrix, expected_returns, target_return),
       method='SLSQP',
       bounds=bounds,
       constraints=constraints
   )
   return result.x, result.fun, lagrangian

# 主程序
portfolio_df = pd.read_csv('portfolio-简化.csv')
all_symbols = portfolio_df['Symbol'].tolist()

stock_symbols = [s for s in all_symbols if s != 'B-T-6.250-15052030']
weights_dict = dict(zip(portfolio_df['Symbol'], 
                      portfolio_df['weights'].str.rstrip('%').astype(float) / 100))

full_cov_symbols = all_symbols
current_weights = np.array([weights_dict[s] for s in full_cov_symbols])

end_date = datetime.now()
start_date = end_date - timedelta(days=365)
price_data = get_stock_data(stock_symbols, start_date, end_date)

stock_returns = price_data.pct_change(fill_method=None)
stock_cov = stock_returns.cov() * 252

full_cov = np.zeros((len(full_cov_symbols), len(full_cov_symbols)))
treasury_idx = full_cov_symbols.index('B-T-6.250-15052030')
non_treasury_idx = [i for i, s in enumerate(full_cov_symbols) if s != 'B-T-6.250-15052030']

for i, row_idx in enumerate(non_treasury_idx):
   for j, col_idx in enumerate(non_treasury_idx):
       full_cov[row_idx, col_idx] = stock_cov.iloc[i, j]

stock_expected_returns = calculate_expected_returns(price_data)
full_expected_returns = np.zeros(len(full_cov_symbols))
for i, symbol in enumerate(full_cov_symbols):
   if symbol != 'B-T-6.250-15052030':
       full_expected_returns[i] = stock_expected_returns[symbol]
   else:
       full_expected_returns[i] = 0.0625

current_portfolio_risk = calculate_portfolio_risk(current_weights, full_cov)
marginal_contributions = calculate_marginal_risk_contribution(current_weights, full_cov)
risk_contributions = current_weights * marginal_contributions

current_return = np.sum(current_weights * full_expected_returns)
optimal_weights, optimal_value, lagrangian_func = optimize_portfolio(full_expected_returns, full_cov, current_return)

# 计算并打印一阶和二阶导数
first_derivatives, second_derivatives = calculate_gradient(current_weights, full_cov, full_expected_returns, current_return)

# 输出结果
print("=== 协方差矩阵 ===")
df_cov = pd.DataFrame(full_cov, index=full_cov_symbols, columns=full_cov_symbols)
# print(df_cov)
df_cov.to_csv('协方差矩阵.csv')

print("\n协方差矩阵特征值:")
eigenvalues = np.linalg.eigvals(full_cov)
print(eigenvalues)

print("\n=== 一阶导数 ===")
print("∂L/∂w_i:")
for i, symbol in enumerate(full_cov_symbols):
   print(f"{symbol}: {first_derivatives[i]:.4f}")
print(f"\n∂L/∂λ₁: {first_derivatives[-2]:.4f}")
print(f"∂L/∂λ₂: {first_derivatives[-1]:.4f}")

print("\n=== 二阶导数矩阵 ===")
# print(pd.DataFrame(second_derivatives))
pd.DataFrame(second_derivatives).to_csv('二阶导数矩阵.csv')

print("\n=== 预期年化收益率 ===")
for symbol, ret in zip(full_cov_symbols, full_expected_returns):
   print(f"{symbol}: {ret:.2%}")

print(f"\n=== 当前组合统计 ===")
print(f"风险: {current_portfolio_risk:.2%}")
print(f"收益率: {current_return:.2%}")

print("\n=== 边际风险贡献 ===")
for symbol, contrib in zip(full_cov_symbols, marginal_contributions):
   print(f"{symbol}: {contrib:.2%}")

print("\n=== 风险贡献 ===")
for symbol, contrib in zip(full_cov_symbols, risk_contributions):
   print(f"{symbol}: {contrib:.2%}")

print("\n=== 最优组合权重 ===")
for symbol, weight in zip(full_cov_symbols, optimal_weights):
   print(f"{symbol}: {weight:.2%}")

optimal_risk = calculate_portfolio_risk(optimal_weights, full_cov)
optimal_return = np.sum(optimal_weights * full_expected_returns)
print(f"\n=== 优化结果 ===")
print(f"初始风险: {current_portfolio_risk:.2%}")
print(f"最优风险: {optimal_risk:.2%}")
print(f"风险降低: {(current_portfolio_risk - optimal_risk)/current_portfolio_risk:.2%}")
print(f"初始收益: {current_return:.2%}")
print(f"最优收益: {optimal_return:.2%}")