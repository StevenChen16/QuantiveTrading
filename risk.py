import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize

class KalmanFilter:
    def __init__(self, dim_state, dim_obs):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        
        # 初始化状态估计和协方差
        self.state = np.zeros(dim_state)
        self.P = np.eye(dim_state)
        
        # 系统参数
        self.F = np.eye(dim_state)  # 状态转移矩阵
        self.H = np.zeros((dim_obs, dim_state))  # 观测矩阵
        self.Q = np.eye(dim_state) * 0.001  # 过程噪声协方差
        self.R = np.eye(dim_obs) * 0.01  # 测量噪声协方差

    def predict(self):
        # 预测步骤
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state

    def update(self, measurement):
        # 更新步骤
        if measurement is None:  # 处理缺失数据
            return self.state
            
        y = measurement - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.state

def calculate_kalman_returns(price_data):
    """使用卡尔曼滤波估计收益率"""
    returns = price_data.pct_change().dropna()
    n_assets = returns.shape[1]
    
    # 初始化滤波器
    kf = KalmanFilter(dim_state=n_assets, dim_obs=n_assets)
    kf.H = np.eye(n_assets)
    
    # 存储滤波结果
    filtered_returns = np.zeros_like(returns)
    
    # 对每个时间点进行滤波
    for t in range(len(returns)):
        kf.predict()
        measurement = returns.iloc[t].values
        filtered_returns[t] = kf.update(measurement)
    
    return pd.DataFrame(filtered_returns, index=returns.index, columns=returns.columns)

def calculate_kalman_volatility(returns_data):
    """使用卡尔曼滤波估计波动率"""
    n_assets = returns_data.shape[1]
    squared_returns = returns_data ** 2
    
    # 初始化滤波器
    kf = KalmanFilter(dim_state=n_assets, dim_obs=n_assets)
    kf.H = np.eye(n_assets)
    
    # 存储滤波结果
    filtered_variance = np.zeros_like(squared_returns)
    
    # 对每个时间点进行滤波
    for t in range(len(squared_returns)):
        kf.predict()
        measurement = squared_returns.iloc[t].values
        filtered_variance[t] = kf.update(measurement)
    
    # 转换为年化波动率
    filtered_volatility = np.sqrt(filtered_variance * 252)
    return pd.DataFrame(filtered_volatility, index=returns_data.index, columns=returns_data.columns)

def calculate_beta(price_data, market_symbol='^GSPC'):
    # 获取市场数据并处理时区
    market = yf.download(market_symbol, 
                        start=price_data.index[0].tz_localize(None), 
                        end=price_data.index[-1].tz_localize(None))['Adj Close']
    market_returns = market.pct_change().dropna()
    
    betas = {}
    for column in price_data.columns:
        asset_returns = price_data[column].pct_change().dropna()
        # 将时间索引转换为naive datetime
        asset_returns.index = asset_returns.index.tz_localize(None)
        common_dates = asset_returns.index.intersection(market_returns.index)
        
        if len(common_dates) > 0:
            asset_returns_aligned = asset_returns[common_dates]
            market_returns_aligned = market_returns[common_dates]
            beta = np.cov(asset_returns_aligned, market_returns_aligned)[0,1] / np.var(market_returns_aligned)
            betas[column] = beta
    
    portfolio_beta = sum(betas[asset] * weights_dict[asset] 
                        for asset in betas.keys() 
                        if asset in weights_dict)
    return betas, portfolio_beta

def calculate_kalman_beta(price_data, market_symbol='^GSPC'):
    """使用卡尔曼滤波估计时变beta"""
    # 获取市场数据
    market = yf.download(market_symbol, 
                        start=price_data.index[0].tz_localize(None), 
                        end=price_data.index[-1].tz_localize(None))['Adj Close']
    market_returns = market.pct_change().dropna()
    
    asset_returns = price_data.pct_change().dropna()
    asset_returns.index = asset_returns.index.tz_localize(None)
    
    # 对齐数据
    common_dates = asset_returns.index.intersection(market_returns.index)
    asset_returns = asset_returns.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    
    n_assets = len(asset_returns.columns)
    
    # 初始化滤波器 (状态向量包括beta和alpha)
    kf = KalmanFilter(dim_state=2*n_assets, dim_obs=n_assets)
    kf.H = np.zeros((n_assets, 2*n_assets))
    
    # 存储滤波结果
    filtered_betas = np.zeros((len(asset_returns), n_assets))
    filtered_alphas = np.zeros((len(asset_returns), n_assets))
    
    # 对每个时间点进行滤波
    for t in range(len(asset_returns)):
        # 更新观测矩阵
        for i in range(n_assets):
            kf.H[i, 2*i:2*i+2] = [1, market_returns.iloc[t]]
        
        kf.predict()
        measurement = asset_returns.iloc[t].values
        state = kf.update(measurement)
        
        # 提取beta和alpha
        for i in range(n_assets):
            filtered_alphas[t, i] = state[2*i]
            filtered_betas[t, i] = state[2*i+1]
    
    # 转换为DataFrame
    betas_df = pd.DataFrame(filtered_betas, 
                           index=asset_returns.index,
                           columns=asset_returns.columns)
    alphas_df = pd.DataFrame(filtered_alphas,
                            index=asset_returns.index,
                            columns=asset_returns.columns)
    
    return betas_df, alphas_df

def calculate_kalman_risk(weights, price_data):
    """计算基于卡尔曼滤波的组合风险指标"""
    # 估计收益率
    filtered_returns = calculate_kalman_returns(price_data)
    
    # 估计波动率
    filtered_volatility = calculate_kalman_volatility(filtered_returns)
    
    # 估计beta
    filtered_betas, filtered_alphas = calculate_kalman_beta(price_data)
    
    # 计算最新风险指标
    latest_returns = filtered_returns.iloc[-1]
    latest_volatility = filtered_volatility.iloc[-1]
    latest_betas = filtered_betas.iloc[-1]
    
    # 计算组合层面指标
    portfolio_return = np.sum(weights * latest_returns)
    portfolio_vol = np.sqrt(np.sum(weights**2 * latest_volatility**2))
    portfolio_beta = np.sum(weights * latest_betas)
    
    return {
        'returns': portfolio_return,
        'volatility': portfolio_vol,
        'beta': portfolio_beta,
        'filtered_returns': filtered_returns,
        'filtered_volatility': filtered_volatility,
        'filtered_betas': filtered_betas,
        'filtered_alphas': filtered_alphas
    }

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
    first_derivatives = np.zeros(n + 2)
    
    for i in range(n):
        sum_term = 0
        for j in range(n):
            sum_term += weights[j] * cov_matrix[i,j]
        first_derivatives[i] = 2 * sum_term
    
    first_derivatives[n] = np.sum(weights) - 1
    first_derivatives[n+1] = np.sum(weights * expected_returns) - target_return
    
    second_derivatives = np.zeros((n+2, n+2))
    second_derivatives[:n,:n] = 2 * cov_matrix
    second_derivatives[:n,n] = 1
    second_derivatives[n,:n] = 1
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

def calculate_var(weights, returns, confidence_level=0.95, periods=252):
    portfolio_returns = returns.dot(weights)
    var_daily = -np.percentile(portfolio_returns, (1-confidence_level)*100)
    var_annual = var_daily * np.sqrt(periods)
    return var_annual

def calculate_risk_metrics(weights, cov_matrix):
    total_risk = calculate_portfolio_risk(weights, cov_matrix)
    component_risks = np.zeros(len(weights))
    
    for i in range(len(weights)):
        for j in range(len(weights)):
            component_risks[i] += weights[i] * weights[j] * cov_matrix[i,j]
            
    risk_decomp = component_risks / total_risk
    
    total_individual_risk = np.sqrt(np.sum(weights**2 * np.diag(cov_matrix)))
    diversification_effect = 1 - total_risk/total_individual_risk
    
    return risk_decomp, diversification_effect

# 主程序
if __name__ == "__main__":
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
    
    # 使用卡尔曼滤波计算风险指标
    kalman_risk = calculate_kalman_risk(current_weights[:len(price_data.columns)], price_data)
    
    print("\n=== 卡尔曼滤波估计结果 ===")
    print(f"组合预期收益率: {kalman_risk['returns']:.2%}")
    print(f"组合波动率: {kalman_risk['volatility']:.2%}")
    print(f"组合Beta: {kalman_risk['beta']:.2f}")
    
    print("\n=== 各资产Kalman Filter估计结果 ===")
    print("\n个股Beta估计:")
    latest_betas = kalman_risk['filtered_betas'].iloc[-1]
    for symbol in price_data.columns:
        print(f"{symbol}: {latest_betas[symbol]:.2f}")
    
    print("\n个股波动率估计:")
    latest_vols = kalman_risk['filtered_volatility'].iloc[-1]
    for symbol in price_data.columns:
        print(f"{symbol}: {latest_vols[symbol]:.2%}")
    
    # 传统方法计算
    stock_returns = price_data.pct_change().dropna()
    stock_cov = stock_returns.cov() * 252
    
    full_cov = np.zeros((len(full_cov_symbols), len(full_cov_symbols)))
    treasury_idx = full_cov_symbols.index('B-T-6.250-15052030')
    non_treasury_idx = [i for i, s in enumerate(full_cov_symbols) 
                       if s != 'B-T-6.250-15052030']
    
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
    optimal_weights, optimal_value, lagrangian_func = optimize_portfolio(
        full_expected_returns, full_cov, current_return)
    
    # 计算并打印一阶和二阶导数
    first_derivatives, second_derivatives = calculate_gradient(
        current_weights, full_cov, full_expected_returns, current_return)
    
    print("\n=== 传统方法 vs Kalman Filter对比 ===")
    print("风险估计:")
    print(f"传统方法: {current_portfolio_risk:.2%}")
    print(f"Kalman Filter: {kalman_risk['volatility']:.2%}")
    
    print("\n收益率估计:")
    print(f"传统方法: {current_return:.2%}")
    print(f"Kalman Filter: {kalman_risk['returns']:.2%}")
    
    # Beta对比
    traditional_betas, traditional_portfolio_beta = calculate_beta(price_data)
    print("\nBeta估计:")
    print(f"传统方法组合Beta: {traditional_portfolio_beta:.2f}")
    print(f"Kalman Filter组合Beta: {kalman_risk['beta']:.2f}")
    
    # 计算VaR
    portfolio_var = calculate_var(current_weights[:len(stock_returns.columns)], 
                                stock_returns)
    
    # 风险分解
    risk_decomp, div_effect = calculate_risk_metrics(current_weights, full_cov)
    
    print("\n=== 风险指标汇总 ===")
    print(f"VaR (95%): {portfolio_var:.2%}")
    print(f"风险分散效应: {div_effect:.2%}")
    
    # 输出结果到CSV
    print("\n=== 保存结果到CSV ===")
    
    # 保存Kalman Filter估计结果
    kalman_results = pd.DataFrame({
        'Symbol': price_data.columns,
        'KF_Beta': latest_betas,
        'KF_Volatility': latest_vols,
        'Traditional_Beta': [traditional_betas.get(s, np.nan) 
                           for s in price_data.columns],
        'Weight': [weights_dict.get(s, np.nan) for s in price_data.columns]
    })
    kalman_results.to_csv('kalman_filter_results.csv')
    
    # 保存时间序列数据
    kalman_risk['filtered_betas'].to_csv('kalman_betas_ts.csv')
    kalman_risk['filtered_volatility'].to_csv('kalman_volatility_ts.csv')
    kalman_risk['filtered_returns'].to_csv('kalman_returns_ts.csv')

    print("结果已保存到CSV文件。")