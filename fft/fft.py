import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hilbert, find_peaks
import pywt
from datetime import datetime, timedelta
from matplotlib import rcParams
import yfinance as yf
import sys

# 设置字体为 SimHei 或其他支持中文的字体
rcParams['font.family'] = 'Microsoft YaHei'

# 避免负号显示为方块
rcParams['axes.unicode_minus'] = False

class StockSpectralAnalysis:
    def __init__(self, data):
        """初始化分析器"""
        self.load_data(data)
        self.compute_basic_metrics()

    def load_data(self, data):
        """加载数据"""
        self.df = data.copy()  # 创建数据的副本
        
        # 如果Date是索引，将其重置为列
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()
        
        # 确保Date列是datetime类型
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date')
        
        # 计算对数收益率
        self.df['log_return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df = self.df.dropna()
        
        
    def perform_fft(self, filter_threshold=None):
        """执行傅里叶变换，可选择性地过滤高频成分"""
        # 准备数据
        returns = self.df['log_return'].values
        n = len(returns)
        
        # 执行FFT
        fft_result = fft(returns)
        freqs = fftfreq(n, d=1)
        
        # 如果指定了过滤阈值，过滤高频成分
        if filter_threshold is not None:
            # 创建低通滤波器
            filter_mask = np.abs(freqs) < filter_threshold
            fft_result_filtered = fft_result * filter_mask
            
            # 执行逆傅里叶变换获取过滤后的收益率
            filtered_returns = np.real(ifft(fft_result_filtered))
            self.df['filtered_returns'] = filtered_returns
            
            # 从过滤后的收益率重建价格序列
            self.df['filtered_price'] = self.df['Close'].iloc[0] * np.exp(filtered_returns.cumsum())
        
        # 计算功率谱
        power_spectrum = np.abs(fft_result)**2
        
        # 只保留正频率部分
        mask = freqs > 0
        self.periods = 1/freqs[mask]
        self.power_spectrum = power_spectrum[mask]
        
        return self.periods, self.power_spectrum
    
    def filter_high_frequency(self, cutoff_period=21):
        """
        过滤高频成分
        参数:
            cutoff_period: 截止周期（天），高于此频率的成分将被过滤
        """
        filter_threshold = 1/cutoff_period  # 将周期转换为频率
        self.perform_fft(filter_threshold=filter_threshold)
        return self.df['filtered_price']
    
    def find_significant_periods(self, n_peaks=5):
        """找出显著周期"""
        peaks, _ = find_peaks(self.power_spectrum)
        peak_periods = self.periods[peaks]
        peak_powers = self.power_spectrum[peaks]
        
        # 按功率大小排序
        significant_indices = np.argsort(peak_powers)[-n_peaks:]
        
        self.sig_periods = peak_periods[significant_indices]
        self.sig_powers = peak_powers[significant_indices]
        
        return self.sig_periods, self.sig_powers
    
    def wavelet_analysis(self, scales=np.arange(1,128)):
        """小波分析"""
        returns = self.df['log_return'].values
        self.coefficients, self.frequencies = pywt.cwt(returns, scales, 'morl')
        return self.coefficients, self.frequencies
    
    def hilbert_phase_analysis(self):
        """希尔伯特变换相位分析"""
        returns = self.df['log_return'].values
        analytic_signal = hilbert(returns)
        self.amplitude = np.abs(analytic_signal)
        self.phase = np.angle(analytic_signal)
        self.inst_frequency = np.diff(self.phase) / (2.0*np.pi)
        
        return self.amplitude, self.phase, self.inst_frequency
    
    def detect_regime_changes(self):
        """检测市场状态变化"""
        # 使用小波变换的能量谱检测
        energy = np.sum(np.abs(self.coefficients)**2, axis=0)
        threshold = np.mean(energy) + 2*np.std(energy)
        regime_changes = np.where(energy > threshold)[0]
        
        # 转换为日期
        change_dates = [self.df.index[i] for i in regime_changes]
        return change_dates, energy, threshold
    
    def plot_comprehensive_analysis(self, show_filtered=True):
        """绘制综合分析图，包括过滤后的价格"""
        # 准备数据
        if show_filtered:
            self.filter_high_frequency()  # 默认使用21天作为截止周期
        else:
            self.perform_fft()
            
        self.find_significant_periods()
        self.wavelet_analysis()
        self.hilbert_phase_analysis()
        change_dates, energy, threshold = self.detect_regime_changes()
        
        # 创建图形
        fig = plt.figure(figsize=(15, 20))
        
        # 1. 价格和移动平均线
        ax1 = plt.subplot(511)
        ax1.plot(self.df['Date'], self.df['Close'], label='原始价格', alpha=0.7)
        if show_filtered and 'filtered_price' in self.df.columns:
            ax1.plot(self.df['Date'], self.df['filtered_price'], 
                    label='过滤后价格', color='red', linewidth=2)
        ax1.plot(self.df['Date'], self.df['MA21'], label='21日均线')
        ax1.plot(self.df['Date'], self.df['MA63'], label='63日均线')
        ax1.set_title('价格走势比较')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 对数收益率
        ax2 = plt.subplot(512)
        ax2.plot(self.df['Date'], self.df['log_return'])
        ax2.set_title('对数收益率')
        ax2.grid(True)
        
        # 3. 傅里叶分析
        ax3 = plt.subplot(513)
        ax3.plot(self.periods, self.power_spectrum)
        ax3.scatter(self.sig_periods, self.sig_powers, color='red', marker='x')
        ax3.set_title('频谱分析')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 4. 小波分析
        ax4 = plt.subplot(514)
        im = ax4.imshow(np.abs(self.coefficients), aspect='auto', cmap='jet')
        ax4.set_title('小波分析（时频图）')
        plt.colorbar(im, ax=ax4)
        
        # 5. 相位分析
        ax5 = plt.subplot(515)
        ax5.plot(self.df['Date'][1:], self.inst_frequency)
        ax5.set_title('瞬时频率（相位变化率）')
        ax5.grid(True)
        
        plt.tight_layout()
        return plt

    def analyze_with_different_filters(self, periods=[5, 21, 63]):
        """使用不同的过滤周期进行分析"""
        plt.figure(figsize=(15, 8))
        
        # 绘制原始价格
        plt.plot(self.df['Date'], self.df['Close'], 
                label='原始价格', alpha=0.5, color='gray')
        
        # 使用不同的过滤周期
        colors = ['blue', 'green', 'red']
        for period, color in zip(periods, colors):
            filtered_prices = self.filter_high_frequency(cutoff_period=period)
            plt.plot(self.df['Date'], filtered_prices, 
                    label=f'过滤周期 {period}天', color=color)
        
        plt.title('不同过滤周期的价格对比')
        plt.legend()
        plt.grid(True)
        return plt

    def compute_basic_metrics(self):
        """计算基本指标，使用FFT滤波替代移动平均"""
        # 计算21日、63日和252日滤波后的价格序列
        self.df['FFT21'] = self.filter_high_frequency(cutoff_period=21)
        self.df['FFT63'] = self.filter_high_frequency(cutoff_period=63)
        self.df['FFT252'] = self.filter_high_frequency(cutoff_period=252)
        
        """计算基本指标"""
        # 计算移动平均
        self.df['MA21'] = self.df['Close'].rolling(window=21).mean()
        self.df['MA63'] = self.df['Close'].rolling(window=63).mean()
        self.df['MA252'] = self.df['Close'].rolling(window=252).mean()

        # 计算波动率
        self.df['vol_21'] = self.df['log_return'].rolling(window=21).std() * np.sqrt(252)

    def get_trading_signals(self):
        """基于FFT滤波生成交易信号"""
        signals = pd.DataFrame(index=self.df.index)
        
        # 使用FFT滤波后的价格序列判断趋势
        signals['trend'] = np.where(self.df['FFT21'] > self.df['FFT63'], 1, -1)
        
        # 波动率信号
        vol_mean = self.df['vol_21'].mean()
        signals['volatility'] = np.where(self.df['vol_21'] > vol_mean, 'high', 'low')
        
        # 相位信号
        analytic_signal = hilbert(self.df['log_return'].values)
        phase = np.angle(analytic_signal)
        phase_diff = np.diff(phase)
        phase_diff = np.append(phase_diff, phase_diff[-1])
        signals['phase'] = np.where(phase_diff > 0, 1, -1)
        
        return signals
    
    def print_analysis_summary(self):
        """打印分析摘要"""
        # 执行FFT分析和寻找显著周期
        self.perform_fft()
        self.find_significant_periods()
        
        # 计算各周期滤波
        fft21 = self.filter_high_frequency(cutoff_period=21)
        fft63 = self.filter_high_frequency(cutoff_period=63)
        
        print("\n=== 股票分析摘要 ===")
        
        # 基本统计
        print("\n1. 基本统计:")
        print(f"分析周期: {self.df['Date'].iloc[0].strftime('%Y-%m-%d')} 至 {self.df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"总交易日数: {len(self.df)}")
        print(f"当前价格: {self.df['Close'].iloc[-1]:.2f}")
        print(f"21日波动率: {self.df['vol_21'].iloc[-1]*100:.2f}%")
        
        # 显著周期
        print("\n2. 主要周期:")
        for period, power in zip(self.sig_periods, self.sig_powers):
            print(f"周期: {period:.1f}天, 相对强度: {power:.2e}")
        
        # 趋势分析
        print("\n3. 趋势分析:")
        current_trend = "上升" if fft21.iloc[-1] > fft63.iloc[-1] else "下降"
        print(f"当前趋势: {current_trend}")
        
        # 市场状态
        print("\n4. 市场状态:")
        current_vol = self.df['vol_21'].iloc[-1]
        avg_vol = self.df['vol_21'].mean()
        print(f"当前波动率状态: {'高波动' if current_vol > avg_vol else '低波动'}")
        
        return

def download_finance_data(ticker):
    """下载金融数据"""
    # 计算日期范围
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        # 使用yfinance下载数据
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # 检查数据是否为空
        if data.empty:
            print(f"无法获取 {ticker} 的数据")
            sys.exit(1)
            
        return data
        
    except Exception as e:
        print(f"下载数据时出错: {str(e)}")
        sys.exit(1)

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("Usage: python script.py <ticker>")
        sys.exit(1)
        
    ticker = sys.argv[1]
    
    # 下载数据
    data = download_finance_data(ticker)
    
    # 使用示例
    analyzer = StockSpectralAnalysis(data)
    
    # 基本分析
    analyzer.print_analysis_summary()
    
    # 使用不同的过滤周期进行分析
    plt = analyzer.analyze_with_different_filters(periods=[5, 21, 63])
    plt.show()
    
    # 显示综合分析图
    plt = analyzer.plot_comprehensive_analysis(show_filtered=True)
    plt.show()

if __name__ == "__main__":
    main()