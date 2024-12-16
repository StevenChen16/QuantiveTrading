import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from typing import List, Tuple, Dict
from scipy import stats

class LaplacianAnalyzer:
    def __init__(self, symbols: List[str], start_date: str = None, end_date: str = None,
                 lookback_years: int = 1):
        """
        Initialize the LaplacianAnalyzer with stock symbols and date range.
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols to analyze
        start_date : str, optional
            Start date for analysis (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for analysis (format: 'YYYY-MM-DD')
        lookback_years : int, optional
            Number of years to look back if start_date is not specified
        """
        self.symbols = symbols
        self.end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()
        self.start_date = pd.Timestamp(start_date) if start_date else self.end_date - pd.DateOffset(years=lookback_years)
        self.data = None
        self.normalized_prices = None
        self.prices_array = None
        self.laplacian_results = {}
        
    def fetch_data(self) -> None:
        """
        Fetch stock data and prepare it for analysis.
        """
        df = pd.DataFrame()
        
        # Download data for each stock
        for symbol in self.symbols:
            stock = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
            df[symbol] = stock['Close']
            
        # Fill missing values and normalize
        self.data = df.fillna(method='ffill')
        self.normalized_prices = self.data.div(self.data.iloc[0]) * 100
        self.prices_array = self.normalized_prices.values.T
        
    def multi_scale_laplacian(self, scales: List[float] = [1, 5, 10]) -> Dict[float, np.ndarray]:
        """
        Compute Laplacian at multiple scales.
        
        Parameters:
        -----------
        scales : List[float]
            List of smoothing scales to use
            
        Returns:
        --------
        Dict[float, np.ndarray]
            Dictionary mapping scales to their respective Laplacian results
        """
        results = {}
        for scale in scales:
            # Apply Gaussian smoothing
            smoothed = ndimage.gaussian_filter(self.prices_array, sigma=scale)
            # Compute Laplacian with boundary handling
            padded = np.pad(smoothed, ((1, 1), (1, 1)), mode='reflect')
            lap = ndimage.laplace(padded)
            results[scale] = lap[1:-1, 1:-1]
        
        self.laplacian_results = results
        return results
    
    def compute_risk_metrics(self, scale: float = 1) -> pd.DataFrame:
        """
        Compute various risk metrics based on Laplacian analysis.
        
        Parameters:
        -----------
        scale : float
            Scale at which to compute risk metrics
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing risk metrics for each stock
        """
        laplacian = self.laplacian_results.get(scale)
        if laplacian is None:
            raise ValueError(f"No Laplacian results found for scale {scale}")
            
        metrics = {}
        for i, symbol in enumerate(self.symbols):
            volatility = np.std(self.prices_array[i])
            laplacian_volatility = np.std(laplacian[i])
            combined_risk = np.sqrt(volatility**2 + laplacian_volatility**2)
            
            metrics[symbol] = {
                'Price_Volatility': volatility,
                'Laplacian_Volatility': laplacian_volatility,
                'Combined_Risk': combined_risk
            }
            
        return pd.DataFrame.from_dict(metrics, orient='index')
    
    def analyze_correlation(self, scale: float = 1) -> pd.DataFrame:
        """
        Analyze correlation between Laplacian values and price changes.
        
        Parameters:
        -----------
        scale : float
            Scale at which to compute correlations
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing correlation metrics
        """
        laplacian = self.laplacian_results.get(scale)
        if laplacian is None:
            raise ValueError(f"No Laplacian results found for scale {scale}")
            
        correlations = {}
        for i, symbol in enumerate(self.symbols):
            price_changes = np.diff(self.prices_array[i])
            lap_values = laplacian[i, :-1]
            
            correlation = stats.pearsonr(lap_values, price_changes)[0]
            correlations[symbol] = {
                'Correlation': correlation
            }
            
        return pd.DataFrame.from_dict(correlations, orient='index')
    
    def evaluate_predictive_power(self, forward_days: int = 5, scale: float = 1) -> pd.DataFrame:
        """
        Evaluate the predictive power of Laplacian values.
        
        Parameters:
        -----------
        forward_days : int
            Number of days to look forward
        scale : float
            Scale at which to evaluate predictive power
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing predictive power metrics
        """
        laplacian = self.laplacian_results.get(scale)
        if laplacian is None:
            raise ValueError(f"No Laplacian results found for scale {scale}")
            
        predictions = {}
        for i, symbol in enumerate(self.symbols):
            future_returns = (self.prices_array[i, forward_days:] - 
                            self.prices_array[i, :-forward_days]) / self.prices_array[i, :-forward_days]
            laplacian_subset = laplacian[i, :-forward_days]
            
            correlation = stats.pearsonr(laplacian_subset, future_returns)[0]
            predictions[symbol] = {
                'Predictive_Correlation': correlation
            }
            
        return pd.DataFrame.from_dict(predictions, orient='index')
    
    def detect_anomalies(self, threshold_std: float = 2.0, scale: float = 1) -> pd.DataFrame:
        """
        Detect anomalies in the price movements using Laplacian values.
        
        Parameters:
        -----------
        threshold_std : float
            Number of standard deviations to use as threshold
        scale : float
            Scale at which to detect anomalies
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing detected anomalies
        """
        laplacian = self.laplacian_results.get(scale)
        if laplacian is None:
            raise ValueError(f"No Laplacian results found for scale {scale}")
            
        anomalies = []
        for i, symbol in enumerate(self.symbols):
            threshold = threshold_std * np.std(laplacian[i])
            anomaly_indices = np.where(np.abs(laplacian[i]) > threshold)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    'Symbol': symbol,
                    'Date': self.data.index[idx],
                    'Laplacian_Value': laplacian[i, idx],
                    'Price': self.data.iloc[idx][symbol],
                    'Normalized_Price': self.normalized_prices.iloc[idx][symbol]
                })
                
        return pd.DataFrame(anomalies)
    
    def visualize_analysis(self, scale: float = 1) -> None:
        """
        Create comprehensive visualization of the analysis.
        
        Parameters:
        -----------
        scale : float
            Scale at which to visualize results
        """
        laplacian = self.laplacian_results.get(scale)
        if laplacian is None:
            raise ValueError(f"No Laplacian results found for scale {scale}")
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot normalized prices
        for i, symbol in enumerate(self.symbols):
            ax1.plot(self.normalized_prices.index, self.normalized_prices[symbol], 
                    label=symbol, alpha=0.7)
        ax1.set_title('Normalized Stock Prices (Starting at 100)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normalized Price')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        
        # Plot prices as heatmap
        im2 = ax2.imshow(self.prices_array, aspect='auto', cmap='viridis',
                        extent=[0, len(self.data), 0, len(self.symbols)])
        ax2.set_title('Price Heatmap')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Stock')
        ax2.set_yticks(np.arange(len(self.symbols)) + 0.5)
        ax2.set_yticklabels(self.symbols)
        plt.colorbar(im2, ax=ax2, label='Normalized Price')
        
        # Plot Laplacian
        im3 = ax3.imshow(laplacian, aspect='auto', cmap='coolwarm',
                        extent=[0, len(self.data), 0, len(self.symbols)])
        ax3.set_title(f'Laplacian of Price Surface (Scale: {scale})')
        ax3.set_xlabel('Trading Days')
        ax3.set_ylabel('Stock')
        ax3.set_yticks(np.arange(len(self.symbols)) + 0.5)
        ax3.set_yticklabels(self.symbols)
        plt.colorbar(im3, ax=ax3, label='Laplacian Value')
        
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC']
    analyzer = LaplacianAnalyzer(symbols)
    
    # Fetch and prepare data
    analyzer.fetch_data()
    
    # Compute multi-scale Laplacian
    scales = [1, 5, 10]
    laplacian_results = analyzer.multi_scale_laplacian(scales)
    
    # Compute risk metrics
    risk_metrics = analyzer.compute_risk_metrics(scale=1)
    print("\nRisk Metrics:")
    print(risk_metrics)
    
    # Analyze correlations
    correlations = analyzer.analyze_correlation(scale=1)
    print("\nCorrelations:")
    print(correlations)
    
    # Evaluate predictive power
    predictions = analyzer.evaluate_predictive_power(forward_days=5, scale=1)
    print("\nPredictive Power:")
    print(predictions)
    
    # Detect anomalies
    anomalies = analyzer.detect_anomalies(threshold_std=2.0, scale=1)
    print("\nDetected Anomalies:")
    print(anomalies)
    
    # Visualize results
    analyzer.visualize_analysis(scale=1)

if __name__ == "__main__":
    main()