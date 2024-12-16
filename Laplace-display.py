import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd

# Define stock symbols
stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC']

# Download data
# Use one year of daily data
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=1)

# Create empty DataFrame to store prices
df = pd.DataFrame()

# Download data for each stock
for symbol in stocks:
    stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
    df[symbol] = stock['Close']

# Fill any missing values using forward fill
df = df.fillna(method='ffill')

# Normalize prices to start from 100 for better comparison
normalized_prices = df.div(df.iloc[0]) * 100

# Convert to numpy array for Laplacian calculation
prices_array = normalized_prices.values.T  # Shape: (n_stocks, n_times)

# Calculate Laplacian
laplacian = ndimage.laplace(prices_array)

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

# Plot normalized prices
for i, symbol in enumerate(stocks):
    ax1.plot(normalized_prices.index, normalized_prices[symbol], 
             label=symbol, alpha=0.7)
ax1.set_title('Normalized Stock Prices (Starting at 100)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Normalized Price')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)

# Plot prices as a heatmap
im2 = ax2.imshow(prices_array, aspect='auto', cmap='viridis',
                 extent=[0, len(df), 0, len(stocks)])
ax2.set_title('Price Heatmap')
ax2.set_xlabel('Trading Days')
ax2.set_ylabel('Stock')
ax2.set_yticks(np.arange(len(stocks)) + 0.5)
ax2.set_yticklabels(stocks)
plt.colorbar(im2, ax=ax2, label='Normalized Price')

# Plot Laplacian
im3 = ax3.imshow(laplacian, aspect='auto', cmap='coolwarm',
                 extent=[0, len(df), 0, len(stocks)])
ax3.set_title('Laplacian of Price Surface')
ax3.set_xlabel('Trading Days')
ax3.set_ylabel('Stock')
ax3.set_yticks(np.arange(len(stocks)) + 0.5)
ax3.set_yticklabels(stocks)
plt.colorbar(im3, ax=ax3, label='Laplacian Value')

plt.tight_layout()
plt.show()

# Print some basic statistics about the Laplacian values
print("\nLaplacian Statistics:")
print(f"Mean Laplacian value: {np.mean(laplacian):.4f}")
print(f"Standard deviation: {np.std(laplacian):.4f}")
print(f"Min value: {np.min(laplacian):.4f}")
print(f"Max value: {np.max(laplacian):.4f}")

# Find the most significant anomalies
threshold = 2 * np.std(laplacian)
anomalies = np.where(np.abs(laplacian) > threshold)
print("\nSignificant price anomalies (beyond 2 standard deviations):")
for stock_idx, time_idx in zip(*anomalies):
    date = df.index[time_idx]
    stock = stocks[stock_idx]
    laplacian_value = laplacian[stock_idx, time_idx]
    print(f"Stock: {stock}, Date: {date.strftime('%Y-%m-%d')}, Laplacian: {laplacian_value:.4f}")