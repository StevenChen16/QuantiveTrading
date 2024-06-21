# Stock Prediction and Quantitative Trading Project

This project combines two related initiatives: a CNN-based stock prediction model and a collection of quantitative trading strategies and tools. It aims to provide a comprehensive suite for stock analysis, prediction, and trading.
Click here to read ([Chinese version](https://github.com/StevenChen16/QuantiveTrading/blob/main/REDME-zh.md))

## Project Overview

The project consists of the following main components:

1. CNN-based Stock Prediction
2. Various Trading Strategies (LSTM, MACD, Bollinger Bands, SVM)
3. Index and Factor Calculator
4. Data Preprocessing and Analysis Tools

## Main Files

### CNN Stock Prediction
- `cnn-big.ipynb`: Main CNN model training and evaluation code
- `resnet.ipynb`: Experiments with ResNet architecture
- `autoencoder.ipynb`: Autoencoder experiments
- `bt-multi-model.py`: Multi-model backtesting code

### Quantitative Trading Strategies
- LSTM model implementation
- MACD (Moving Average Convergence Divergence) strategy
- Bollinger Bands implementation
- SVM (Support Vector Machine) prediction model

### Tools
- `indexCalculator`: Calculates various financial indices and factors

## Requirements

The project dependencies include:

- pandas
- numpy 
- scikit-learn
- tqdm
- tensorflow
- matplotlib
- yfinance

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. CNN Stock Prediction:
   - Run Jupyter notebooks to train and evaluate models.
   - Use `bt-multi-model.py` for backtesting.

2. Quantitative Trading Strategies:
   - Each strategy is implemented in its own script or notebook.
   - Data is downloaded from Yahoo Finance using the yfinance library.

3. Index Calculator:
   - Use this tool to calculate important financial factors such as Sharpe ratio, Sortino ratio, Beta, and Alpha for individual stocks or portfolios.

## Model Architectures

The project uses various model architectures, including:

- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM)
- ResNet
- Autoencoder
- Support Vector Machine (SVM)

## Data Sources

- Yahoo Finance (via yfinance library)
- For more China A-share data, refer to:
  - [Kaggle Dataset](https://www.kaggle.com/datasets/stevenchen116/stochchina)
  - [Hugging Face Dataset](https://huggingface.co/datasets/StevenChen16/Stock-China-daily)

## Results

Model performance and backtesting results can be found in the respective notebooks and scripts.

## Future Work

- Experiment with more feature engineering
- Optimize model architectures
- Implement additional backtesting strategies
- Integrate more data sources

## Contributing

Issues, suggestions for improvement, and pull requests are welcome!

## Contact

For inquiries about per-second data or other questions, please contact: [i@stevenchen.site](mailto:i@stevenchen.site)

## License

MIT License

Copyright (c) [2023-2024] [Steven Chen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.







更多中国A股数据详见：

More China A-share data is detailed at:

[https://www.kaggle.com/datasets/stevenchen116/stochchina](https://www.kaggle.com/datasets/stevenchen116/stockchina)
[huggingface](https://huggingface.co/datasets/StevenChen16/Stock-China-daily)


如果您需要以秒为单位的数据，请通过邮箱与我联系：
If you need data on a per-second basis, please contact me via email:
[i@stevenchen.site](mailto:i@stevenchen.site)
