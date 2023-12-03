这是一个开源的量化交易（quantive trading）项目

This is an open source quantitative trading project

#### 1.LSTM

使用LSTM模型进行预测，其模型已经放在目录中。程序通过yfinance库从Yahoo Finance下载数据。

Use LSTM model for prediction, its model has been placed in the directory . Program through the yfinance library to download data from Yahoo Finance.

#### 2.MACD

双均线策略是十分常见的模型，本项目使用python复现了该模型。程序通过yfinance库从Yahoo Finance下载数据。

The Double SMA strategy is a very common model and this project reproduces the model using python. The program downloads data from Yahoo Finance via the yfinance library.

#### 3.布林带(Bollinger Band)

可以绘制布林带以判断购买时机。程序通过yfinance库从Yahoo Finance下载数据。

Bollinger bands can be plotted to determine the timing of a purchase. The program downloads data from Yahoo Finance via the yfinance library.

#### 4.SVM

通过机器学习的svm方法对股票进行预测。程序通过yfinance库从Yahoo Finance下载数据。

Predicts stocks using the svm method of machine learning. The program downloads data from Yahoo Finance via the yfinance library.

#### 5. indexCalculator

众所周知，对于任何量化交易或非量化交易来说，因子都是很重要的判断标准，其中包括夏普比率(Sharpe ratio)，索提诺比率(sortino ratio)，贝塔（Beta/β），阿尔法（alpha/α），特雷诺比率（Treynor Ratio），VaR等。这个程序提供了计算以上这些基本因子的方法，并且支持计算投资组合的因子。程序通过yfinance库从Yahoo Finance下载数据。

As we all know, factors are very important judgment criteria for any quantitative or non-quantitative trading, which include Sharpe ratio, sortino ratio, beta, alpha, Treynor Ratio, VaR, etc. This program provides the ability to calculate these basic criteria for any quantitative or non-quantitative trading. This program provides methods for calculating these basic factors above and supports calculating factors for portfolios. The program downloads data from Yahoo Finance via the yfinance library.

p.s.

```
制作这个calculator是因为我正在参加宾夕法尼亚大学(UPenn)沃顿商学院(Wharton)开设的Wharton Global High School Investment Competition.在这个比赛中，学生们需要使用$100,000进行投资，构建自己的投资组合。而我在我的团队中使用了这些因子进行选股和择时，取得了较为不错的效果。

This calculator was created because I am participating in the Wharton Global High School Investment Competition run by the Wharton School of the University of Pennsylvania (UPenn).In this competition, students are required to invest 100,000 to construct their own In this competition, students were asked to invest 100,000 to build their own portfolios. In this competition, students were asked to invest $100,000 in order to build their portfolios. I used these factors in my team for stock picking and timing, and I had good results.
```







更多中国A股数据详见：

More China A-share data is detailed at:

[https://www.kaggle.com/datasets/stevenchen116/stochchina](https://www.kaggle.com/datasets/stevenchen116/stockchina)