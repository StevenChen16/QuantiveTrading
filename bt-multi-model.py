import backtrader as bt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 明确指定 mse 函数
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(),
}

# 加载训练好的短期和长期模型
short_term_model = tf.keras.models.load_model("model/stock_prediction_cnn_model_30_10_2400.h5", custom_objects=custom_objects)
long_term_model = tf.keras.models.load_model("model/stock_prediction_cnn_model_60_30_1400.h5", custom_objects=custom_objects)

# 定义回测策略
class MultiStockStrategy(bt.Strategy):
    def __init__(self):
        self.stocks = self.datas
        self.short_time_window = 30  # 短期模型的时间窗口大小
        self.long_time_window = 60   # 长期模型的时间窗口大小
        self.recent_short_data = {stock._name: [] for stock in self.stocks}
        self.recent_long_data = {stock._name: [] for stock in self.stocks}

    def preprocess_data(self, data):
        scaler = MinMaxScaler()
        # 处理 NaN 和 Inf 值
        data = np.nan_to_num(data)
        return scaler.fit_transform(data)

    def next(self):
        predictions_short = {}
        predictions_long = {}

        for stock in self.stocks:
            data_name = stock._name
            self.recent_short_data[data_name].append([
                stock.open[0], stock.close[0], stock.high[0], stock.low[0],
                stock.volume[0], stock.money[0], stock.avg[0], stock.high_limit[0],
                stock.low_limit[0], stock.pre_close[0], stock.paused[0], stock.factor[0],
                stock.MA5[0], stock.MA10[0], stock.RSI[0], stock.WilliamsR[0]
            ])

            self.recent_long_data[data_name].append([
                stock.open[0], stock.close[0], stock.high[0], stock.low[0],
                stock.volume[0], stock.money[0], stock.avg[0], stock.high_limit[0],
                stock.low_limit[0], stock.pre_close[0], stock.paused[0], stock.factor[0],
                stock.MA5[0], stock.MA10[0], stock.RSI[0], stock.WilliamsR[0]
            ])

            # 确保收集到足够的短期数据
            if len(self.recent_short_data[data_name]) > self.short_time_window:
                self.recent_short_data[data_name].pop(0)

            # 确保收集到足够的长期数据
            if len(self.recent_long_data[data_name]) > self.long_time_window:
                self.recent_long_data[data_name].pop(0)

            # 进行短期预测
            if len(self.recent_short_data[data_name]) == self.short_time_window:
                short_data_np = np.array(self.recent_short_data[data_name])
                short_data_scaled = self.preprocess_data(short_data_np)
                x_short_data = np.expand_dims(short_data_scaled, axis=0)
                x_short_data = np.expand_dims(x_short_data, axis=-1)
                predictions_short[data_name] = short_term_model.predict(x_short_data)[0][0]

            # 进行长期预测
            if len(self.recent_long_data[data_name]) == self.long_time_window:
                long_data_np = np.array(self.recent_long_data[data_name])
                long_data_scaled = self.preprocess_data(long_data_np)
                x_long_data = np.expand_dims(long_data_scaled, axis=0)
                x_long_data = np.expand_dims(x_long_data, axis=-1)
                predictions_long[data_name] = long_term_model.predict(x_long_data)[0][0]

        # 计算综合买入股票的权重
        buy_stocks = {k: (v + predictions_long[k]) / 2 for k, v in predictions_short.items() if v > 0.02 and k in predictions_long and predictions_long[k] > 0.02}
        total_weight = sum(buy_stocks.values())

        # 卖出预测亏损的股票
        for stock in self.stocks:
            data_name = stock._name
            if (predictions_short.get(data_name, 0) < 0 or predictions_long.get(data_name, 0) < 0) and self.getposition(stock).size > 0:
                self.sell(data=stock, size=self.getposition(stock).size)

        # 按权重买入股票
        for stock in self.stocks:
            data_name = stock._name
            if data_name in buy_stocks:
                weight = buy_stocks[data_name] / total_weight
                cash = self.broker.get_cash()
                # 检查 stock.close[0] 是否为 NaN
                if not np.isnan(stock.close[0]):
                    buy_qty = int((cash * weight) / stock.close[0])
                    if buy_qty > 0:
                        self.buy(data=stock, size=buy_qty)

# 加载股票数据
class CustomCSVData(bt.feeds.GenericCSVData):
    lines = (
        'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor', 'MA5', 'MA10', 'RSI', 'WilliamsR'
    )

    params = (
        ('money', 6),
        ('avg', 7),
        ('high_limit', 8),
        ('low_limit', 9),
        ('pre_close', 10),
        ('paused', 11),
        ('factor', 12),
        ('MA5', 13),
        ('MA10', 14),
        ('RSI', 15),
        ('WilliamsR', 16),
    )

# 初始化Cerebro引擎
cerebro = bt.Cerebro()

# 设置自定义时间范围
fromdate = pd.Timestamp('2020-01-01')
todate = pd.Timestamp('2020-06-01')

# 添加多只股票数据
stock_symbols = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', 
                 '000011', '000012', '000014', '000016',  '000019', '000020', '000021', 
                 '000025', '000026', '000027', '000028', '000029', '000030']  # 示例股票代码
# stock_symbols = ['000001', '000002', '000004', '000005', '000006', '000007', '000008', '000009', '000010', 
#                  '000011', '000012', '000014', '000016', '000017', '000018', '000019', '000020', '000021', 
#                  '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029', '000030', 
#                  '000031', '000032', '000033', '000034', '000035', '000036', '000037', '000038', '000039', 
#                  '000040', '000042', '000043', '000045', '000046', '000048', '000049', '000050', '000055', 
#                  '000056', '000058', '000059', '000060', '000061', '000062', '000063', '000065', '000066', 
#                  '000068', '000069', '000070', '000078', '000088', '000089', '000090', '000096', '000099', 
#                  '000100', '000150', '000151', '000153', '000155', '000156', '000157', '000158', '000159', 
#                  '000166', '000301', '000333', '000338', '000400', '000401', '000402', '000403', '000404', 
#                  '000406', '000407', '000408', '000409', '000410', '000411', '000413', '000415', '000416', 
#                  '000417', '000418', '000419', '000420', '000421', '000422', '000423', '000425', '000426', 
#                  '000428', '300245', '600616']
for symbol in stock_symbols:
    data = CustomCSVData(
        dataname=f'data/{symbol}.csv',
        dtformat=('%Y-%m-%d'),
        fromdate=fromdate,
        todate=todate,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        money=6,
        avg=7,
        high_limit=8,
        low_limit=9,
        pre_close=10,
        paused=11,
        factor=12,
        MA5=13,
        MA10=14,
        RSI=15,
        WilliamsR=16,
        name=symbol
    )
    cerebro.adddata(data)

# 将策略添加到Cerebro
cerebro.addstrategy(MultiStockStrategy)

# 设置初始资金
cerebro.broker.set_cash(100000.0)

# 设置交易手续费
cerebro.broker.setcommission(commission=0.001)

# 运行回测
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.run()
print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

# 调整绘图参数
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=len(stock_symbols), ncols=1, figsize=(15, 5 * len(stock_symbols)))

cerebro.plot()
