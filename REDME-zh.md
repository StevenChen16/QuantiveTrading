# 股票预测与量化交易项目

本项目结合了两个相关的计划：基于 CNN 的股票预测模型和一系列量化交易策略及工具。它旨在提供一套全面的股票分析、预测和交易解决方案。

## 项目概览

该项目包含以下主要组成部分：

1. 基于 CNN 的股票预测
2. 多种交易策略（LSTM、MACD、布林带、SVM）
3. 指数和因子计算器
4. 数据预处理和分析工具

## 主要文件

### CNN 股票预测
- `cnn-big.ipynb`: 主要的 CNN 模型训练和评估代码
- `resnet.ipynb`: ResNet 架构实验
- `autoencoder.ipynb`: 自编码器实验
- `bt-multi-model.py`: 多模型回测代码

### 量化交易策略
- LSTM 模型实现
- MACD（移动平均收敛散度）策略
- 布林带实现
- SVM（支持向量机）预测模型

### 工具
- `indexCalculator`: 计算各种金融指数和因子

## 依赖要求

项目依赖包括：

- pandas
- numpy 
- scikit-learn
- tqdm
- tensorflow
- matplotlib
- yfinance

安装依赖：
```
pip install -r requirements.txt
```

## 使用方法

1. CNN 股票预测：
   - 运行 Jupyter notebooks 来训练和评估模型。
   - 使用 `bt-multi-model.py` 进行回测。

2. 量化交易策略：
   - 每个策略都在其自己的脚本或 notebook 中实现。
   - 使用 yfinance 库从 Yahoo Finance 下载数据。

3. 指数计算器：
   - 使用此工具计算重要的金融因子，如夏普比率、索提诺比率、贝塔系数和阿尔法系数，可用于单个股票或投资组合。

## 模型架构

该项目使用多种模型架构，包括：

- 卷积神经网络 (CNN)
- 长短期记忆网络 (LSTM)
- ResNet
- 自编码器
- 支持向量机 (SVM)

## 数据来源

- Yahoo Finance（通过 yfinance 库）
- 更多中国 A 股数据，请参考：
  - [Kaggle 数据集](https://www.kaggle.com/datasets/stevenchen116/stochchina)
  - [Hugging Face 数据集](https://huggingface.co/datasets/StevenChen16/Stock-China-daily)

## 结果

模型性能和回测结果可以在相应的 notebooks 和脚本中找到。

## 未来工作

- 尝试更多特征工程
- 优化模型架构
- 实现额外的回测策略
- 整合更多数据源

## 贡献

欢迎提出问题、改进建议和拉取请求！

## 联系方式

如需询问关于每秒数据或其他问题，请联系：[i@stevenchen.site](mailto:i@stevenchen.site)

## 许可证

MIT 许可证

版权所有 (c) [2023-2024] [Steven Chen]

特此免费授予任何获得本软件副本和相关文档文件（"软件"）的人不受限制地处理本软件的权利，包括不受限制地使用、复制、修改、合并、发布、分发、再许可和/或出售本软件副本的权利，以及允许向其提供本软件的人这样做，但须符合以下条件：

上述版权声明和本许可声明应包含在本软件的所有副本或大部分内容中。

本软件按"原样"提供，不附带任何形式的明示或暗示保证，包括但不限于对适销性、特定用途适用性和非侵权性的保证。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，无论是在合同诉讼、侵权行为还是其他方面，起因于、源于或与本软件有关，或与本软件的使用或其他交易有关。