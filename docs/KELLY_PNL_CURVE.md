# Kelly Strategy - PnL Curve Visualization

## 概述

Kelly策略现在支持在测试集上生成PnL曲线，用于评估训练后的机器学习模型的性能。

## 功能特性

### 1. PnL曲线可视化
- **三个子图展示**：
  - 累积PnL对比（等额下注 vs Kelly加权）
  - ML模型胜率预测曲线
  - Kelly准则仓位大小

### 2. 统计指标
- 总交易次数
- 胜率
- 最终PnL（等额和Kelly加权）
- 平均盈利/亏损
- 盈亏比

### 3. 模型评估
- 测试集性能
- Kelly准则优化效果
- 风险收益分析

## 使用方法

### 方式一：使用独立回测工具

```bash
# 基本用法
python tools/kelly_backtest.py --name test --symbol BTCUSDC

# 自定义参数
python tools/kelly_backtest.py \
    --name test \
    --symbol BTCUSDC \
    --interval 1h \
    --days 30 \
    --ma-periods 7 25 99 \
    --verbose

# 只显示统计，不保存图表
python tools/kelly_backtest.py --name test --symbol BTCUSDC --no-plot
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name` | 策略名称（必需，需匹配已训练模型） | - |
| `--symbol` | 交易对 | BTCUSDC |
| `--interval` | K线周期 | 1h |
| `--days` | 测试数据天数 | 30 |
| `--ma-periods` | 均线周期 | 7 25 99 |
| `--no-plot` | 不保存图表，仅显示统计 | False |
| `--verbose` | 详细日志 | False |

### 方式二：在代码中调用

```python
from vquant.analysis.kelly import KellyTrader

# 初始化Kelly trader（会加载训练好的模型）
trader = KellyTrader(symbol="BTCUSDC", name="test", config_dir="config")

# 准备测试数据
test_data = [
    {
        'features': [feature_array],  # 特征数组
        'actual_pnl': 0.5,            # 实际PnL百分比
        'timestamp': datetime.now()    # 时间戳
    },
    # ... more samples
]

# 生成PnL曲线
plot_path = trader.plot_test_pnl_curve(test_data)
print(f"PnL curve saved to: {plot_path}")
```

## 特征说明

模型使用以下特征进行预测：

1. **价格特征**：当前价格
2. **动量特征**：5周期和20周期动量
3. **波动率**：收益率标准差
4. **成交量比率**：近期成交量/平均成交量
5. **资金费率**：永续合约资金费率
6. **RSI**：相对强弱指标
7. **MA偏离度**：价格相对MA7的偏离
8. **MACD**：MACD指标值
9. **ATR百分比**：平均真实波幅百分比

## 输出示例

### 图表输出
- 保存位置：`charts/kelly_test_pnl_{name}_{timestamp}.png`
- 分辨率：1200x1000像素
- 格式：PNG

### 日志输出
```
2025-12-26 10:30:00 INFO [kelly_backtest.py:180]: Fetching 30 days of data...
2025-12-26 10:30:05 INFO [kelly_backtest.py:195]: Fetched 720 bars
2025-12-26 10:30:10 INFO [kelly_backtest.py:250]: Prepared 670 test samples
2025-12-26 10:30:15 INFO [kelly.py:1050]: Test PnL curve saved to: charts/kelly_test_pnl_test_20251226_103015.png
2025-12-26 10:30:15 INFO [kelly.py:1051]: Test performance: Equal PnL=12.5%, Kelly PnL=18.3%
```

## Kelly准则说明

### Kelly公式
```
f* = (bp - q) / b
```

其中：
- `f*` = 最优下注比例
- `b` = 盈亏比（平均盈利/平均亏损）
- `p` = 胜率
- `q` = 1 - p（败率）

### Kelly分数设置
- `1.0` = Full Kelly（最大增长，高波动）
- `0.5` = Half Kelly（平衡）
- `0.25` = Quarter Kelly（保守，推荐）
- `0.1` = 1/10 Kelly（非常安全）

配置文件中通过 `kelly_fraction` 参数设置。

## 注意事项

1. **模型训练要求**：
   - 必须先训练模型（至少100个样本）
   - 模型文件：`data/kelly_model_{name}.pkl`
   - 缩放器文件：`data/kelly_scaler_{name}.pkl`

2. **数据要求**：
   - 至少需要100根K线用于计算指标
   - 推荐使用30天以上的测试数据

3. **性能考虑**：
   - 大量数据可能需要较长处理时间
   - 建议分批测试或使用更大的时间周期

## 故障排除

### 问题：找不到训练好的模型
```
ERROR: No trained model found
```
**解决**：确保已运行足够多的交易来训练模型（至少100笔）

### 问题：特征维度不匹配
```
ERROR: Feature dimension mismatch
```
**解决**：确保测试数据的特征提取方式与训练时一致

### 问题：数据不足
```
WARNING: Not enough data for indicators
```
**解决**：增加 `--days` 参数值或使用更大的时间周期

## 相关文件

- `vquant/analysis/kelly.py` - Kelly策略核心实现
- `tools/kelly_backtest.py` - 回测工具
- `config/kelly_{name}.json` - 配置文件
- `data/kelly_model_{name}.pkl` - 训练好的模型
- `data/kelly_state_{name}.json` - 策略状态

## 示例输出

生成的图表包含三个子图：

1. **累积PnL曲线**
   - 蓝色线：等额下注的累积PnL
   - 绿色线：Kelly加权的累积PnL
   - 展示Kelly准则的优化效果

2. **胜率预测曲线**
   - 紫色线：ML模型预测的胜率
   - 红色虚线：50%基准线
   - 展示模型的预测能力

3. **仓位大小柱状图**
   - 橙色柱：Kelly准则计算的仓位大小
   - 展示动态仓位管理策略

图表底部显示关键统计指标。
