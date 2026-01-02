# Kalshi预测市场交易策略

## 策略概述

这是一个创新的量化交易策略，利用[Kalshi](https://kalshi.com/)预测市场的数据来交易数字货币。Kalshi是一个受监管的预测市场平台，允许用户交易关于未来事件的二元期权。

### 核心思想

**群体智慧**：预测市场汇聚了众多参与者的信息和观点，市场价格反映了集体对未来事件的预期。通过分析加密货币相关的预测市场，我们可以获得独特的市场情绪信号。

### 策略优势

1. **独特数据源**：与传统技术分析不同，预测市场提供前瞻性的情绪指标
2. **信息聚合**：市场价格自动整合了大量参与者的观点和信息
3. **量化情绪**：将主观的市场情绪转化为可量化的交易信号
4. **风险管理**：结合技术指标过滤，降低误判风险

## 策略逻辑

### 1. 数据采集

从Kalshi获取与目标加密货币相关的预测市场数据：
- 价格预测市场（如"比特币价格是否超过$50,000"）
- 监管政策市场
- 行业发展市场
- 宏观经济市场

### 2. 情绪计算

从多个相关市场的订单簿数据计算综合情绪得分：

```python
sentiment_score = average([市场1的yes价格, 市场2的yes价格, ...])
```

- **情绪得分范围**：0-1
  - 接近1：市场普遍看涨
  - 接近0：市场普遍看跌
  - 接近0.5：市场中性

### 3. 置信度评估

基于交易量计算置信度：

```python
confidence = min(total_volume / 10000, 1.0)
```

置信度越高，说明参与者越多，信号越可靠。

### 4. 仓位决策

结合情绪得分、置信度和技术指标，生成仓位建议：

```
最终仓位 = 基础仓位 × 置信度 × 技术过滤系数
```

- **基础仓位**：根据情绪得分计算
  - 情绪 > 0.65 → 做多
  - 情绪 < 0.35 → 做空
  - 0.35 ≤ 情绪 ≤ 0.65 → 空仓

- **技术过滤**：
  - RSI超买/超卖调整
  - 均线趋势确认
  - 黄金交叉/死亡交叉

### 5. 风险控制

- 最低置信度要求
- 最大仓位限制
- 技术指标二次确认

## 使用方法

### 安装依赖

```bash
pip install requests pandas numpy
```

### 基础用法

```bash
# 分析BTC走势（不登录，使用公开数据）
python main.py --symbol BTCUSDC --predictor kalshi --name kalshi_btc

# 使用Kalshi账户登录获取更多数据
python main.py --symbol BTCUSDC --predictor kalshi --name kalshi_btc \
  --kalshi-email your@email.com \
  --kalshi-password yourpassword

# 启用实盘交易
python main.py --symbol BTCUSDC --predictor kalshi --name kalshi_btc \
  --trade --volume 0.01 --threshold 0.3
```

### 配置文件

创建 `config/kalshi_strategy.json`：

```json
{
  "sentiment_threshold_long": 0.65,
  "sentiment_threshold_short": 0.35,
  "confidence_threshold": 0.3,
  "max_position": 1.0,
  "use_technical_filter": true,
  "cache_minutes": 30,
  "rsi_oversold": 30,
  "rsi_overbought": 70,
  "ma_periods": [20, 50, 200]
}
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sentiment_threshold_long` | float | 0.65 | 做多阈值，情绪得分超过此值考虑做多 |
| `sentiment_threshold_short` | float | 0.35 | 做空阈值，情绪得分低于此值考虑做空 |
| `confidence_threshold` | float | 0.3 | 最低置信度要求 |
| `max_position` | float | 1.0 | 最大仓位（1.0=满仓） |
| `use_technical_filter` | bool | true | 是否使用技术指标过滤 |
| `cache_minutes` | int | 30 | Kalshi数据缓存时长（分钟） |
| `rsi_oversold` | int | 30 | RSI超卖阈值 |
| `rsi_overbought` | int | 70 | RSI超买阈值 |
| `ma_periods` | list | [20,50,200] | 移动平均线周期 |

## 策略示例

### 示例1：保守策略

适合风险厌恶型交易者：

```json
{
  "sentiment_threshold_long": 0.70,
  "sentiment_threshold_short": 0.30,
  "confidence_threshold": 0.5,
  "max_position": 0.5,
  "use_technical_filter": true
}
```

特点：
- 更高的情绪阈值（只在极端情绪时交易）
- 更高的置信度要求
- 限制最大仓位50%

### 示例2：激进策略

适合风险偏好型交易者：

```json
{
  "sentiment_threshold_long": 0.60,
  "sentiment_threshold_short": 0.40,
  "confidence_threshold": 0.2,
  "max_position": 1.0,
  "use_technical_filter": false
}
```

特点：
- 更低的情绪阈值（更频繁交易）
- 更低的置信度要求
- 允许满仓交易
- 不使用技术指标过滤

### 示例3：均衡策略

平衡风险和收益：

```json
{
  "sentiment_threshold_long": 0.65,
  "sentiment_threshold_short": 0.35,
  "confidence_threshold": 0.3,
  "max_position": 0.8,
  "use_technical_filter": true
}
```

## 监控指标

运行策略时，重点关注以下指标：

1. **情绪得分**：当前市场情绪（0-1）
2. **置信度**：信号可靠性（0-1）
3. **相关市场数量**：参与计算的市场数量
4. **建议仓位**：最终仓位建议（-1到1）
5. **技术指标**：RSI、均线等

### 日志示例

```
2026-01-02 10:30:00 [INFO] === 更新市场情绪数据: BTC ===
2026-01-02 10:30:05 [INFO]   情绪得分: 0.720
2026-01-02 10:30:05 [INFO]   置信度: 0.450
2026-01-02 10:30:05 [INFO]   市场数量: 8
2026-01-02 10:30:06 [INFO] 做多信号: 情绪=0.720, 仓位=0.88
2026-01-02 10:30:06 [INFO] RSI: 45.32
2026-01-02 10:30:06 [INFO] 最终建议仓位: 0.396
2026-01-02 10:30:06 [INFO] 操作建议: 建议做多
```

## 回测与优化

### 数据收集

建议先收集历史数据用于回测：

```python
from vquant.data.kalshi_fetcher import KalshiFetcher
from datetime import datetime, timedelta

fetcher = KalshiFetcher()

# 收集历史数据
start_date = datetime.now() - timedelta(days=90)
end_date = datetime.now()

history = fetcher.get_market_history(
    market_ticker="YOUR_MARKET_TICKER",
    start_date=start_date,
    end_date=end_date
)
```

### 参数优化

通过网格搜索找到最优参数：

```python
# 参数网格
param_grid = {
    'sentiment_threshold_long': [0.60, 0.65, 0.70],
    'confidence_threshold': [0.2, 0.3, 0.4],
    'max_position': [0.5, 0.8, 1.0]
}

# 对每组参数进行回测
# 选择夏普比率最高的参数组合
```

## 风险提示

⚠️ **重要风险提示**：

1. **数据延迟**：Kalshi市场更新可能滞后于实际价格变动
2. **流动性风险**：某些预测市场交易量较小，信号可能不可靠
3. **市场失效**：预测市场可能受到操纵或存在偏差
4. **技术风险**：API可能不稳定或限流
5. **策略容量**：大额交易可能影响市场价格

**建议**：
- 从小额资金开始测试
- 设置严格的止损
- 分散投资，不要孤注一掷
- 定期审查和调整策略参数
- 监控策略表现，及时止损

## 未来改进方向

1. **多因子模型**：整合更多数据源（社交媒体、链上数据等）
2. **机器学习**：使用ML模型优化信号生成
3. **动态参数**：根据市场波动自动调整阈值
4. **高频策略**：利用短期价格波动
5. **跨市场套利**：同时交易Kalshi和现货市场

## 相关资源

- [Kalshi官网](https://kalshi.com/)
- [Kalshi API文档](https://trading-api.readme.io/reference/introduction)
- [预测市场介绍](https://en.wikipedia.org/wiki/Prediction_market)
- [Binance API文档](https://binance-docs.github.io/apidocs/)

## 技术支持

如有问题，请查看日志文件或提交issue。

## 许可证

本策略仅供学习和研究使用，使用者需自行承担交易风险。

---

**最后更新**: 2026-01-02
**版本**: 1.0.0
