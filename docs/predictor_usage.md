# 量化预测器 (QuantPredictor) 使用说明

## 概述

QuantPredictor 是一个基于技术指标和市场数据的量化预测模型，作为 AI Advisor 的平级替代方案。它不依赖大模型API，而是通过规则化的因子评分系统来预测市场方向。

## 两种分析模式对比

### 1. AI Advisor 模式（默认）
- **特点**：使用大模型（Qwen/GPT-4/Claude等）分析K线图
- **优势**：能识别复杂图表模式，提供详细的市场洞察
- **劣势**：需要API密钥，有调用成本，响应较慢
- **适用场景**：需要深度分析和详细解读的场景

### 2. Quantitative Predictor 模式
- **特点**：基于6个量化因子的评分系统
- **优势**：完全本地计算，无需API，速度快，结果可解释
- **劣势**：仅基于预设规则，无法识别复杂图表形态
- **适用场景**：需要快速决策、高频交易、或没有API的场景

## 使用方法

### 使用 AI Advisor（默认）
```bash
python main.py --symbol DOGEUSDC --interval 1h --service qwen --name my-strategy
```

### 使用 Quantitative Predictor
```bash
python main.py --symbol DOGEUSDC --interval 1h --use-predictor --name my-strategy
```

### 带交易执行的预测
```bash
# Predictor模式 + 交易
python main.py --symbol DOGEUSDC --interval 1h --use-predictor --trade --volume 100 --name my-strategy

# AI Advisor模式 + 交易
python main.py --symbol DOGEUSDC --interval 1h --service qwen --trade --volume 100 --name my-strategy
```

## Predictor 评分因子

QuantPredictor 基于以下6个因子进行综合评分：

### 1. Funding Rate（资金费率）- 权重 25%
反映多空力量对比：
- 正值且高：多头支付空头，市场过热 → 看跌
- 负值且低：空头支付多头，市场超跌 → 看涨
- 中性区间：-0.02% ~ +0.02%

**评分规则**：
- \> 0.15%: -100 (极度看跌)
- \> 0.10%: -80
- \> 0.05%: -50
- \> 0.02%: -20
- -0.02% ~ 0.02%: 0 (中性)
- < -0.02%: +20
- < -0.05%: +50
- < -0.10%: +80
- < -0.15%: +100 (极度看涨)

### 2. MA7 拐点 - 权重 25%
检测MA7在最近4根K线的转折：
- **向上拐点** (upward): +80
- **向下拐点** (downward): -80
- **延续趋势** (continuing): 0

### 3. MA 趋势 - 权重 15%
均线排列状态：
- **多头排列** (MA7 > MA25 > MA99): +100
- **空头排列** (MA7 < MA25 < MA99): -100
- **金叉区域** (MA7 > MA25): +50
- **死叉区域** (MA7 < MA25): -50

### 4. RSI - 权重 15%
超买超卖指标：
- RSI > 80: -100 (严重超买)
- RSI > 70: -80 (超买)
- RSI > 60: -40
- RSI 40-60: 0 (中性)
- RSI < 30: +80 (超卖)
- RSI < 20: +100 (严重超卖)

### 5. MACD - 权重 10%
动量指标：
- MACD > Signal 且都为正: +100 (强多头)
- MACD > Signal 且都为负: +50 (弱多头)
- MACD < Signal 且都为正: -50 (弱空头)
- MACD < Signal 且都为负: -100 (强空头)

### 6. Volume - 权重 10%
成交量确认：
- 放量 > 100%: +80
- 放量 > 50%: +50
- 正常: +20 ~ -20
- 缩量 < -50%: -50

## 输出格式

### Predictor 输出
```json
{
  "symbol": "DOGEUSDC",
  "position": -0.39,
  "confidence": "medium",
  "current_price": 0.12794,
  "score": -38.46,
  "factors": {
    "funding_rate": -20.0,
    "ma7_inflection": -80.0,
    "ma_trend": -50.0,
    "rsi": 40.0,
    "macd": -50.0,
    "volume": -20.0
  },
  "reasoning": "综合评分偏空，建议小仓位做空。资金费率 0.08% 偏高，多头占优但需谨慎；MA7 出现向下拐点，短期趋势转空；MA7 下穿 MA25，短期看空；MACD 死叉向下，动量减弱。",
  "analysis_type": "predictor"
}
```

### 字段说明
- **position**: 仓位建议 (-1.0 到 1.0)
  - 1.0: 满仓做多
  - 0.5: 半仓做多
  - 0.0: 空仓观望
  - -0.5: 半仓做空
  - -1.0: 满仓做空

- **confidence**: 置信度
  - "high": 因子方向一致性高（>75%）且评分绝对值大（>50）
  - "medium": 因子方向部分一致（>50%）或评分中等（>30）
  - "low": 因子方向分歧大或评分接近0

- **score**: 综合评分 (-100 到 100)
  - 正值：看涨信号
  - 负值：看跌信号
  - 绝对值越大，信号越强

- **factors**: 各因子得分明细

## 配置建议

### 保守策略
```bash
# 只在高置信度时交易，小仓位
python main.py --symbol BTCUSDC --use-predictor --trade --volume 50 --name conservative
# 在代码中添加置信度过滤：
if result['confidence'] == 'high' and abs(result['position']) > 0.3:
    trader.trade(result, args)
```

### 激进策略
```bash
# 中等置信度即交易，大仓位
python main.py --symbol DOGEUSDC --use-predictor --trade --volume 200 --name aggressive
```

### 混合策略
在代码中同时使用两种模式，取平均：
```python
# 1. 运行 Predictor
predictor_result = predictor.predict(stats)
# 2. 运行 Advisor
advisor_result = advisor.analyze(...)
# 3. 混合决策
final_position = (predictor_result['position'] + advisor_result['position']) / 2
```

## 注意事项

1. **Predictor 不能完全替代 AI Advisor**
   - Predictor 基于规则，适合标准行情
   - AI Advisor 能识别复杂形态，适合特殊行情

2. **因子权重可以调整**
   - 在 `vquant/analysis/predictor.py` 中修改 `self.weights`
   - 根据不同市场特性调整各因子权重

3. **建议先回测**
   - 使用历史数据验证策略效果
   - 调整因子评分阈值和权重

4. **风险控制**
   - 设置止损止盈
   - 控制仓位规模
   - 避免在低置信度时交易

## 示例场景

### 场景1：无API密钥时使用
```bash
# 没有配置AI API密钥，使用Predictor
python main.py --symbol BTCUSDC --use-predictor --name my-bot
```

### 场景2：快速决策
```bash
# 每分钟运行一次，使用Predictor保证速度
*/1 * * * * cd /path/to/graph-quant && python main.py --symbol ETHUSDC --use-predictor --quiet
```

### 场景3：对比两种模式
```bash
# 分别运行两种模式，对比结果
python main.py --symbol DOGEUSDC --name doge-advisor
python main.py --symbol DOGEUSDC --use-predictor --name doge-predictor
# 查看结果差异
diff charts/doge-advisor_*.json charts/doge-predictor_*.json
```

## 未来改进方向

1. 机器学习优化因子权重
2. 增加更多技术指标（Bollinger Bands、Fibonacci等）
3. 支持多时间周期分析
4. 增加市场情绪因子（Twitter、新闻等）
5. 回测系统集成
