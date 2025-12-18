# 策略仓位区分功能更新

## 概述
为Trader类添加了策略标识符（strategy_id）功能，现在可以区分不同策略的仓位，避免多个策略之间的仓位冲突。

## 主要变更

### 1. Trader类更新 (`vquant/executor/trader.py`)

#### 新增参数
- 添加了 `strategy_id` 参数到构造函数
  ```python
  def __init__(self, init_pos, account: str = 'li', strategy_id: str = 'default')
  ```

#### 新增方法
- `get_strategy_position(symbol: str) -> float`: 获取当前策略的仓位
  - 通过查询历史订单，筛选包含本策略ID的订单
  - 计算本策略的净仓位（买入为正，卖出为负）
  - 如果无法获取订单历史，回退到查询总仓位

#### 订单标识
- 在下单时，自动添加 `newClientOrderId` 字段
- 格式：`{strategy_id}_{timestamp_ms}`
- 例如：`default_1734518400000`

#### 仓位计算改进
- 现在只计算本策略的仓位，而非账户总仓位
- 同时记录账户总仓位供参考
- 日志中清晰显示策略仓位和总仓位的区别

### 2. main.py更新

#### 新增命令行参数
```bash
--strategy-id STRATEGY_ID
    策略标识符，用于区分不同策略的仓位
    默认值: 'default'
```

#### Trader初始化更新
```python
trader = Trader(args.init_pos, account=args.account, strategy_id=args.strategy_id)
```

## 使用示例

### 基本用法（使用默认策略ID）
```bash
python main.py --symbol BTCUSDC --interval 1h --trade
```

### 指定策略ID
```bash
python main.py --symbol BTCUSDC --interval 1h --trade --strategy-id strategy_a
```

### 多策略场景
```bash
# 策略A - 短期交易
python main.py --symbol BTCUSDC --interval 1h --trade --strategy-id short_term

# 策略B - 长期交易（同一账户，不同策略ID）
python main.py --symbol BTCUSDC --interval 4h --trade --strategy-id long_term
```

## 工作原理

1. **订单标记**: 每个订单都会在 `clientOrderId` 中包含策略ID前缀
   - 例如：`strategy_a_1734518400000`

2. **仓位计算**: 
   - 获取所有历史订单
   - 筛选 `clientOrderId` 以本策略ID开头的订单
   - 只统计状态为 `FILLED` 的订单
   - 计算净仓位：BUY订单加数量，SELL订单减数量

3. **容错机制**:
   - 如果无法获取订单历史（如API限制、权限问题）
   - 自动回退到查询账户总仓位
   - 记录警告日志

## 日志示例

```
INFO: Checking position for BTCUSDC...
DEBUG: Target position: 0.75, current price: 43250.5
DEBUG: Strategy 'strategy_a' position for BTCUSDC: 0.002
INFO: Current strategy 'strategy_a' position: 0.002
INFO: Total account position: 0.005
INFO: Preparing order: BUY 0.001 BTCUSDC @ 43250.5
```

## 注意事项

1. **历史订单限制**: `get_strategy_position` 默认查询最近500个订单。如果策略运行时间很长，可能需要调整 `limit` 参数。

2. **订单状态**: 只统计 `FILLED` 状态的订单，未成交或部分成交的订单不计入仓位。

3. **策略ID命名**: 建议使用有意义的策略ID，如：
   - `trend_following`
   - `mean_reversion`
   - `scalping_1m`

4. **兼容性**: 向后兼容，不指定 `strategy_id` 时使用默认值 `'default'`。

## 技术细节

### 仓位计算算法
```python
position = 0.0
for order in orders:
    if order['clientOrderId'].startswith(f"{strategy_id}_") and order['status'] == 'FILLED':
        qty = float(order['executedQty'])
        if order['side'] == 'BUY':
            position += qty
        else:
            position -= qty
```

### 订单ID生成
```python
import time
client_order_id = f"{self.strategy_id}_{int(time.time() * 1000)}"
```

## 未来改进建议

1. **持久化策略仓位**: 将策略仓位缓存到本地数据库，减少API调用
2. **多交易对支持**: 同时管理多个交易对的策略仓位
3. **仓位对账**: 定期对账策略仓位与实际订单，确保一致性
4. **策略组**: 支持策略分组，一组策略共享仓位限制
