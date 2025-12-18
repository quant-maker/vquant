# 策略仓位独立管理

## 问题
之前使用 `self.cli.position(symbol=symbol)` 获取的是账户**所有策略**的总仓位，无法区分各个策略独立的仓位。

## 解决方案
在JSON状态文件中维护每个策略自己的仓位，不依赖交易所API返回的总仓位。

## 状态文件格式

### 有挂单时
```json
{
  "symbol": "BTCUSDC",
  "order_id": 12345678,
  "side": "BUY",
  "quantity": 0.002,
  "current_position": 0.0075
}
```

### 无挂单时（仅记录仓位）
```json
{
  "symbol": "BTCUSDC",
  "current_position": 0.0075
}
```

## 仓位更新逻辑

### 1. 初次使用
```python
from vquant.executor.position import PositionManager

# 初始化策略，设置初始仓位（通常为0）
position_mgr = PositionManager(client, 'my_strategy')
position_mgr.set_initial_position('BTCUSDC', 0.0)
```

### 2. 下单时
```python
# 当前仓位: 0
# 目标仓位: 0.002
# 下单: BUY 0.002

# 保存时记录期望的仓位
position_mgr.save_new_order(
    symbol='BTCUSDC',
    order_id=12345678,
    side='BUY',
    quantity=0.002,
    expected_position=0.002  # 期望成交后的仓位
)
```

### 3. 下次执行检查订单状态

#### 场景A：订单完全成交
```python
# 订单状态: FILLED
# 操作: 保留 current_position=0.002，清除订单信息

# 文件变为:
{
  "symbol": "BTCUSDC",
  "current_position": 0.002
}
```

#### 场景B：订单完全未成交
```python
# 订单状态: NEW
# 实际成交: 0
# 操作: 撤销订单，仓位不变

# 文件恢复为:
{
  "symbol": "BTCUSDC",
  "current_position": 0.0  # 保持原来的仓位
}
```

#### 场景C：订单部分成交
```python
# 订单: BUY 0.002
# 实际成交: 0.001
# 期望仓位: 0.002
# 操作: 撤销剩余订单，调整仓位

# 计算实际仓位:
# expected_pos = 0.002
# orig_qty = 0.002
# filled_qty = 0.001
# actual_pos = expected_pos - orig_qty + filled_qty
#            = 0.002 - 0.002 + 0.001
#            = 0.001

# 文件变为:
{
  "symbol": "BTCUSDC",
  "current_position": 0.001
}
```

## 完整流程示例

### 首次启动策略
```python
# 1. 初始化
trader = Trader(init_pos=0, account='li', strategy_id='trend_a')

# 2. 设置初始仓位（如果有现有仓位需要接管）
trader.position_mgr.set_initial_position('BTCUSDC', 0.0)
```

### 第1次执行（10:00）
```
当前仓位: 0
AI建议: 0.8 (看多)
目标仓位: 0.01 × 0.8 = 0.008

操作:
1. 检查上次订单 → 无
2. 当前仓位 = 0
3. 需要买入 = 0.008 - 0 = 0.008
4. 下单: BUY 0.008 @ 43250
5. 保存状态:
   {
     "symbol": "BTCUSDC",
     "order_id": 100001,
     "side": "BUY",
     "quantity": 0.008,
     "current_position": 0.008
   }
```

### 第2次执行（11:00）- 订单已成交
```
状态文件:
{
  "order_id": 100001,
  "side": "BUY",
  "quantity": 0.008,
  "current_position": 0.008
}

AI建议: 0.9 (更看多)
目标仓位: 0.01 × 0.9 = 0.009

操作:
1. 检查上次订单 → 已成交
   清除订单信息，保留仓位:
   {
     "symbol": "BTCUSDC",
     "current_position": 0.008
   }
2. 当前仓位 = 0.008
3. 需要买入 = 0.009 - 0.008 = 0.001
4. 下单: BUY 0.001 @ 43500
5. 保存状态:
   {
     "order_id": 100002,
     "side": "BUY",
     "quantity": 0.001,
     "current_position": 0.009
   }
```

### 第3次执行（12:00）- 订单未成交
```
状态文件:
{
  "order_id": 100002,
  "side": "BUY",
  "quantity": 0.001,
  "current_position": 0.009
}

AI建议: 0.5 (看多减弱)
目标仓位: 0.01 × 0.5 = 0.005

操作:
1. 检查上次订单 → 未成交
   撤销订单，恢复原仓位:
   {
     "symbol": "BTCUSDC",
     "current_position": 0.008  # 因为订单未成交
   }
2. 当前仓位 = 0.008
3. 需要卖出 = 0.005 - 0.008 = -0.003
4. 下单: SELL 0.003 @ 43400
5. 保存状态:
   {
     "order_id": 100003,
     "side": "SELL",
     "quantity": 0.003,
     "current_position": 0.005
   }
```

### 第4次执行（13:00）- 订单部分成交
```
状态文件:
{
  "order_id": 100003,
  "side": "SELL",
  "quantity": 0.003,
  "current_position": 0.005
}

订单状态: PARTIALLY_FILLED (成交 0.002)

AI建议: 0.6
目标仓位: 0.006

操作:
1. 检查上次订单 → 部分成交
   - 原订单: SELL 0.003
   - 实际成交: 0.002
   - 期望仓位: 0.005
   - 实际仓位 = 0.005 + 0.003 - 0.002 = 0.006
   撤销剩余订单，调整仓位:
   {
     "symbol": "BTCUSDC",
     "current_position": 0.006
   }
2. 当前仓位 = 0.006
3. 需要调整 = 0.006 - 0.006 = 0
4. 无需下单 ✓
```

## 多策略独立管理

### 策略A
```bash
python main.py --symbol BTCUSDC --interval 1h --trade \
    --strategy-id strategy_a --volume 0.01
```

**状态文件**: `logs/orders/strategy_a_order_state.json`
```json
{
  "symbol": "BTCUSDC",
  "current_position": 0.008
}
```

### 策略B（同一账户）
```bash
python main.py --symbol BTCUSDC --interval 4h --trade \
    --strategy-id strategy_b --volume 0.02
```

**状态文件**: `logs/orders/strategy_b_order_state.json`
```json
{
  "symbol": "BTCUSDC",
  "current_position": 0.015
}
```

### 账户总仓位
```
strategy_a: 0.008 BTC
strategy_b: 0.015 BTC
账户总计: 0.023 BTC ← self.cli.position() 返回的值
```

## 代码变化

### PositionManager

**Before:**
```python
def get_current_position(self, symbol: str) -> float:
    # 错误：获取的是所有策略的总仓位
    posinfo = self.cli.position(symbol=symbol)
    return float(posinfo[0]['positionAmt']) if posinfo else 0
```

**After:**
```python
def get_current_position(self, symbol: str) -> float:
    # 正确：从本策略的状态文件读取
    state = self._load_order_state()
    if state and state.get('symbol') == symbol:
        return float(state.get('current_position', 0))
    return 0
```

### 状态保存

**Before:**
```python
def save_new_order(self, symbol, order_id, side, quantity):
    # 只保存订单信息，不保存仓位
    state = {
        'symbol': symbol,
        'order_id': order_id,
        'side': side,
        'quantity': quantity
    }
```

**After:**
```python
def save_new_order(self, symbol, order_id, side, quantity, expected_position):
    # 保存订单信息和期望的仓位
    state = {
        'symbol': symbol,
        'order_id': order_id,
        'side': side,
        'quantity': quantity,
        'current_position': expected_position  # 新增
    }
```

## 优势

### 1. 策略独立
- 每个策略维护自己的仓位
- 不受其他策略影响
- 可以在同一账户运行多个策略

### 2. 精确控制
- 知道订单成交后的确切仓位
- 可以处理部分成交的情况
- 不依赖API延迟和更新

### 3. 容错性强
- 即使API临时失败，仓位信息也不会丢失
- 状态文件持久化保存
- 重启后可以继续运行

### 4. 调试方便
```bash
# 查看策略当前仓位
cat logs/orders/strategy_a_order_state.json

# 输出:
{
  "symbol": "BTCUSDC",
  "current_position": 0.008
}
```

## 注意事项

### 1. 初始化
首次使用时必须设置初始仓位：
```python
# 如果策略从0开始
position_mgr.set_initial_position('BTCUSDC', 0.0)

# 如果接管已有仓位
position_mgr.set_initial_position('BTCUSDC', 0.005)
```

### 2. 状态文件同步
- 不要手动修改状态文件
- 不要同时运行同一strategy_id的多个实例
- 定期备份 `logs/orders/` 目录

### 3. 对账
建议定期对账策略仓位与实际仓位：
```python
# 策略仓位
strategy_pos = position_mgr.get_current_position('BTCUSDC')

# 实际仓位（所有策略总和）
actual_pos = client.position(symbol='BTCUSDC')[0]['positionAmt']

print(f"Strategy: {strategy_pos}, Actual: {actual_pos}")
```

### 4. 迁移现有策略
如果已经在运行，需要：
1. 停止策略
2. 查询当前实际仓位
3. 设置初始仓位
4. 重新启动

```bash
# 1. 停止策略
# 2. 查询仓位（假设为 0.008）
# 3. 设置初始仓位
python -c "
from vquant.executor.position import PositionManager
from binance.fut.usdm import USDM
from binance.auth.utils import load_api_keys

api_key, private_key = load_api_keys('li')
client = USDM(api_key=api_key, private_key=private_key)
mgr = PositionManager(client, 'my_strategy')
mgr.set_initial_position('BTCUSDC', 0.008)
"
# 4. 重新启动策略
```

## 总结

通过在JSON文件中维护策略仓位：

✅ **独立管理**: 每个策略有自己的仓位记录  
✅ **精确追踪**: 准确知道每次交易后的仓位变化  
✅ **部分成交**: 正确处理部分成交的情况  
✅ **多策略支持**: 可在同一账户运行多个策略  
✅ **持久化**: 重启后不丢失仓位信息  

这是实现多策略管理的关键基础设施！
