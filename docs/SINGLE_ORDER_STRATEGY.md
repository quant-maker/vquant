# 单挂单策略说明

## 概述
Trader现在采用更简单高效的单挂单策略，特别适合定时执行的交易策略。

## 核心思想
每个策略在任何时刻最多只保留**一个挂单**，避免订单堆积和仓位计算复杂性。

## 执行流程

### 每次执行时的步骤

```
第N次执行（例如：每小时执行一次）
├── Step 1: 检查上次的订单
│   ├── 如果已成交 → 清除状态，继续
│   ├── 如果未成交 → 撤销订单，清除状态
│   └── 如果不存在 → 直接继续
│
├── Step 2: 查询当前实际持仓
│   └── 获取账户在该交易对的实际仓位
│
├── Step 3: 计算需要调整的数量
│   ├── 目标仓位 = AI建议 × 最大仓位
│   ├── 需要买卖 = 目标仓位 - 当前仓位 + 初始偏移
│   └── 如果为0 → 无需调整，结束
│
└── Step 4: 下新单
    ├── 生成订单ID（含策略标识）
    ├── 发送限价单到交易所
    └── 保存订单状态到文件
```

## 订单状态管理

### 状态文件
- 位置：`logs/orders/{strategy_id}_order_state.json`
- 内容：
```json
{
    "symbol": "BTCUSDC",
    "order_id": 12345678,
    "side": "BUY",
    "quantity": 0.002
}
```

### 状态转换
```
初始状态（无订单）
    ↓ 下单
订单已保存
    ↓ 下次执行检查
订单已成交 → 清除状态 → 初始状态
订单未成交 → 撤销 → 清除状态 → 初始状态
```

## 示例场景

### 场景1：首次执行
```
时间: 10:00
状态: 无历史订单
当前持仓: 0 BTC
AI建议: 0.75 (看多)
最大仓位: 0.01 BTC

执行:
1. 检查上次订单 → 不存在
2. 当前仓位 = 0
3. 目标仓位 = 0.01 × 0.75 = 0.0075 BTC
4. 需要买入 = 0.0075 - 0 = 0.0075 BTC
5. 下单: BUY 0.0075 BTC @ 43250
6. 保存订单ID: 12345678
```

### 场景2：上次订单未成交
```
时间: 11:00 (1小时后)
状态: 订单12345678未成交
当前持仓: 0 BTC (因为订单未成交)
AI建议: 0.85 (更看多)
最大仓位: 0.01 BTC

执行:
1. 检查上次订单 → 存在，状态=NEW
2. 撤销订单12345678
3. 当前仓位 = 0
4. 目标仓位 = 0.01 × 0.85 = 0.0085 BTC
5. 需要买入 = 0.0085 - 0 = 0.0085 BTC
6. 下单: BUY 0.0085 BTC @ 43500
7. 保存订单ID: 12345679
```

### 场景3：上次订单已成交
```
时间: 12:00
状态: 订单12345679已成交
当前持仓: 0.0085 BTC
AI建议: 0.5 (看多减弱)
最大仓位: 0.01 BTC

执行:
1. 检查上次订单 → 已成交，清除状态
2. 当前仓位 = 0.0085
3. 目标仓位 = 0.01 × 0.5 = 0.005 BTC
4. 需要卖出 = 0.005 - 0.0085 = -0.0035 BTC
5. 下单: SELL 0.0035 BTC @ 43800
6. 保存订单ID: 12345680
```

### 场景4：无需调整
```
时间: 13:00
状态: 订单12345680已成交
当前持仓: 0.005 BTC
AI建议: 0.5 (保持)
最大仓位: 0.01 BTC

执行:
1. 检查上次订单 → 已成交，清除状态
2. 当前仓位 = 0.005
3. 目标仓位 = 0.01 × 0.5 = 0.005 BTC
4. 需要调整 = 0.005 - 0.005 = 0
5. 无需下单 ✓
```

## 优势

### 1. 简单高效
- ❌ 不需要查询大量历史订单
- ❌ 不需要复杂的订单匹配算法
- ✅ 只需检查一个订单状态
- ✅ 文件IO操作极少

### 2. 状态清晰
- 任何时刻只有一个挂单
- 不会出现订单堆积
- 容易追踪和调试

### 3. 容错性好
- 订单状态文件丢失 → 自动按无订单处理
- API查询失败 → 清除状态，下次重试
- 策略重启 → 自动恢复上次订单状态

### 4. 适合定时策略
- 每N分钟执行一次
- 不需要实时监控
- 降低API调用频率

## 代码示例

### 订单状态操作
```python
# 保存订单状态
self._save_order_state(
    symbol='BTCUSDC',
    order_id=12345678,
    side='BUY',
    quantity=0.0075
)

# 读取订单状态
state = self._load_order_state()
# {'symbol': 'BTCUSDC', 'order_id': 12345678, 'side': 'BUY', 'quantity': 0.0075}

# 清除订单状态
self._clear_order_state()
```

### 处理上次订单
```python
# 返回值：需要调整的数量
adjustment = self._handle_previous_order('BTCUSDC')
# adjustment = 0: 订单已成交或不存在
# adjustment < 0: 上次BUY订单未成交，需要补回
# adjustment > 0: 上次SELL订单未成交，需要补回
```

## 日志示例

```log
INFO: ============================================================
INFO: Strategy 'trend_following' trading BTCUSDC...
DEBUG: Target position ratio: 0.75, current price: 43250.5

INFO: Step 1: Checking previous order...
INFO: Checking previous order: 12345678
INFO: Previous order status: NEW
INFO: Canceling previous order: 12345678
INFO: Canceled order: 0.0075 unfilled (filled: 0.0)

INFO: Step 2: Getting current position...
DEBUG: Current position for BTCUSDC: 0.0
INFO: Current filled position: 0.0

INFO: Step 3: Calculating target volume...
INFO: Target position: 0.0075, Current: 0.0, Init offset: 0.0
INFO: Calculated volume to trade: 0.0075

INFO: Step 4: Placing new order...
INFO: Preparing order: BUY 0.0075 BTCUSDC @ 43250.5
INFO: Sending order: {'symbol': 'BTCUSDC', 'side': 'BUY', ...}
INFO: ✓ Order placed successfully: OrderID=12345679
DEBUG: Saved order state: {'symbol': 'BTCUSDC', 'order_id': 12345679, ...}
INFO: ============================================================
```

## 与原方案对比

### 原方案（历史订单查询）
```python
# 查询500个历史订单
orders = self.cli.all_orders(symbol=symbol, limit=500)

# 遍历筛选本策略订单
for order in orders:
    if order['clientOrderId'].startswith(f"{strategy_id}_"):
        if order['status'] == 'FILLED':
            # 累计仓位...
```

**问题：**
- API调用开销大
- 处理时间长
- 受limit限制
- 长期运行可能丢失历史

### 新方案（单订单跟踪）
```python
# 读取上次订单状态（文件IO）
state = self._load_order_state()

# 只查询这一个订单
if state:
    order_info = self.cli.query_order(
        symbol=symbol, 
        orderId=state['order_id']
    )
```

**优势：**
- 只查询一个订单
- 响应速度快
- 无历史限制
- 逻辑简单清晰

## 注意事项

### 1. 适用场景
✅ 定时执行策略（如每小时、每4小时）
✅ 限价单策略
✅ 单交易对管理
❌ 高频交易（秒级）
❌ 市价单策略
❌ 需要保留多个挂单的策略

### 2. 文件管理
- 订单状态文件自动创建在 `logs/orders/` 目录
- 不同策略ID有独立的状态文件
- 建议定期清理已完成的状态文件

### 3. 异常处理
- 订单查询失败 → 清除状态，下次重试
- 撤单失败 → 记录错误，清除状态
- 文件读写失败 → 记录错误，继续执行

### 4. 多策略运行
- 每个策略ID独立管理
- 不同策略可以同时运行
- 通过strategy_id区分

## 配置参数

```bash
# 基本用法
python main.py --symbol BTCUSDC --interval 1h --trade

# 指定策略ID
python main.py --symbol BTCUSDC --interval 1h --trade \
    --strategy-id my_trend_strategy

# 多策略示例
# 策略1：短期（每小时）
python main.py --symbol BTCUSDC --interval 1h --trade \
    --strategy-id short_term --volume 0.01

# 策略2：长期（每4小时）
python main.py --symbol BTCUSDC --interval 4h --trade \
    --strategy-id long_term --volume 0.02
```

## 总结

单挂单策略通过以下方式实现了简单高效的订单管理：

1. **状态持久化**：将订单信息保存到文件
2. **自动清理**：每次执行前清理上次订单
3. **仓位同步**：基于实际持仓计算调整量
4. **单一订单**：永远只保留一个挂单

这种方案特别适合需要定期调整仓位的自动化交易策略。
