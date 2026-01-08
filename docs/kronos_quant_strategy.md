# Kronos Quant Strategy

## 概述

`kronos_quant` 是基于 `kronos` 策略的改进版本，主要解决了原始策略中仓位频繁细微变化的问题。

## 问题背景

原始 `kronos` 策略直接使用网页返回的置信度作为仓位信号，导致：
1. **仓位频繁变化**：置信度的细微波动会导致仓位不断调整
2. **永远无法满仓**：置信度很难达到 1.0，导致仓位永远无法达到满仓状态
3. **交易成本增加**：频繁的小幅调仓会产生大量交易成本

## 解决方案

### 1. 非线性量化映射

使用非线性函数将连续的置信度映射到离散的仓位等级：

```
置信度 -> 非线性调整 -> 离散仓位等级
```

**默认仓位等级**: `[0.0, 0.25, 0.5, 0.75, 1.0]`

### 2. 置信度敏感区域

在不同的置信度区间使用不同的策略：

- **低置信区 (< 0.6)**: 降低仓位，使用保守映射减少风险
- **中置信区 (0.6-0.8)**: 细粒度控制，平衡响应性和稳定性
- **高置信区 (> 0.8)**: 提升仓位，使用激进映射抓住机会

**示例**:
```
置信度 0.50 (低) -> 50% 仓位 (保守)
置信度 0.70 (中) -> 75% 仓位 (平衡)
置信度 0.85 (高) -> 100% 仓位 (激进满仓)
```

### 3. 变化阈值

只有当仓位变化超过一定阈值时才执行调仓：

- **默认阈值**: 10% 相对变化
- **动态调整**: 当置信度发生大幅变化 (>15%) 时，降低阈值到 5%

### 4. 三种映射曲线

可配置的非线性映射方式：

1. **sigmoid** (默认): S型曲线，平滑过渡
2. **quadratic**: 二次函数，中等非线性
3. **cubic**: 三次函数，强烈非线性

## 配置示例

`config/kronos_quant_strategy.json`:

```json
{
  "quantization": {
    "position_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
    "min_change_threshold": 0.1,
    "sensitivity_curve": "sigmoid",
    "sensitivity_power": 2.0,
    "high_confidence_zone": [0.6, 0.8],
    "high_confidence_multiplier": 1.5
  }
}
```

## 使用方法

```bash
# 使用 kronos_quant 策略
python main.py --predictor kronos_quant \
               --kronos-config config/kronos_quant_strategy.json \
               --name my_kronos_quant

# 开启实盘交易
python main.py --predictor kronos_quant \
               --kronos-config config/kronos_quant_strategy.json \
               --name my_kronos_quant \
               --trade
```

## 效果对比

| 场景 | Kronos (原始) | Kronos Quant (改进) |
|------|--------------|-------------------|
| 置信度 0.50 -> 0.55 | 仓位 50% -> 55% (调仓) | 仓位 50% -> 50% (不变) |
| 置信度 0.60 -> 0.65 | 仓位 60% -> 65% (调仓) | 仓位 75% -> 75% (不变) |
| 置信度 0.50 -> 0.70 | 仓位 50% -> 70% (调仓) | 仓位 50% -> 75% (调仓) |
| 置信度 0.85 | 仓位 85% | 仓位 100% (满仓) |
| 置信度 0.90 | 仓位 90% | 仓位 100% (满仓) |

## 优势

1. ✅ **减少交易频率**: 只在显著变化时调仓
2. ✅ **降低交易成本**: 避免频繁的小额交易
3. ✅ **可达满仓状态**: 高置信度可映射到 100% 仓位
4. ✅ **保持响应性**: 在关键区域仍然敏感
5. ✅ **可配置灵活**: 支持多种映射曲线和参数

## 关键参数说明

- `position_levels`: 离散仓位等级，可自定义
- `min_change_threshold`: 最小变化阈值 (相对变化百分比)
- `sensitivity_curve`: 映射曲线类型 (sigmoid/quadratic/cubic)
- `high_confidence_zone`: 高敏感度区间
- `high_confidence_multiplier`: 高置信区的敏感度倍数

## 注意事项

1. 配置文件必须包含 `quantization` 部分，否则使用默认配置
2. 如果使用 `kronos_strategy.json`，将自动使用默认量化配置
3. 建议先在模拟环境测试参数，再应用到实盘
