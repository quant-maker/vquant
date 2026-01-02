#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
仓位稳定性测试
演示如何避免频繁调仓
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vquant.analysis.kalshi import KalshiTrader
from datetime import datetime, timedelta


def simulate_position_changes():
    """模拟多次策略运行，展示仓位稳定性"""
    
    print("\n" + "="*80)
    print("仓位稳定性测试")
    print("="*80)
    
    trader = KalshiTrader(symbol="BTCUSDC", name="stability_test")
    
    # 模拟场景1：小幅波动
    print("\n【场景1：小幅波动 - 应该保持仓位稳定】")
    print("-" * 80)
    
    scenarios = [
        (0.40, "初始信号: 买入0.40"),
        (0.42, "小幅上升: 买入0.42 (+0.02)"),
        (0.38, "小幅下降: 买入0.38 (-0.04)"),
        (0.41, "小幅回升: 买入0.41 (+0.03)"),
    ]
    
    for target, desc in scenarios:
        print(f"\n{desc}")
        # 模拟计算出的目标仓位
        trader.last_position_time = datetime.now() - timedelta(minutes=30)  # 30分钟前
        stable = trader._apply_position_stability(target)
        print(f"  目标仓位: {target:.3f}")
        print(f"  实际仓位: {stable:.3f}")
        print(f"  说明: {'✓ 保持稳定' if abs(stable - trader.last_position) < 0.01 else '△ 调整'}")
    
    # 重置
    trader.last_position = 0.0
    trader.last_position_time = None
    trader.position_history = []
    
    # 模拟场景2：明显信号变化
    print("\n\n【场景2：明显信号变化 - 应该调整仓位】")
    print("-" * 80)
    
    scenarios2 = [
        (0.40, "初始信号: 买入0.40"),
        (0.60, "强烈信号: 买入0.60 (+0.20)"),  # 变化>0.15，应该调整
        (0.70, "继续增强: 买入0.70 (+0.10)"),
    ]
    
    for target, desc in scenarios2:
        print(f"\n{desc}")
        trader.last_position_time = datetime.now() - timedelta(minutes=70)  # 超过最小持仓时间
        stable = trader._apply_position_stability(target)
        print(f"  目标仓位: {target:.3f}")
        print(f"  实际仓位: {stable:.3f}")
        if abs(target - stable) > 0.05:
            print(f"  说明: △ 平滑调整 (目标 {target:.3f} → 实际 {stable:.3f})")
        else:
            print(f"  说明: ✓ 达到目标")
    
    # 重置
    trader.last_position = 0.0
    trader.last_position_time = None
    trader.position_history = []
    
    # 模拟场景3：持仓时间不足
    print("\n\n【场景3：持仓时间不足 - 需要更强信号才能调整】")
    print("-" * 80)
    
    scenarios3 = [
        (0.40, 0, "初始信号: 买入0.40"),
        (0.55, 20, "20分钟后信号变化: 买入0.55 (+0.15)"),  # 时间不足，但变化不够大
        (0.70, 30, "30分钟后强烈信号: 买入0.70 (+0.30)"),  # 时间不足，但变化很大，应该调整
    ]
    
    for target, minutes, desc in scenarios3:
        print(f"\n{desc}")
        if trader.last_position_time:
            trader.last_position_time = datetime.now() - timedelta(minutes=minutes)
        stable = trader._apply_position_stability(target)
        print(f"  目标仓位: {target:.3f}")
        print(f"  实际仓位: {stable:.3f}")
        time_held = (datetime.now() - trader.last_position_time).total_seconds() / 60 if trader.last_position_time else 0
        print(f"  持仓时间: {time_held:.0f}分钟")
        print(f"  说明: {'✓ 保持' if abs(stable - trader.last_position) < 0.01 else '△ 调整'}")
    
    # 输出历史记录
    print("\n\n【仓位变化历史】")
    print("-" * 80)
    if trader.position_history:
        for i, record in enumerate(trader.position_history):
            time_str = record['time'].strftime('%H:%M:%S')
            pos = record['position']
            target = record.get('target', pos)
            print(f"{i+1}. {time_str} - 实际: {pos:.3f} (目标: {target:.3f})")
    
    print("\n" + "="*80)
    print("总结：")
    print("1. 小幅波动(<0.15)时，保持原仓位，避免频繁交易")
    print("2. 明显变化(>0.15)时，平滑调整仓位，避免剧烈波动")
    print("3. 持仓时间不足(<60分钟)时，需要更强信号(2倍变化)才调整")
    print("="*80 + "\n")


def test_smoothing_effect():
    """测试平滑效果"""
    
    print("\n" + "="*80)
    print("仓位平滑效果演示")
    print("="*80)
    
    print("\n假设目标仓位从 0.0 突然变到 0.8")
    print("不同平滑系数的效果：\n")
    
    smoothing_factors = [0.3, 0.5, 0.8, 1.0]
    target = 0.8
    current = 0.0
    
    print(f"{'平滑系数':<10} {'第1次':<10} {'第2次':<10} {'第3次':<10} {'第4次':<10}")
    print("-" * 60)
    
    for alpha in smoothing_factors:
        positions = [current]
        pos = current
        for _ in range(4):
            pos = pos * (1 - alpha) + target * alpha
            positions.append(pos)
        
        result = f"{alpha:<10.1f} "
        for p in positions[1:]:
            result += f"{p:<10.3f} "
        print(result)
    
    print("\n说明：")
    print("  - 系数越小(0.3)，调整越平滑，需要多次才达到目标")
    print("  - 系数越大(1.0)，调整越快，一次到位")
    print("  - 推荐0.3-0.5，既平滑又不会太慢")
    print("="*80 + "\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    simulate_position_changes()
    test_smoothing_effect()


if __name__ == "__main__":
    main()
