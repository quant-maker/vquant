#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Fear & Greed Index趋势分析演示
展示情绪变化如何影响交易决策
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vquant.data.fear_greed_fetcher import FearGreedFetcher


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    fetcher = FearGreedFetcher()
    
    print("\n" + "="*80)
    print("Fear & Greed Index 趋势分析")
    print("="*80)
    
    # 获取当前状态
    current = fetcher.get_current_index()
    print(f"\n【当前状态】")
    print(f"指数值: {current['value']}/100")
    print(f"分类: {current['classification']}")
    print(f"更新时间: {current['datetime']}")
    
    # 获取历史数据
    df = fetcher.get_historical(limit=14)
    
    if df.empty:
        print("\n无法获取历史数据")
        return
    
    print(f"\n【最近14天历史】")
    print("-" * 80)
    print(f"{'日期':<12} {'指数':<8} {'分类':<20} {'日变化':<8}")
    print("-" * 80)
    
    for i in range(min(14, len(df))):
        date = df.index[i].strftime('%Y-%m-%d')
        value = df['value'].iloc[i]
        classification = df['classification'].iloc[i]
        
        # 计算日变化
        if i > 0:
            change = value - df['value'].iloc[i-1]
            change_str = f"{change:+.0f}"
        else:
            change_str = "-"
        
        print(f"{date:<12} {value:<8} {classification:<20} {change_str:<8}")
    
    # 趋势分析
    print(f"\n【趋势分析】")
    print("-" * 80)
    
    # 7天趋势
    trend_7d = fetcher.get_trend(days=7)
    print(f"\n7天趋势:")
    print(f"  当前值: {trend_7d['current']}")
    print(f"  7天前: {trend_7d['current'] - trend_7d['change']:.0f}")
    print(f"  变化: {trend_7d['change']:+.0f}")
    print(f"  平均值: {trend_7d['avg_value']:.1f}")
    print(f"  趋势: {trend_7d['trend']}")
    
    if trend_7d['trend'] == 'increasing':
        print(f"  解读: 市场从恐慌向贪婪转变")
    elif trend_7d['trend'] == 'decreasing':
        print(f"  解读: 市场从贪婪向恐慌转变")
    else:
        print(f"  解读: 市场情绪相对稳定")
    
    # 3天趋势
    trend_3d = fetcher.get_trend(days=3)
    print(f"\n3天短期趋势:")
    print(f"  当前值: {trend_3d['current']}")
    print(f"  3天前: {trend_3d['current'] - trend_3d['change']:.0f}")
    print(f"  变化: {trend_3d['change']:+.0f}")
    print(f"  趋势: {trend_3d['trend']}")
    
    # 交易信号解读
    print(f"\n【交易信号解读】")
    print("-" * 80)
    
    signal = fetcher.interpret_signal(current['value'])
    print(f"\n基础信号 (不考虑趋势):")
    print(f"  信号: {signal['signal']}")
    print(f"  建议仓位: {signal['position']:.2f}")
    print(f"  说明: {signal['description']}")
    
    # 结合趋势的解读
    print(f"\n结合趋势的增强分析:")
    
    if current['value'] <= 30:
        if trend_7d['change'] > 0:
            print(f"  ✓ 极度恐慌/恐慌 + 情绪改善")
            print(f"    → 强烈买入信号！")
            print(f"    → 市场正从底部恢复，这是最佳买入时机")
            enhanced_position = min(signal['position'] + 0.2, 1.0)
        elif trend_7d['change'] < -5:
            print(f"  ⚠ 极度恐慌/恐慌 + 继续恶化")
            print(f"    → 谨慎买入，等待企稳信号")
            print(f"    → 可能还会继续下跌")
            enhanced_position = max(signal['position'] - 0.2, 0.0)
        else:
            print(f"  - 极度恐慌/恐慌 + 趋势稳定")
            print(f"    → 标准买入信号")
            enhanced_position = signal['position']
    
    elif current['value'] >= 70:
        if trend_7d['change'] < 0:
            print(f"  ✓ 极度贪婪/贪婪 + 情绪回落")
            print(f"    → 强烈卖出信号！")
            print(f"    → 市场见顶回落，考虑获利了结")
            enhanced_position = max(signal['position'] - 0.2, -1.0)
        elif trend_7d['change'] > 5:
            print(f"  ⚠ 极度贪婪/贪婪 + 继续上升")
            print(f"    → 谨慎卖出，泡沫可能继续膨胀")
            print(f"    → 但风险在累积")
            enhanced_position = min(signal['position'] + 0.1, 0.0)
        else:
            print(f"  - 极度贪婪/贪婪 + 趋势稳定")
            print(f"    → 标准卖出信号")
            enhanced_position = signal['position']
    
    else:
        print(f"  - 中性区间")
        print(f"    → 观望，等待更明确的信号")
        enhanced_position = 0.0
    
    print(f"\n  基础建议仓位: {signal['position']:.2f}")
    print(f"  趋势增强后: {enhanced_position:.2f}")
    
    # 统计分析
    print(f"\n【统计特征】")
    print("-" * 80)
    print(f"14天统计:")
    print(f"  最高: {df['value'].max()}")
    print(f"  最低: {df['value'].min()}")
    print(f"  平均: {df['value'].mean():.1f}")
    print(f"  标准差: {df['value'].std():.1f}")
    print(f"  波动率: {df['value'].std() / df['value'].mean() * 100:.1f}%")
    
    # 分类统计
    print(f"\n14天分类分布:")
    for classification, count in df['classification'].value_counts().items():
        print(f"  {classification}: {count}天 ({count/len(df)*100:.0f}%)")
    
    print("\n" + "="*80)
    print("提示：反向指标策略认为，极度恐慌是买入机会，极度贪婪是卖出机会")
    print("      趋势分析帮助识别最佳入场时机（恐慌中的恢复 或 贪婪中的回落）")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
