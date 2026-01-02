#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kalshi策略使用示例
演示如何使用Kalshi预测市场数据进行数字货币交易
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vquant.analysis.kalshi import KalshiTrader
from vquant.data.kalshi_fetcher import KalshiFetcher


def example_1_basic_analysis():
    """示例1：基础市场分析"""
    print("\n" + "="*60)
    print("示例1：基础市场分析")
    print("="*60)
    
    # 创建交易器（无需登录）
    trader = KalshiTrader(symbol="BTCUSDC", name="example1")
    
    # 分析市场
    result = trader.analyze(interval="1h", days=7)
    
    # 输出结果
    print(f"\n交易对: {result['symbol']}")
    print(f"策略: {result['strategy']}")
    print(f"建议仓位: {result['position']:.3f}")
    print(f"操作建议: {result['action']}")
    print(f"\n情绪数据:")
    print(f"  情绪得分: {result['sentiment']['sentiment_score']:.3f}")
    print(f"  置信度: {result['sentiment']['confidence']:.3f}")
    print(f"  相关市场: {result['sentiment']['markets_count']}")


def example_2_market_sentiment():
    """示例2：获取市场情绪"""
    print("\n" + "="*60)
    print("示例2：获取加密货币市场情绪")
    print("="*60)
    
    fetcher = KalshiFetcher()
    
    # 获取BTC情绪
    print("\n--- Bitcoin 情绪 ---")
    btc_sentiment = fetcher.get_crypto_sentiment("BTC")
    print(f"情绪得分: {btc_sentiment['sentiment_score']:.3f}")
    print(f"置信度: {btc_sentiment['confidence']:.3f}")
    print(f"市场数量: {btc_sentiment['markets_count']}")
    
    # 获取ETH情绪
    print("\n--- Ethereum 情绪 ---")
    eth_sentiment = fetcher.get_crypto_sentiment("ETH")
    print(f"情绪得分: {eth_sentiment['sentiment_score']:.3f}")
    print(f"置信度: {eth_sentiment['confidence']:.3f}")
    print(f"市场数量: {eth_sentiment['markets_count']}")


def example_3_search_markets():
    """示例3：搜索相关市场"""
    print("\n" + "="*60)
    print("示例3：搜索Kalshi预测市场")
    print("="*60)
    
    fetcher = KalshiFetcher()
    
    # 搜索比特币相关市场
    print("\n--- Bitcoin相关市场 ---")
    btc_markets = fetcher.search_markets("Bitcoin", limit=5)
    for i, market in enumerate(btc_markets, 1):
        print(f"{i}. {market.get('ticker')}")
        print(f"   标题: {market.get('title')}")
        print(f"   状态: {market.get('status')}")
        print(f"   交易量: {market.get('volume', 0)}")
        print()


def example_4_custom_config():
    """示例4：使用自定义配置"""
    print("\n" + "="*60)
    print("示例4：使用自定义配置")
    print("="*60)
    
    # 创建自定义配置
    import json
    import tempfile
    
    custom_config = {
        "sentiment_threshold_long": 0.70,  # 更保守
        "sentiment_threshold_short": 0.30,
        "confidence_threshold": 0.4,
        "max_position": 0.8,
        "use_technical_filter": True,
    }
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(custom_config, f, indent=2)
        config_path = f.name
    
    print(f"使用配置文件: {config_path}")
    print(f"配置内容: {json.dumps(custom_config, indent=2)}")
    
    # 使用配置创建交易器
    trader = KalshiTrader(
        symbol="ETHUSDC",
        name="example4",
        config_path=config_path
    )
    
    # 分析
    result = trader.analyze(interval="4h", days=7)
    print(f"\n建议仓位: {result['position']:.3f}")
    print(f"操作建议: {result['action']}")
    
    # 清理临时文件
    import os
    os.unlink(config_path)


def example_5_multi_symbol():
    """示例5：多币种分析"""
    print("\n" + "="*60)
    print("示例5：多币种分析对比")
    print("="*60)
    
    symbols = ["BTCUSDC", "ETHUSDC"]
    
    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        trader = KalshiTrader(symbol=symbol, name=f"multi_{symbol}")
        result = trader.analyze(interval="1h", days=7)
        
        print(f"建议仓位: {result['position']:.3f}")
        print(f"操作建议: {result['action']}")
        print(f"情绪得分: {result['sentiment']['sentiment_score']:.3f}")


def main():
    """运行所有示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    print("\n" + "="*60)
    print("Kalshi策略使用示例")
    print("="*60)
    
    try:
        # 运行示例（选择要运行的示例）
        example_1_basic_analysis()
        # example_2_market_sentiment()
        # example_3_search_markets()
        # example_4_custom_config()
        # example_5_multi_symbol()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("示例运行完成")
    print("="*60)


if __name__ == "__main__":
    main()
