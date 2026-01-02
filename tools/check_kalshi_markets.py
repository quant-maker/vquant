#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
检查Kalshi当前可用的市场
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vquant.data.kalshi_fetcher import KalshiFetcher


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    fetcher = KalshiFetcher()
    
    # 获取所有开放市场
    print("\n" + "="*80)
    print("Kalshi 当前开放的市场")
    print("="*80)
    
    params = {"limit": 200, "status": "open"}
    response = fetcher.session.get(f"{fetcher.BASE_URL}/markets", params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        markets = data.get("markets", [])
        
        print(f"\n共找到 {len(markets)} 个开放市场\n")
        
        # 按类别分组
        categories = {}
        for market in markets:
            title = market.get("title", "")
            ticker = market.get("ticker", "")
            category = market.get("category", "其他")
            
            if category not in categories:
                categories[category] = []
            categories[category].append((ticker, title))
        
        # 显示每个类别
        for category, items in sorted(categories.items()):
            print(f"\n【{category}】({len(items)} 个市场)")
            print("-" * 80)
            for ticker, title in items[:5]:  # 只显示前5个
                print(f"  {ticker}")
                print(f"  {title[:100]}...")
                print()
            if len(items) > 5:
                print(f"  ... 还有 {len(items) - 5} 个市场\n")
        
        # 搜索可能与加密货币相关的关键词
        crypto_keywords = ["crypto", "bitcoin", "btc", "ethereum", "eth", "coin", "blockchain", "currency"]
        print("\n" + "="*80)
        print("搜索加密货币相关市场")
        print("="*80)
        
        found_crypto = False
        for market in markets:
            title = market.get("title", "").lower()
            subtitle = market.get("subtitle", "").lower()
            
            if any(keyword in title or keyword in subtitle for keyword in crypto_keywords):
                found_crypto = True
                print(f"\n✓ 找到: {market.get('ticker')}")
                print(f"  标题: {market.get('title')}")
                print(f"  类别: {market.get('category')}")
        
        if not found_crypto:
            print("\n✗ 未找到加密货币相关市场")
            print("\n提示: Kalshi可能没有加密货币相关的预测市场。")
            print("      你可以考虑以下替代方案：")
            print("      1. 使用其他预测市场平台（如Polymarket）")
            print("      2. 使用社交媒体情绪分析")
            print("      3. 使用链上数据分析")
            print("      4. 关注相关的经济/政治市场（可能间接影响加密货币）")
    
    else:
        print(f"\n错误: API返回状态码 {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
