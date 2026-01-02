#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kalshi Data Fetcher - 从Kalshi预测市场获取数据

Kalshi是一个预测市场平台，用户可以交易关于各种事件的二元期权。
我们可以利用这些预测数据来辅助数字货币交易决策。

相关市场示例：
- 比特币价格预测市场
- 以太坊价格预测市场
- 加密货币监管政策市场
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from time import sleep

logger = logging.getLogger(__name__)


class KalshiFetcher:
    """Kalshi数据获取器"""
    
    # Kalshi API端点（已迁移到新地址）
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        """
        初始化Kalshi数据获取器
        
        Args:
            email: Kalshi账户邮箱
            password: Kalshi账户密码
        """
        self.email = email
        self.password = password
        self.token = None
        self.session = requests.Session()
        
        # 如果提供了账户信息，尝试登录
        if email and password:
            self.login()
    
    def login(self) -> bool:
        """登录Kalshi账户"""
        try:
            response = self.session.post(
                f"{self.BASE_URL}/login",
                json={"email": self.email, "password": self.password}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data.get("token")
            if self.token:
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                logger.info("成功登录Kalshi账户")
                return True
            else:
                logger.error("登录失败: 未返回token")
                return False
        except Exception as e:
            logger.error(f"登录Kalshi失败: {e}")
            return False
    
    def search_markets(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        搜索相关市场
        
        Args:
            query: 搜索关键词（如 "Bitcoin", "Ethereum", "crypto"）
            limit: 返回结果数量
            
        Returns:
            市场列表
        """
        try:
            params = {
                "limit": limit,
                "status": "open",  # 只获取开放的市场
            }
            
            # 如果提供了搜索词，添加到查询中
            if query:
                params["event_ticker"] = query
            
            response = self.session.get(
                f"{self.BASE_URL}/markets",
                params=params,
                timeout=10
            )
            
            # 打印调试信息
            logger.debug(f"请求URL: {response.url}")
            logger.debug(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                logger.info(f"找到 {len(markets)} 个相关市场")
                
                # 打印市场详情用于调试
                if markets:
                    for market in markets[:3]:  # 只打印前3个
                        logger.debug(f"  市场: {market.get('ticker')} - {market.get('title')}")
                
                return markets
            else:
                logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return []
            
        except Exception as e:
            logger.error(f"搜索市场失败: {e}", exc_info=True)
            return []
    
    def get_market_details(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        获取市场详细信息
        
        Args:
            market_ticker: 市场代码
            
        Returns:
            市场详情
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/markets/{market_ticker}"
            )
            response.raise_for_status()
            data = response.json()
            return data.get("market")
        except Exception as e:
            logger.error(f"获取市场详情失败 {market_ticker}: {e}")
            return None
    
    def get_orderbook(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        获取市场订单簿
        
        Args:
            market_ticker: 市场代码
            
        Returns:
            订单簿数据
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/markets/{market_ticker}/orderbook"
            )
            response.raise_for_status()
            data = response.json()
            return data.get("orderbook")
        except Exception as e:
            logger.error(f"获取订单簿失败 {market_ticker}: {e}")
            return None
    
    def get_market_history(self, market_ticker: str, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        获取市场历史交易数据
        
        Args:
            market_ticker: 市场代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            历史数据DataFrame
        """
        try:
            params = {}
            if start_date:
                params["min_ts"] = int(start_date.timestamp())
            if end_date:
                params["max_ts"] = int(end_date.timestamp())
            
            response = self.session.get(
                f"{self.BASE_URL}/markets/{market_ticker}/history",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            # 转换为DataFrame
            history = data.get("history", [])
            if not history:
                logger.warning(f"没有找到历史数据: {market_ticker}")
                return pd.DataFrame()
            
            df = pd.DataFrame(history)
            if "ts" in df.columns:
                df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
                df.set_index("timestamp", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取市场历史失败 {market_ticker}: {e}")
            return pd.DataFrame()
    
    def get_crypto_sentiment(self, crypto: str = "BTC") -> Dict[str, Any]:
        """
        获取加密货币相关的市场情绪指标
        
        Args:
            crypto: 加密货币符号 (BTC, ETH, etc.)
            
        Returns:
            情绪指标字典
        """
        try:
            # 首先获取所有开放的市场
            logger.info(f"获取所有开放市场以搜索 {crypto} 相关内容...")
            
            params = {
                "limit": 200,  # 获取更多市场
                "status": "open",
            }
            
            response = self.session.get(
                f"{self.BASE_URL}/markets",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"获取市场列表失败: {response.status_code}")
                return {"sentiment_score": 0.5, "confidence": 0, "markets_count": 0}
            
            data = response.json()
            all_markets = data.get("markets", [])
            logger.info(f"获取到 {len(all_markets)} 个开放市场")
            
            # 在标题中搜索加密货币相关关键词
            crypto_keywords = {
                "BTC": ["bitcoin", "btc", "crypto", "cryptocurrency"],
                "ETH": ["ethereum", "eth", "crypto", "cryptocurrency"],
                "DOGE": ["dogecoin", "doge", "crypto"],
            }
            
            keywords = crypto_keywords.get(crypto.upper(), [crypto.lower(), "crypto"])
            relevant_markets = []
            
            for market in all_markets:
                title = market.get("title", "").lower()
                subtitle = market.get("subtitle", "").lower()
                
                # 检查标题或副标题是否包含关键词
                if any(keyword in title or keyword in subtitle for keyword in keywords):
                    relevant_markets.append(market)
                    logger.info(f"找到相关市场: {market.get('ticker')} - {market.get('title')}")
            
            if not relevant_markets:
                logger.warning(f"在 {len(all_markets)} 个市场中没有找到 {crypto} 相关的市场")
                return {"sentiment_score": 0.5, "confidence": 0, "markets_count": 0}
            
            logger.info(f"找到 {len(relevant_markets)} 个 {crypto} 相关市场")
            
            # 计算综合情绪得分
            sentiment_scores = []
            total_volume = 0
            
            for market in relevant_markets:
                ticker = market.get("ticker")
                if not ticker:
                    continue
                
                # 获取订单簿
                orderbook = self.get_orderbook(ticker)
                if not orderbook:
                    continue
                
                # 从yes合约价格推断情绪
                yes_asks = orderbook.get("yes", [])
                if yes_asks and len(yes_asks) > 0:
                    yes_price = yes_asks[0].get("price", 5000) / 100  # 转换为0-1
                    sentiment_scores.append(yes_price)
                    logger.debug(f"  {ticker}: yes_price={yes_price:.3f}")
                
                # 累计交易量作为置信度
                volume = market.get("volume", 0)
                total_volume += volume
                
                sleep(0.2)  # 避免API限流
            
            if not sentiment_scores:
                logger.warning("无法从任何市场获取价格数据")
                return {"sentiment_score": 0.5, "confidence": 0, "markets_count": len(relevant_markets)}
            
            # 计算平均情绪得分
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            return {
                "sentiment_score": avg_sentiment,  # 0-1之间，越高越看涨
                "confidence": min(total_volume / 10000, 1.0),  # 基于交易量的置信度
                "markets_count": len(relevant_markets),
                "individual_scores": sentiment_scores,
            }
            
        except Exception as e:
            logger.error(f"获取加密货币情绪失败: {e}", exc_info=True)
            return {"sentiment_score": 0.5, "confidence": 0, "markets_count": 0}
    
    def get_btc_price_prediction(self, threshold: float = 50000) -> Dict[str, Any]:
        """
        获取比特币价格预测
        
        Args:
            threshold: 价格阈值
            
        Returns:
            价格预测信息
        """
        try:
            # 搜索比特币价格相关市场
            markets = self.search_markets("Bitcoin", limit=20)
            
            price_predictions = []
            for market in markets:
                title = market.get("title", "").lower()
                ticker = market.get("ticker", "")
                
                # 查找包含价格信息的市场
                if "price" in title or "$" in title:
                    orderbook = self.get_orderbook(ticker)
                    if orderbook:
                        yes_price = orderbook.get("yes", [{}])[0].get("price", 50) / 100
                        price_predictions.append({
                            "market": title,
                            "ticker": ticker,
                            "probability": yes_price,
                            "volume": market.get("volume", 0)
                        })
                    sleep(0.1)
            
            return {
                "predictions": price_predictions,
                "count": len(price_predictions)
            }
            
        except Exception as e:
            logger.error(f"获取BTC价格预测失败: {e}")
            return {"predictions": [], "count": 0}


def main():
    """测试函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # 创建fetcher（无需登录也可以获取公开数据）
    fetcher = KalshiFetcher()
    
    # 测试API连接
    logger.info("=== 测试Kalshi API连接 ===")
    try:
        response = fetcher.session.get(f"{fetcher.BASE_URL}/markets", params={"limit": 5}, timeout=10)
        logger.info(f"API状态: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"成功连接! 获取到 {len(data.get('markets', []))} 个市场")
            # 显示前几个市场
            for market in data.get('markets', [])[:3]:
                logger.info(f"  - {market.get('ticker')}: {market.get('title')}")
        else:
            logger.error(f"API错误: {response.text}")
    except Exception as e:
        logger.error(f"连接失败: {e}")
    
    # 获取加密货币情绪
    logger.info("\n=== 获取BTC市场情绪 ===")
    btc_sentiment = fetcher.get_crypto_sentiment("BTC")
    logger.info(f"BTC情绪得分: {btc_sentiment['sentiment_score']:.2f}")
    logger.info(f"置信度: {btc_sentiment['confidence']:.2f}")
    logger.info(f"相关市场数: {btc_sentiment['markets_count']}")


if __name__ == "__main__":
    main()
