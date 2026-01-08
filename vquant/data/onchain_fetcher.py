#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
OnChain Data Fetcher - Fetch on-chain and derivative market data
Supports multiple data sources for comprehensive market analysis
"""

import logging
import requests
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OnChainFetcher:
    """
    Fetch on-chain and derivatives data from multiple sources
    
    Data sources:
    1. Binance API - Futures data (open interest, long/short ratio, taker buy/sell)
    2. CryptoQuant API - Exchange flows (optional, requires API key)
    3. Glassnode API - On-chain metrics (optional, requires API key)
    
    Free tier data (no API key required):
    - Open Interest from Binance
    - Long/Short Ratio from Binance
    - Top Trader positions from Binance
    - Taker Buy/Sell Volume from Binance
    
    Authenticated data (optional):
    - Liquidation orders (requires API keys in environment or config)
    """
    
    def __init__(self, binance_base_url: str = "https://fapi.binance.com",
                 account: str = None):
        """
        Initialize OnChain data fetcher
        
        Args:
            binance_base_url: Binance Futures API base URL
            account: Optional account name for loading API credentials
                    If provided, will enable authenticated liquidation data
        """
        self.binance_base = binance_base_url
        self.account = account
        self.usdm_client = None
        
        # Try to initialize authenticated client if account provided
        if account:
            try:
                import sys
                if '/home/ubuntu/bcp' not in sys.path:
                    sys.path.insert(0, '/home/ubuntu/bcp')
                
                from binance.fut.usdm import USDM
                from binance.auth.utils import load_api_keys
                
                api_key, private_key = load_api_keys(account)
                self.usdm_client = USDM(api_key=api_key, private_key=private_key)
                logger.info(f"Authenticated USDM client initialized for account: {account}")
            except Exception as e:
                logger.warning(f"Failed to initialize authenticated client: {e}")
                logger.info("Continuing with unauthenticated mode (liquidation data unavailable)")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
    
    def fetch_open_interest(self, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """
        Fetch current open interest from Binance
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            
        Returns:
            Open interest data or None on failure
        """
        try:
            url = f"{self.binance_base}/fapi/v1/openInterest"
            params = {"symbol": symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Open Interest: {data.get('openInterest')} contracts")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch open interest: {e}")
            return None
    
    def fetch_open_interest_history(self, symbol: str = "BTCUSDT", 
                                   period: str = "5m", 
                                   limit: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch open interest historical data
        
        Args:
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points (max 500)
            
        Returns:
            List of historical open interest data
        """
        try:
            url = f"{self.binance_base}/futures/data/openInterestHist"
            params = {
                "symbol": symbol,
                "period": period,
                "limit": limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched {len(data)} open interest history points")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch OI history: {e}")
            return None
    
    def fetch_long_short_ratio(self, symbol: str = "BTCUSDT", 
                              period: str = "5m", 
                              limit: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch top trader long/short position ratio
        
        Args:
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points (max 500)
            
        Returns:
            List of long/short ratio data
        """
        try:
            url = f"{self.binance_base}/futures/data/topLongShortPositionRatio"
            params = {
                "symbol": symbol,
                "period": period,
                "limit": limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                latest = data[-1]
                logger.info(f"Long/Short Ratio: {float(latest['longShortRatio']):.4f} "
                          f"(Long: {float(latest['longAccount']):.2f}%, "
                          f"Short: {float(latest['shortAccount']):.2f}%)")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch long/short ratio: {e}")
            return None
    
    def fetch_taker_buy_sell_volume(self, symbol: str = "BTCUSDT", 
                                   period: str = "5m", 
                                   limit: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch taker buy/sell volume ratio
        
        Args:
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points (max 500)
            
        Returns:
            List of taker volume data
        """
        try:
            url = f"{self.binance_base}/futures/data/takerlongshortRatio"
            params = {
                "symbol": symbol,
                "period": period,
                "limit": limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                latest = data[-1]
                ratio = float(latest['buySellRatio'])
                logger.info(f"Taker Buy/Sell Ratio: {ratio:.4f}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch taker volume: {e}")
            return None
    
    def fetch_liquidation_orders(self, symbol: str = "BTCUSDT", 
                                start_time: Optional[int] = None,
                                end_time: Optional[int] = None,
                                limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch recent liquidation orders
        
        Two modes:
        1. With USDM client (authenticated): Gets complete liquidation data
        2. Without authentication: Returns empty list with warning
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            limit: Number of liquidation records (max 1000)
            
        Returns:
            List of liquidation data or empty list if unavailable
        """
        # Try authenticated method first if client available
        if self.usdm_client:
            try:
                params = {"symbol": symbol, "limit": limit}
                if start_time:
                    params['startTime'] = start_time
                if end_time:
                    params['endTime'] = end_time
                
                # Use authenticated request
                data = self.usdm_client.sign_request(
                    "GET", 
                    "/fapi/v1/allForceOrders",
                    params
                )
                
                # Calculate liquidation statistics
                if data:
                    long_liq = sum(float(x['origQty']) for x in data if x['side'] == 'SELL')
                    short_liq = sum(float(x['origQty']) for x in data if x['side'] == 'BUY')
                    total_liq = long_liq + short_liq
                    
                    logger.info(f"Liquidations: Long={long_liq:.2f}, Short={short_liq:.2f}, "
                              f"Total={total_liq:.2f} contracts")
                
                return data
                
            except Exception as e:
                logger.warning(f"Authenticated liquidation fetch failed: {e}")
                return []
        
        # Try unauthenticated public endpoint (usually requires auth)
        try:
            url = f"{self.binance_base}/fapi/v1/allForceOrders"
            params = {"symbol": symbol}
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            if limit:
                params['limit'] = limit
            
            response = self.session.get(url, params=params, timeout=10)
            
            # If 400 error, likely means authentication required
            if response.status_code == 400:
                logger.debug("Liquidation data requires authentication, skipping...")
                return []
            
            response.raise_for_status()
            data = response.json()
            
            # Calculate liquidation statistics
            if data:
                long_liq = sum(float(x['origQty']) for x in data if x['side'] == 'SELL')
                short_liq = sum(float(x['origQty']) for x in data if x['side'] == 'BUY')
                total_liq = long_liq + short_liq
                
                logger.info(f"Liquidations: Long={long_liq:.2f}, Short={short_liq:.2f}, "
                          f"Total={total_liq:.2f} contracts")
            
            return data
            
        except Exception as e:
            logger.debug(f"Unauthenticated liquidation fetch failed (expected): {e}")
            return []
    
    def fetch_all_metrics(self, symbol: str = "BTCUSDT", 
                         period: str = "5m", 
                         history_limit: int = 30) -> Dict[str, Any]:
        """
        Fetch all available on-chain and derivatives metrics
        
        Args:
            symbol: Trading symbol
            period: Time period for historical data
            history_limit: Number of historical data points
            
        Returns:
            Dictionary containing all metrics
        """
        logger.info(f"Fetching on-chain metrics for {symbol}...")
        
        metrics = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'open_interest': None,
            'open_interest_history': None,
            'long_short_ratio': None,
            'taker_volume': None,
            'liquidations': None,
        }
        
        # Fetch all metrics with small delays to avoid rate limits
        metrics['open_interest'] = self.fetch_open_interest(symbol)
        time.sleep(0.2)
        
        metrics['open_interest_history'] = self.fetch_open_interest_history(
            symbol, period, history_limit
        )
        time.sleep(0.2)
        
        metrics['long_short_ratio'] = self.fetch_long_short_ratio(
            symbol, period, history_limit
        )
        time.sleep(0.2)
        
        metrics['taker_volume'] = self.fetch_taker_buy_sell_volume(
            symbol, period, history_limit
        )
        time.sleep(0.2)
        
        metrics['liquidations'] = self.fetch_liquidation_orders(symbol, 100)
        
        logger.info("On-chain metrics fetch completed")
        return metrics
    
    def analyze_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze on-chain metrics and generate trading signals
        
        Args:
            metrics: Dictionary of on-chain metrics
            
        Returns:
            Analysis result with signal strength
            
        Raises:
            ValueError: If liquidation data is not available
        """
        # Require liquidation data
        if not metrics.get('liquidations') or len(metrics['liquidations']) == 0:
            error_msg = (
                "清算数据不可用。OnChain策略必须要有清算数据才能运行。\n"
                "请使用 --account 参数提供认证账户以获取清算数据。\n"
                "例如: python main.py --predictor onchain --account your_account_name"
            )
            logger.error(error_msg)
            raise ValueError("Liquidation data is required for OnChain strategy. "
                           "Please provide --account parameter for authentication.")
        
        analysis = {
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'signal_details': [],
            'overall_sentiment': 'NEUTRAL'
        }
        
        # Analyze Open Interest trend
        if metrics['open_interest_history']:
            oi_data = metrics['open_interest_history']
            if len(oi_data) >= 2:
                recent_oi = float(oi_data[-1]['sumOpenInterest'])
                past_oi = float(oi_data[0]['sumOpenInterest'])
                oi_change = (recent_oi - past_oi) / past_oi * 100
                
                if oi_change > 5:
                    analysis['bullish_signals'] += 1
                    analysis['signal_details'].append(
                        f"OI增加 {oi_change:.2f}% (多头兴趣增强)"
                    )
                elif oi_change < -5:
                    analysis['bearish_signals'] += 1
                    analysis['signal_details'].append(
                        f"OI减少 {abs(oi_change):.2f}% (市场兴趣降低)"
                    )
                else:
                    analysis['neutral_signals'] += 1
        
        # Analyze Long/Short Ratio
        if metrics['long_short_ratio']:
            ls_data = metrics['long_short_ratio']
            if ls_data:
                latest_ls = float(ls_data[-1]['longShortRatio'])
                
                if latest_ls > 2.0:
                    analysis['bearish_signals'] += 1
                    analysis['signal_details'].append(
                        f"多空比过高 {latest_ls:.2f} (过度看多，反向指标)"
                    )
                elif latest_ls < 0.5:
                    analysis['bullish_signals'] += 1
                    analysis['signal_details'].append(
                        f"多空比过低 {latest_ls:.2f} (过度看空，反向指标)"
                    )
                elif latest_ls > 1.2:
                    analysis['bullish_signals'] += 1
                    analysis['signal_details'].append(
                        f"多空比偏多 {latest_ls:.2f} (看多情绪)"
                    )
                elif latest_ls < 0.8:
                    analysis['bearish_signals'] += 1
                    analysis['signal_details'].append(
                        f"多空比偏空 {latest_ls:.2f} (看空情绪)"
                    )
                else:
                    analysis['neutral_signals'] += 1
        
        # Analyze Taker Buy/Sell
        if metrics['taker_volume']:
            taker_data = metrics['taker_volume']
            if taker_data:
                recent_ratios = [float(x['buySellRatio']) for x in taker_data[-5:]]
                avg_ratio = sum(recent_ratios) / len(recent_ratios)
                
                if avg_ratio > 1.2:
                    analysis['bullish_signals'] += 1
                    analysis['signal_details'].append(
                        f"主动买入强劲 {avg_ratio:.2f} (买盘压力)"
                    )
                elif avg_ratio < 0.8:
                    analysis['bearish_signals'] += 1
                    analysis['signal_details'].append(
                        f"主动卖出强劲 {avg_ratio:.2f} (卖盘压力)"
                    )
                else:
                    analysis['neutral_signals'] += 1
        
        # Analyze Liquidations (required)
        # This check is redundant since we already checked at the beginning,
        # but kept for safety
        liq_data = metrics['liquidations']
        long_liq = sum(float(x['origQty']) for x in liq_data if x['side'] == 'SELL')
        short_liq = sum(float(x['origQty']) for x in liq_data if x['side'] == 'BUY')
        
        if long_liq > short_liq * 2:
            analysis['bearish_signals'] += 1
            analysis['signal_details'].append(
                f"多头爆仓严重 (多单={long_liq:.0f}, 空单={short_liq:.0f})"
            )
        elif short_liq > long_liq * 2:
            analysis['bullish_signals'] += 1
            analysis['signal_details'].append(
                f"空头爆仓严重 (空单={short_liq:.0f}, 多单={long_liq:.0f})"
            )
        else:
            analysis['neutral_signals'] += 1
            analysis['signal_details'].append(
                f"爆仓平衡 (多单={long_liq:.0f}, 空单={short_liq:.0f})"
            )
        
        # Determine overall sentiment
        total_signals = (analysis['bullish_signals'] + 
                        analysis['bearish_signals'] + 
                        analysis['neutral_signals'])
        
        bull_pct = 0.0
        bear_pct = 0.0
        
        if total_signals > 0:
            bull_pct = analysis['bullish_signals'] / total_signals
            bear_pct = analysis['bearish_signals'] / total_signals
            
            if bull_pct > 0.6:
                analysis['overall_sentiment'] = 'BULLISH'
            elif bear_pct > 0.6:
                analysis['overall_sentiment'] = 'BEARISH'
            else:
                analysis['overall_sentiment'] = 'NEUTRAL'
        
        analysis['confidence'] = max(bull_pct, bear_pct) if total_signals > 0 else 0.0
        
        return analysis


def test_onchain_fetcher():
    """Test on-chain data fetcher"""
    fetcher = OnChainFetcher()
    
    # Test all metrics
    metrics = fetcher.fetch_all_metrics("BTCUSDT", period="1h", history_limit=24)
    
    # Analyze metrics
    analysis = fetcher.analyze_metrics(metrics)
    
    print("\n" + "=" * 60)
    print("ON-CHAIN ANALYSIS RESULT")
    print("=" * 60)
    print(f"Overall Sentiment: {analysis['overall_sentiment']}")
    print(f"Confidence: {analysis['confidence']:.2%}")
    print(f"\nSignal Summary:")
    print(f"  Bullish: {analysis['bullish_signals']}")
    print(f"  Bearish: {analysis['bearish_signals']}")
    print(f"  Neutral: {analysis['neutral_signals']}")
    print(f"\nSignal Details:")
    for detail in analysis['signal_details']:
        print(f"  - {detail}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"
    )
    test_onchain_fetcher()
