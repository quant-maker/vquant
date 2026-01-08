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
    Fetch on-chain and derivatives data from Binance
    
    Data sources:
    1. Binance API - Futures data (open interest, long/short ratio, taker buy/sell)
    2. CryptoQuant API - Exchange flows (optional, requires API key)
    3. Glassnode API - On-chain metrics (optional, requires API key)
    
    Available data (no API key required):
    - Open Interest from Binance
    - Long/Short Ratio from Binance
    - Top Trader positions from Binance
    - Taker Buy/Sell Volume from Binance
    """
    
    def __init__(self, binance_base_url: str = "https://fapi.binance.com"):
        """
        Initialize OnChain data fetcher
        
        Args:
            binance_base_url: Binance Futures API base URL
        """
        self.binance_base = binance_base_url
        self.usdm_client = None
        
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
    
    def fetch_global_long_short_ratio(self, symbol: str = "BTCUSDT",
                                     period: str = "5m",
                                     limit: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch global long/short account ratio (all users)
        
        Args:
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points (max 500)
            
        Returns:
            List of global long/short ratio data
        """
        try:
            url = f"{self.binance_base}/futures/data/globalLongShortAccountRatio"
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
                logger.info(f"Global Long/Short Ratio: {float(latest['longShortRatio']):.4f} "
                          f"(Long: {float(latest['longAccount']):.2%}, "
                          f"Short: {float(latest['shortAccount']):.2%})")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch global long/short ratio: {e}")
            return None
    
    def fetch_premium_index(self, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """
        Fetch premium index (mark price, index price, funding rate)
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Premium index data or None on failure
        """
        try:
            url = f"{self.binance_base}/fapi/v1/premiumIndex"
            params = {"symbol": symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                mark_price = float(data['markPrice'])
                index_price = float(data['indexPrice'])
                premium = (mark_price - index_price) / index_price * 100
                
                logger.info(f"Premium: {premium:.4f}% (Mark: {mark_price:.2f}, Index: {index_price:.2f})")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch premium index: {e}")
            return None
    
    def fetch_24hr_stats(self, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """
        Fetch 24hr ticker statistics
        
        Args:
            symbol: Trading symbol
            
        Returns:
            24hr stats or None on failure
        """
        try:
            url = f"{self.binance_base}/fapi/v1/ticker/24hr"
            params = {"symbol": symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                volume = float(data['volume'])
                price_change_pct = float(data['priceChangePercent'])
                
                logger.info(f"24h Volume: {volume:.2f}, Price Change: {price_change_pct:.2f}%")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch 24hr stats: {e}")
            return None
    

    
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
            'global_long_short_ratio': None,
            'taker_volume': None,
            'premium_index': None,
            '24hr_stats': None,
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
        
        metrics['global_long_short_ratio'] = self.fetch_global_long_short_ratio(
            symbol, period, history_limit
        )
        time.sleep(0.2)
        
        metrics['taker_volume'] = self.fetch_taker_buy_sell_volume(
            symbol, period, history_limit
        )
        time.sleep(0.2)
        
        metrics['premium_index'] = self.fetch_premium_index(symbol)
        time.sleep(0.2)
        
        metrics['24hr_stats'] = self.fetch_24hr_stats(symbol)
        
        logger.info("On-chain metrics fetch completed")
        return metrics
    
    def analyze_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze on-chain metrics and generate trading signals
        
        Args:
            metrics: Dictionary of on-chain metrics
            
        Returns:
            Analysis result with signal strength
        """
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
        
        # Analyze Global Long/Short Ratio (all users)
        if metrics.get('global_long_short_ratio'):
            global_ls_data = metrics['global_long_short_ratio']
            if global_ls_data:
                latest_global_ls = float(global_ls_data[-1]['longShortRatio'])
                
                if latest_global_ls > 3.0:
                    analysis['bearish_signals'] += 1
                    analysis['signal_details'].append(
                        f"全市场多空比极高 {latest_global_ls:.2f} (极度看多，反向指标)"
                    )
                elif latest_global_ls < 0.5:
                    analysis['bullish_signals'] += 1
                    analysis['signal_details'].append(
                        f"全市场多空比极低 {latest_global_ls:.2f} (极度看空，反向指标)"
                    )
                elif latest_global_ls > 1.5:
                    analysis['bearish_signals'] += 1
                    analysis['signal_details'].append(
                        f"全市场偏多 {latest_global_ls:.2f} (散户看多)"
                    )
                elif latest_global_ls < 0.8:
                    analysis['bullish_signals'] += 1
                    analysis['signal_details'].append(
                        f"全市场偏空 {latest_global_ls:.2f} (散户看空)"
                    )
                else:
                    analysis['neutral_signals'] += 1
        
        # Analyze Premium Index
        if metrics.get('premium_index'):
            premium_data = metrics['premium_index']
            mark_price = float(premium_data['markPrice'])
            index_price = float(premium_data['indexPrice'])
            premium = (mark_price - index_price) / index_price * 100
            
            if premium > 0.5:
                analysis['bearish_signals'] += 1
                analysis['signal_details'].append(
                    f"合约溢价过高 {premium:.3f}% (市场过热)"
                )
            elif premium < -0.5:
                analysis['bullish_signals'] += 1
                analysis['signal_details'].append(
                    f"合约折价 {premium:.3f}% (市场恐慌)"
                )
            else:
                analysis['neutral_signals'] += 1
                analysis['signal_details'].append(
                    f"合约溢价正常 {premium:.3f}%"
                )
        
        # Analyze 24hr Volume and Price Change
        if metrics.get('24hr_stats'):
            stats_24h = metrics['24hr_stats']
            price_change_pct = float(stats_24h['priceChangePercent'])
            volume = float(stats_24h['volume'])
            
            # Price momentum
            if abs(price_change_pct) > 5:
                if price_change_pct > 0:
                    analysis['bullish_signals'] += 1
                    analysis['signal_details'].append(
                        f"24h强势上涨 {price_change_pct:.2f}%"
                    )
                else:
                    analysis['bearish_signals'] += 1
                    analysis['signal_details'].append(
                        f"24h大幅下跌 {price_change_pct:.2f}%"
                    )
            else:
                analysis['neutral_signals'] += 1
        
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
