#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
OnChain Trader - Trading strategy based on on-chain and derivatives data
Analyzes multiple on-chain metrics to generate trading signals

IMPORTANT: This strategy REQUIRES liquidation data which needs authentication.
You must provide --account parameter when running this strategy.
Example: python main.py --predictor onchain --account your_account_name
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from .base import BasePredictor
from ..data.onchain_fetcher import OnChainFetcher

logger = logging.getLogger(__name__)


class OnChainTrader(BasePredictor):
    """
    On-Chain Data Trading Strategy
    
    IMPORTANT: This strategy REQUIRES liquidation data and authentication.
    You MUST provide --account parameter when using this strategy.
    Example: python main.py --predictor onchain --account your_account_name
    
    Features:
    1. Fetch real-time on-chain and derivatives data from Binance
    2. Analyze multiple metrics:
       - Open Interest changes
       - Long/Short ratio (top traders)
       - Taker buy/sell volume ratio
       - Liquidation data (REQUIRED - needs authentication)
    3. Combine on-chain signals with technical analysis
    4. Generate position signals based on sentiment
    
    Signal Generation:
    - Bullish: High taker buy ratio, rising OI, short liquidations
    - Bearish: High taker sell ratio, falling OI, long liquidations
    - Neutral: Mixed or weak signals
    """
    
    def __init__(self, symbol: str = "BTCUSDC", name: str = "onchain",
                 config_path: Optional[str] = None, account: str = None):
        """
        Initialize OnChain trader
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDC)
            name: Strategy name
            config_path: Path to configuration file
            account: Account name for authenticated liquidation data (REQUIRED)
                    Without this, the strategy will fail when trying to fetch
                    liquidation data. Example: 'your_account_name'
        
        Raises:
            Warning: If account is not provided, strategy will fail during analysis
        """
        super().__init__(symbol, name)
        
        if not account:
            logger.warning(
                "⚠️  OnChain策略需要认证账户才能获取清算数据！\n"
                "   请使用 --account 参数提供账户名称\n"
                "   例如: python main.py --predictor onchain --account your_account_name"
            )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Use the symbol directly without conversion
        self.futures_symbol = symbol
        
        # Initialize on-chain data fetcher with optional account for authentication
        self.fetcher = OnChainFetcher(account=account)
        
        # Strategy parameters
        self.data_period = self.config.get('data_period', '1h')
        self.history_limit = self.config.get('history_limit', 24)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        
        # Position sizing
        self.max_position = self.config.get('max_position', 1.0)
        self.min_position = self.config.get('min_position', 0.3)
        
        # Signal weights
        self.weights = self.config.get('signal_weights', {
            'open_interest': 1.0,
            'long_short_ratio': 1.0,
            'taker_volume': 1.5,
            'liquidations': 1.2
        })
        
        logger.info(f"OnChain Trader initialized: {symbol} -> {self.futures_symbol}")
        logger.info(f"Data period: {self.data_period}, History: {self.history_limit}")
        logger.info(f"Min confidence: {self.min_confidence}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return {}
    
    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, 
                    stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Prepare data for analysis (no chart generation for on-chain strategy)
        
        Returns:
            Tuple of (None, None) - no chart generation
        """
        # On-chain strategy doesn't generate charts
        # All data is fetched fresh during analyze()
        return None, None
    
    def analyze(self, stats: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Core analysis logic - fetch and analyze on-chain metrics
        
        Args:
            stats: Market statistics from technical analysis
            **kwargs: Additional parameters
            
        Returns:
            Analysis result with position signal
            
        Raises:
            ValueError: If liquidation data is not available
        """
        logger.info("Fetching on-chain metrics...")
        
        # Fetch on-chain metrics
        metrics = self.fetcher.fetch_all_metrics(
            symbol=self.futures_symbol,
            period=self.data_period,
            history_limit=self.history_limit
        )
        
        # Validate liquidation data availability
        if not metrics.get('liquidations') or len(metrics['liquidations']) == 0:
            error_msg = (
                "\n清算数据不可用！OnChain策略必须要有清算数据才能运行。\n"
                "请使用 --account 参数提供认证账户以获取清算数据。\n"
                "例如: python main.py --predictor onchain --account your_account_name\n"
            )
            logger.error(error_msg)
            raise ValueError(
                "Liquidation data is required for OnChain strategy. "
                "Please provide --account parameter for authentication."
            )
        
        logger.info(f"清算数据已获取: {len(metrics['liquidations'])} 条记录")
        
        # Analyze metrics (will also validate liquidation data)
        onchain_analysis = self.fetcher.analyze_metrics(metrics)
        
        # Get technical analysis sentiment from stats
        tech_sentiment = self._analyze_technical_stats(stats)
        
        # Combine on-chain and technical analysis
        combined_analysis = self._combine_signals(onchain_analysis, tech_sentiment)
        
        return {
            'metrics': metrics,
            'onchain_analysis': onchain_analysis,
            'tech_sentiment': tech_sentiment,
            'combined_analysis': combined_analysis
        }
    
    def _analyze_technical_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze technical indicators from stats
        
        Args:
            stats: Market statistics
            
        Returns:
            Technical analysis sentiment
        """
        sentiment = {
            'direction': 'NEUTRAL',
            'strength': 0.5,
            'signals': []
        }
        
        try:
            # Price vs MA analysis
            current_price = stats.get('current_price', 0)
            ma7 = stats.get('ma7', 0)
            ma25 = stats.get('ma25', 0)
            ma99 = stats.get('ma99', 0)
            
            bullish_count = 0
            bearish_count = 0
            
            if current_price > ma7:
                bullish_count += 1
            else:
                bearish_count += 1
            
            if current_price > ma25:
                bullish_count += 1
            else:
                bearish_count += 1
            
            if current_price > ma99:
                bullish_count += 1
            else:
                bearish_count += 1
            
            # RSI analysis
            rsi = stats.get('rsi', 50)
            if rsi > 70:
                bearish_count += 1
                sentiment['signals'].append(f"RSI超买 ({rsi:.1f})")
            elif rsi < 30:
                bullish_count += 1
                sentiment['signals'].append(f"RSI超卖 ({rsi:.1f})")
            
            # Determine sentiment
            total = bullish_count + bearish_count
            if total > 0:
                bull_ratio = bullish_count / total
                if bull_ratio > 0.6:
                    sentiment['direction'] = 'BULLISH'
                    sentiment['strength'] = bull_ratio
                elif bull_ratio < 0.4:
                    sentiment['direction'] = 'BEARISH'
                    sentiment['strength'] = 1 - bull_ratio
                else:
                    sentiment['direction'] = 'NEUTRAL'
                    sentiment['strength'] = 0.5
        
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
        
        return sentiment
    
    def _combine_signals(self, onchain_analysis: Dict[str, Any], 
                        tech_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine on-chain and technical signals
        
        Args:
            onchain_analysis: On-chain analysis result
            tech_sentiment: Technical analysis sentiment
            
        Returns:
            Combined analysis with final position signal
        """
        # Map sentiments to numeric values
        sentiment_map = {
            'BULLISH': 1,
            'NEUTRAL': 0,
            'BEARISH': -1
        }
        
        onchain_score = sentiment_map.get(onchain_analysis['overall_sentiment'], 0)
        tech_score = sentiment_map.get(tech_sentiment['direction'], 0)
        
        # Weighted average (on-chain 60%, technical 40%)
        combined_score = (onchain_score * 0.6 + tech_score * 0.4)
        
        # Calculate confidence
        onchain_conf = onchain_analysis.get('confidence', 0.5)
        tech_conf = tech_sentiment.get('strength', 0.5)
        combined_confidence = (onchain_conf * 0.6 + tech_conf * 0.4)
        
        # Determine final position
        if combined_score > 0.3 and combined_confidence >= self.min_confidence:
            position = 'LONG'
            position_size = self._calculate_position_size(combined_confidence)
        elif combined_score < -0.3 and combined_confidence >= self.min_confidence:
            position = 'SHORT'
            position_size = self._calculate_position_size(combined_confidence)
        else:
            position = 'NEUTRAL'
            position_size = 0.0
        
        return {
            'position': position,
            'position_size': position_size,
            'confidence': combined_confidence,
            'combined_score': combined_score,
            'onchain_sentiment': onchain_analysis['overall_sentiment'],
            'tech_sentiment': tech_sentiment['direction']
        }
    
    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence level
        
        Args:
            confidence: Confidence level (0-1)
            
        Returns:
            Position size as fraction of max position
        """
        # Linear scaling between min and max position
        position_range = self.max_position - self.min_position
        position_size = self.min_position + (confidence * position_range)
        return round(position_size, 2)
    
    def generate_output(self, result: Dict[str, Any], 
                       stats: Dict[str, Any], args) -> Dict[str, Any]:
        """
        Generate standardized output
        
        Args:
            result: Analysis result
            stats: Market statistics
            args: Command line arguments
            
        Returns:
            Standardized output dictionary
        """
        combined = result['combined_analysis']
        onchain = result['onchain_analysis']
        
        # Build reasoning text
        reasoning_parts = [
            f"链上分析: {onchain['overall_sentiment']}",
            f"技术分析: {result['tech_sentiment']['direction']}",
            f"综合评分: {combined['combined_score']:.2f}",
            f"信号详情:"
        ]
        
        for signal in onchain['signal_details']:
            reasoning_parts.append(f"  - {signal}")
        
        reasoning = "\n".join(reasoning_parts)
        
        # Print reasoning to console
        print("\n" + "=" * 60)
        print("链上数据策略分析结果")
        print("=" * 60)
        print(reasoning)
        print("=" * 60)
        print(f"最终仓位: {combined['position']}")
        print(f"仓位大小: {combined['position_size']:.2%}")
        print(f"置信度: {combined['confidence']:.2%}")
        print("=" * 60 + "\n")
        
        # Return standardized format
        return {
            'symbol': self.symbol,
            'position': combined['position'],
            'confidence': combined['confidence'],
            'position_size': combined['position_size'],
            'current_price': stats.get('current_price', 0),
            'reasoning': reasoning,
            'analysis_type': 'onchain',
            'onchain_metrics': result['metrics'],
            'onchain_analysis': onchain,
            'tech_sentiment': result['tech_sentiment']
        }


def test_onchain_trader():
    """Test on-chain trader"""
    import argparse
    from vquant.model.vision import fetch_binance_klines
    from vquant.data.indicators import prepare_market_stats
    
    # Create test args
    args = argparse.Namespace(
        symbol='BTCUSDC',
        interval='1h',
        limit=72,
        ma_periods=[7, 25, 99]
    )
    
    # Fetch data
    df = fetch_binance_klines(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        extra_data=max(args.ma_periods) - 1
    )
    
    if df is None:
        print("Failed to fetch data")
        return
    
    # Calculate MAs
    ma_dict = {}
    for period in args.ma_periods:
        ma_dict[period] = df['Close'].rolling(window=period).mean()
    
    df_display = df.iloc[-args.limit:].copy()
    
    # Prepare stats
    stats = prepare_market_stats(df, df_display, ma_dict, args)
    
    # Create trader and run analysis
    trader = OnChainTrader(symbol="BTCUSDC", name="test_onchain")
    result, _ = trader.run(df, df_display, ma_dict, {}, stats, args)
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"
    )
    test_onchain_trader()
