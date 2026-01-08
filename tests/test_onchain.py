#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Tests for OnChain strategy
"""

import pytest
import logging
from vquant.data.onchain_fetcher import OnChainFetcher

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_onchain_fetcher():
    """Test on-chain data fetcher"""
    fetcher = OnChainFetcher()
    
    # Test open interest
    oi = fetcher.fetch_open_interest("BTCUSDT")
    assert oi is not None, "Should fetch open interest"
    assert 'openInterest' in oi, "Should have openInterest field"
    
    # Test OI history
    oi_history = fetcher.fetch_open_interest_history("BTCUSDT", period="1h", limit=10)
    assert oi_history is not None, "Should fetch OI history"
    assert len(oi_history) > 0, "Should have history data"
    
    # Test long/short ratio
    ls_ratio = fetcher.fetch_long_short_ratio("BTCUSDT", period="1h", limit=10)
    assert ls_ratio is not None, "Should fetch long/short ratio"
    assert len(ls_ratio) > 0, "Should have ratio data"
    
    # Test taker volume
    taker_vol = fetcher.fetch_taker_buy_sell_volume("BTCUSDT", period="1h", limit=10)
    assert taker_vol is not None, "Should fetch taker volume"
    assert len(taker_vol) > 0, "Should have volume data"


@pytest.mark.integration
def test_onchain_metrics_all():
    """Test fetching all metrics"""
    fetcher = OnChainFetcher()
    
    metrics = fetcher.fetch_all_metrics("BTCUSDT", period="1h", history_limit=24)
    
    assert metrics is not None, "Should fetch metrics"
    assert metrics['symbol'] == "BTCUSDT", "Should have correct symbol"
    assert metrics['open_interest'] is not None, "Should have open interest"
    assert metrics['open_interest_history'] is not None, "Should have OI history"
    assert metrics['long_short_ratio'] is not None, "Should have L/S ratio"
    assert metrics['taker_volume'] is not None, "Should have taker volume"


@pytest.mark.integration
def test_onchain_analysis():
    """Test on-chain metrics analysis"""
    fetcher = OnChainFetcher()
    
    metrics = fetcher.fetch_all_metrics("BTCUSDT", period="1h", history_limit=24)
    analysis = fetcher.analyze_metrics(metrics)
    
    assert analysis is not None, "Should have analysis result"
    assert 'overall_sentiment' in analysis, "Should have sentiment"
    assert analysis['overall_sentiment'] in ['BULLISH', 'BEARISH', 'NEUTRAL'], \
        "Sentiment should be valid"
    assert 'confidence' in analysis, "Should have confidence"
    assert 0 <= analysis['confidence'] <= 1, "Confidence should be 0-1"
    assert 'signal_details' in analysis, "Should have signal details"


@pytest.mark.unit
def test_onchain_trader_init():
    """Test OnChain trader initialization"""
    from vquant.analysis.onchain import OnChainTrader
    
    trader = OnChainTrader(symbol="BTCUSDC", name="test")
    
    assert trader.symbol == "BTCUSDC", "Should store symbol"
    assert trader.name == "test", "Should store name"
    assert trader.futures_symbol == "BTCUSDT", "Should convert symbol to BTCUSDT"
    assert trader.fetcher is not None, "Should have fetcher"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s"
    )
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
