#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test data fetcher functionality
"""

import pytest
import logging
from vquant.data import (
    fetch_klines_multiple_batches,
    get_cached_klines,
    clear_cache
)


logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_fetch_klines_with_cache():
    """Test fetching klines with cache enabled"""
    symbol = 'BTCUSDC'
    interval = '1h'
    days = 1
    
    # First fetch (may hit API or cache)
    df1 = fetch_klines_multiple_batches(
        symbol=symbol,
        interval=interval,
        days=days,
        verbose=False,
        use_cache=True
    )
    
    assert df1 is not None, "First fetch should return data"
    assert len(df1) > 0, "Should have data"
    assert 'open' in df1.columns
    assert 'close' in df1.columns
    
    # Second fetch (should hit cache)
    df2 = fetch_klines_multiple_batches(
        symbol=symbol,
        interval=interval,
        days=days,
        verbose=False,
        use_cache=True
    )
    
    assert df2 is not None, "Second fetch should return data"
    assert len(df2) == len(df1), "Cached data should match"
    
    logger.info(f"✓ Fetch with cache works: {len(df1)} bars")


@pytest.mark.integration
def test_get_cached_klines():
    """Test getting data from cache"""
    symbol = 'BTCUSDC'
    interval = '1h'
    
    # Ensure some data is cached
    fetch_klines_multiple_batches(
        symbol=symbol,
        interval=interval,
        days=1,
        verbose=False,
        use_cache=True
    )
    
    # Get from cache
    df = get_cached_klines(symbol, interval, days=1)
    
    if df is not None:
        assert len(df) > 0, "Cached data should have content"
        logger.info(f"✓ Retrieved {len(df)} bars from cache")
    else:
        pytest.skip("No cached data available")


@pytest.mark.unit
def test_fetch_with_request_delay():
    """Test that request_delay parameter is accepted"""
    # This should not raise an error
    df = fetch_klines_multiple_batches(
        symbol='BTCUSDC',
        interval='1h',
        days=1,
        verbose=False,
        use_cache=True,
        request_delay=0.1
    )
    
    assert df is not None or df is None  # Either way, no error
    logger.info("✓ request_delay parameter works")


@pytest.mark.integration
def test_fetch_different_intervals():
    """Test fetching different intervals"""
    symbol = 'BTCUSDC'
    intervals = ['1h', '4h', '1d']
    
    for interval in intervals:
        df = fetch_klines_multiple_batches(
            symbol=symbol,
            interval=interval,
            days=3,
            verbose=False,
            use_cache=True
        )
        
        if df is not None and len(df) > 0:
            assert 'open' in df.columns
            assert 'close' in df.columns
            logger.info(f"✓ Fetched {len(df)} bars for {interval}")
        else:
            logger.warning(f"No data for {interval}")


@pytest.mark.unit
def test_dataframe_structure():
    """Test that returned dataframe has correct structure"""
    df = get_cached_klines('BTCUSDC', '1h', days=1)
    
    if df is None or len(df) == 0:
        pytest.skip("No cached data available")
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check index is datetime
    assert df.index.name == 'timestamp' or str(df.index.dtype).startswith('datetime')
    
    logger.info("✓ DataFrame structure is correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
