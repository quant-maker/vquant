#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test resample functionality
"""

import pytest
import logging
from vquant.data import get_cached_klines, resample_klines


logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_resample_1h_to_4h():
    """Test resampling 1h data to 4h"""
    df_1h = get_cached_klines('BTCUSDC', '1h', days=7)
    
    if df_1h is None or len(df_1h) == 0:
        pytest.skip("No cached 1h data available for BTCUSDC")
    
    logger.info(f"Loaded {len(df_1h)} 1h bars")
    logger.info(f"Date range: {df_1h.index.min()} to {df_1h.index.max()}")
    
    df_4h = resample_klines(df_1h, '4h')
    
    assert df_4h is not None, "Resampled dataframe should not be None"
    assert len(df_4h) > 0, "Resampled dataframe should have data"
    assert len(df_4h) < len(df_1h), "4h bars should be fewer than 1h bars"
    
    # Check OHLCV columns exist
    assert 'open' in df_4h.columns
    assert 'high' in df_4h.columns
    assert 'low' in df_4h.columns
    assert 'close' in df_4h.columns
    assert 'volume' in df_4h.columns
    
    logger.info(f"✓ Resampled to {len(df_4h)} 4h bars")


@pytest.mark.integration
def test_resample_1h_to_1d():
    """Test resampling 1h data to 1d"""
    df_1h = get_cached_klines('BTCUSDC', '1h', days=7)
    
    if df_1h is None or len(df_1h) == 0:
        pytest.skip("No cached 1h data available for BTCUSDC")
    
    df_1d = resample_klines(df_1h, '1d')
    
    assert df_1d is not None, "Resampled dataframe should not be None"
    assert len(df_1d) > 0, "Resampled dataframe should have data"
    assert len(df_1d) < len(df_1h), "Daily bars should be fewer than 1h bars"
    
    logger.info(f"✓ Resampled to {len(df_1d)} daily bars")


@pytest.mark.integration
def test_resample_preserves_ohlc_logic():
    """Test that resample preserves OHLC logic"""
    df_1h = get_cached_klines('BTCUSDC', '1h', days=3)
    
    if df_1h is None or len(df_1h) == 0:
        pytest.skip("No cached 1h data available for BTCUSDC")
    
    df_4h = resample_klines(df_1h, '4h')
    
    # Check that high >= low for all bars
    assert (df_4h['high'] >= df_4h['low']).all(), "High should be >= Low"
    
    # Check that high >= open and high >= close
    assert (df_4h['high'] >= df_4h['open']).all(), "High should be >= Open"
    assert (df_4h['high'] >= df_4h['close']).all(), "High should be >= Close"
    
    # Check that low <= open and low <= close
    assert (df_4h['low'] <= df_4h['open']).all(), "Low should be <= Open"
    assert (df_4h['low'] <= df_4h['close']).all(), "Low should be <= Close"
    
    logger.info("✓ OHLC logic preserved")


@pytest.mark.unit
def test_resample_invalid_interval():
    """Test resample with invalid interval"""
    df_1h = get_cached_klines('BTCUSDC', '1h', days=1)
    
    if df_1h is None or len(df_1h) == 0:
        pytest.skip("No cached 1h data available for BTCUSDC")
    
    with pytest.raises(ValueError):
        resample_klines(df_1h, '3h')  # Invalid interval


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

