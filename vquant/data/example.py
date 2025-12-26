#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: Resample 1-minute data to higher timeframes
Demonstrates how to aggregate cached 1m data to 1h, 4h, or 1d
"""

import logging
from vquant.data import get_cached_klines, resample_klines


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resample_example():
    """Example of resampling 1-minute data to different timeframes"""
    
    symbol = 'BTCUSDC'
    
    # Step 1: Get 1-minute data from cache (e.g., last 30 days)
    logger.info(f"Loading 1-minute data for {symbol}...")
    df_1m = get_cached_klines(symbol, '1m', days=30)
    
    if df_1m is None or len(df_1m) == 0:
        logger.error(f"No 1-minute data found for {symbol}")
        logger.info("Please run: python data_manager.py prefetch --symbol BTCUSDC --start-date 2023-01-01")
        return
    
    logger.info(f"Loaded {len(df_1m):,} 1-minute bars")
    logger.info(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")
    
    # Step 2: Resample to different timeframes
    timeframes = ['1h', '4h', '1d']
    
    for tf in timeframes:
        logger.info(f"\nResampling to {tf}...")
        df_resampled = resample_klines(df_1m, tf)
        
        logger.info(f"Result: {len(df_resampled)} bars")
        logger.info(f"First bar: {df_resampled.index[0]}")
        logger.info(f"Last bar: {df_resampled.index[-1]}")
        logger.info(f"Sample data:\n{df_resampled[['open', 'high', 'low', 'close', 'volume']].head()}")
    
    logger.info("\n" + "="*60)
    logger.info("Resampling completed successfully!")
    logger.info("="*60)
    logger.info("\nUsage in training:")
    logger.info("  df_1m = get_cached_klines('BTCUSDC', '1m', days=365)")
    logger.info("  df_1h = resample_klines(df_1m, '1h')  # For 1-hour strategy")
    logger.info("  df_4h = resample_klines(df_1m, '4h')  # For 4-hour strategy")
    logger.info("  df_1d = resample_klines(df_1m, '1d')  # For daily strategy")


if __name__ == "__main__":
    try:
        resample_example()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
