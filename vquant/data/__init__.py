#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Data module for fetching and caching market data
"""

from .fetcher import (
    fetch_klines_multiple_batches,
    prepare_training_data,
    get_cached_klines,
    clear_cache,
    prefetch_all_data,
    resample_klines
)
from .indicators import (
    prepare_market_stats,
    add_funding_rate_to_stats,
)

__all__ = [
    'fetch_klines_multiple_batches',
    'prepare_training_data',
    'get_cached_klines',
    'clear_cache',
    'prefetch_all_data',
    'resample_klines',
    'prepare_market_stats',
    'add_funding_rate_to_stats',
]
