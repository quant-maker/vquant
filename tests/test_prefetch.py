#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test prefetch functionality
"""

import pytest
import logging
from vquant.data import prefetch_all_data


logger = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.integration
def test_prefetch_single_symbol():
    """Test prefetching data for a single symbol"""
    stats = prefetch_all_data(
        symbols=['BTCUSDC'],
        start_date='2025-12-20',  # Just last few days
        request_delay=0.3
    )
    
    assert stats is not None, "Stats should not be None"
    assert 'total' in stats
    assert 'success' in stats
    assert 'failed' in stats
    assert stats['total'] == 1, "Should have 1 task"
    
    logger.info(f"✓ Prefetch completed: {stats['success']}/{stats['total']} successful")


@pytest.mark.slow
@pytest.mark.integration
def test_prefetch_multiple_symbols():
    """Test prefetching data for multiple symbols"""
    stats = prefetch_all_data(
        symbols=['BTCUSDC', 'ETHUSDC'],
        start_date='2025-12-25',  # Just 1 day
        request_delay=0.3
    )
    
    assert stats is not None
    assert stats['total'] == 2, "Should have 2 tasks"
    assert stats['success'] + stats['failed'] == stats['total']
    
    logger.info(f"✓ Prefetch completed: {stats['success']}/{stats['total']} successful")


@pytest.mark.unit
def test_prefetch_returns_stats():
    """Test that prefetch returns proper stats structure"""
    # This will use cache if available, should be fast
    stats = prefetch_all_data(
        symbols=['BTCUSDC'],
        start_date='2025-12-26',  # Today only
        request_delay=0.1
    )
    
    # Verify stats structure
    assert isinstance(stats, dict)
    assert 'total' in stats
    assert 'success' in stats
    assert 'failed' in stats
    assert 'tasks' in stats
    assert 'start_date' in stats
    assert 'interval' in stats
    
    assert stats['interval'] == '1m', "Should only fetch 1m data"
    assert stats['start_date'] == '2025-12-26'
    
    logger.info("✓ Stats structure is correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

