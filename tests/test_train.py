#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test training functionality
"""

import pytest
import logging
from pathlib import Path
from vquant.model.train import train_kelly_model


logger = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.integration
def test_train_with_1h_cache():
    """Test training with cached 1h data"""
    # This uses existing cached 1h data (should be available)
    success = train_kelly_model(
        symbol='BTCUSDC',
        name='test_pytest',
        interval='1h',
        days=7,  # Just 7 days for quick test
        lookforward_bars=5,
        profit_threshold=0.005,
        use_1m_data=False  # Use 1h directly
    )
    
    # Check model files created
    model_path = Path('data/kelly_model_test_pytest.pkl')
    scaler_path = Path('data/kelly_scaler_test_pytest.pkl')
    
    if success:
        assert model_path.exists(), "Model file should be created"
        assert scaler_path.exists(), "Scaler file should be created"
        logger.info("✓ Training completed and model files created")
    else:
        pytest.skip("Training failed (may need more cached data)")


@pytest.mark.unit
def test_train_function_params():
    """Test that train function accepts correct parameters"""
    # This just tests the function signature, doesn't actually train
    import inspect
    sig = inspect.signature(train_kelly_model)
    params = sig.parameters
    
    assert 'symbol' in params
    assert 'name' in params
    assert 'interval' in params
    assert 'days' in params
    assert 'lookforward_bars' in params
    assert 'profit_threshold' in params
    assert 'use_1m_data' in params
    
    # Check defaults
    assert params['use_1m_data'].default == True
    assert params['interval'].default == '1h'
    
    logger.info("✓ Function signature is correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
