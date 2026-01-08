# AGENTS.md - Coding Agent Guide for VQuant

This file contains essential information for AI coding agents working on this quantitative trading system.

## Project Overview

**VQuant** is a Python-based quantitative trading system that uses 1-minute market data as the foundation and dynamically aggregates to other timeframes (1h, 4h, 1d). It supports multiple trading strategies (Quant, Kelly, Martingale, Kalshi, Kronos) with AI-powered analysis.

**Key Technologies:**
- Python 3.11+
- pandas for data manipulation
- SQLite for local data caching
- pytest for testing
- scikit-learn for ML models
- Multiple AI service APIs (Qwen, OpenAI, DeepSeek, GitHub Copilot)

## Build/Lint/Test Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run fast tests only (exclude slow API tests)
pytest -m "not slow"

# Run specific test categories
pytest -m unit              # Fast unit tests only
pytest -m integration       # Integration tests (require cached data)

# Run a single test file
pytest tests/test_fetcher.py -v

# Run a specific test function
pytest tests/test_fetcher.py::test_fetch_klines_with_cache -v

# Run with coverage report
pytest --cov=vquant --cov-report=html

# Show print statements during tests
pytest -s

# Verbose output with detailed info
pytest -v -s
```

**Test Markers:**
- `@pytest.mark.unit` - Fast unit tests (no network/database)
- `@pytest.mark.integration` - Integration tests (require cached data)
- `@pytest.mark.slow` - Slow tests (involve API requests)

### Data Management

```bash
# List all cached data
python -m vquant.data.manager list

# Prefetch historical data (recommended before training)
python -m vquant.data.manager prefetch --all --start-date 2023-01-01
python -m vquant.data.manager prefetch --symbol BTCUSDC --start-date 2023-01-01

# View specific cached data
python -m vquant.data.manager view --symbol BTCUSDC --interval 1m --days 7
```

### Model Training

```bash
# Train models using 1m data (dynamically aggregated)
python -m vquant.model.train --symbol BTCUSDC --interval 1h --days 365
python -m vquant.model.train --symbol ETHUSDC --interval 4h --days 365
python -m vquant.model.train --symbol BNBUSDC --interval 1d --days 730
```

### Running Strategies

```bash
# Get help on all options
python main.py --help

# Run different predictors
python main.py --predictor quant --name my_quant
python main.py --predictor kelly --name my_kelly
python main.py --predictor kronos --name my_kronos

# With custom parameters
python main.py --symbol BTCUSDC --interval 1h --service qwen --verbose
python main.py --symbol ETHUSDC --interval 4h --service copilot --model gpt-4o --trade
```

## Code Style Guidelines

### File Headers
All Python files should start with:
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Module Description - Brief description of the module
Detailed explanation if necessary
"""
```

### Imports
Follow this import order (separate with blank lines):
1. Standard library imports
2. Third-party imports
3. Local application imports

Example:
```python
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from pathlib import Path

from vquant.data.fetcher import DataCache
from vquant.analysis.base import BasePredictor
```

### Type Hints
- Use type hints for all function parameters and return values
- Import types from `typing` module
- Common types: `Optional[T]`, `Dict[K, V]`, `List[T]`, `Tuple[...]`, `Any`

Example:
```python
def get_cached_data(
    self,
    symbol: str,
    interval: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """Retrieve cached data from database"""
    pass
```

### Docstrings
Use Google-style docstrings for all classes and functions:

```python
def prepare_stats(self, df: pd.DataFrame, args) -> Dict[str, Any]:
    """
    Prepare comprehensive market statistics
    
    Args:
        df: Full dataframe with OHLCV data
        args: Command line arguments
        
    Returns:
        Statistics dictionary containing all market data and indicators
    """
    pass
```

### Naming Conventions
- **Files**: lowercase with underscores (`data_fetcher.py`, `position_manager.py`)
- **Classes**: PascalCase (`DataCache`, `BasePredictor`, `PositionManager`)
- **Functions/Methods**: lowercase with underscores (`get_cached_data`, `prepare_stats`)
- **Constants**: UPPERCASE with underscores (`DEFAULT_LIMIT`, `API_ENDPOINT`)
- **Private methods**: prefix with single underscore (`_init_db`, `_load_state`)

### Logging
Use the standard logging module (NOT print statements):

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### Error Handling
- Use try-except blocks for external operations (API calls, file I/O, database)
- Log errors with context information
- Return `None` or raise exceptions depending on context

Example:
```python
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
except requests.RequestException as e:
    logger.error(f"Failed to fetch data from {url}: {e}")
    return None
```

### Abstract Classes
Use ABC for base classes that should be inherited:

```python
from abc import ABC, abstractmethod

class BasePredictor(ABC):
    @abstractmethod
    def analyze(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Core analysis logic - must be implemented by subclasses"""
        pass
```

### Database Operations
- Always use context managers or explicit close() for connections
- Use parameterized queries to prevent SQL injection
- Create indexes for frequently queried columns

Example:
```python
conn = sqlite3.connect(self.db_path)
try:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM klines WHERE symbol = ? AND interval = ?", 
                   [symbol, interval])
    result = cursor.fetchall()
finally:
    conn.close()
```

### Configuration
- Store configuration in JSON files under `config/` directory
- Use `.env` files for sensitive data (API keys)
- Load environment variables with `python-dotenv`

## Project Structure

```
vquant/
├── analysis/      # Trading strategies (quant, kelly, martin, kalshi, kronos)
├── data/          # Data fetching & management (fetcher, indicators, manager)
├── model/         # ML models & training (train, calibrator, vision)
└── executor/      # Trade execution (trader, position)
```

## Testing Guidelines

1. **Mark tests appropriately**: Use `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.slow`
2. **Test data**: Integration tests should use cached data (not live API calls)
3. **Assertions**: Use descriptive assertion messages
4. **Fixtures**: Use pytest fixtures for common setup
5. **Test structure**: Arrange-Act-Assert pattern

Example:
```python
@pytest.mark.integration
def test_fetch_klines_with_cache():
    """Test fetching klines with cache enabled"""
    # Arrange
    symbol = 'BTCUSDC'
    interval = '1h'
    
    # Act
    df = fetch_klines_multiple_batches(symbol, interval, days=1, use_cache=True)
    
    # Assert
    assert df is not None, "First fetch should return data"
    assert len(df) > 0, "Should have data"
    assert 'close' in df.columns, "Should have OHLC columns"
```

## Common Patterns

### 1-Minute Data Architecture
- Always fetch 1m data and resample to target interval
- Use `fetch_klines_multiple_batches()` with appropriate `days` parameter
- Resample with `resample_to_interval()` for analysis

### Position Management
- Use `PositionManager` class for tracking positions
- Strategy name must be unique for each strategy instance
- Check pending orders before placing new ones

### AI Service Integration
- Support multiple AI providers (configured via command line)
- Pass market statistics as context
- Handle API failures gracefully with retries

---

**Last Updated:** 2026-01-08
**For Questions:** See documentation in `docs/` directory or README.md
