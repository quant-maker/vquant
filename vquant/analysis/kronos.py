#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kronos Trader - Trading strategy based on Kronos official predictions
Scrapes predictions from Kronos official website and generates trading signals
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .base import BasePredictor
from ..data.kronos_scraper import KronosScraper
from ..data.json_history import JSONHistoryManager

logger = logging.getLogger(__name__)


class KronosTrader(BasePredictor):
    """
    Kronos-based Trading Strategy
    
    Features:
    1. Scrape Kronos official BTC/USDT predictions
    2. Check data freshness to avoid trading on stale data
    3. Convert predictions to standardized position signals
    4. Support configurable risk parameters
    
    Safety Features:
    - Automatic data staleness detection
    - Configurable maximum data age threshold
    - Trading block when official website stops updating
    """
    
    def __init__(self, symbol: str = "BTCUSDC", name: str = "kronos", 
                 config_path: Optional[str] = None):
        """
        Initialize Kronos trader
        
        Args:
            symbol: Trading symbol (default: BTCUSDC)
                   Note: Will be converted to BTCUSDT for Kronos scraper
            name: Strategy name
            config_path: Path to configuration file (optional)
        """
        super().__init__(symbol, name)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize scraper
        max_staleness = self.config.get('max_staleness_hours', 24)
        kronos_url = self.config.get('kronos_url', 'https://shiyu-coder.github.io/Kronos-demo/')
        fetch_timeout = self.config.get('fetch_timeout_minutes', 30)
        max_retries = self.config.get('max_retries', 3)
        
        self.scraper = KronosScraper(
            url=kronos_url,
            max_staleness_hours=max_staleness,
            fetch_timeout_minutes=fetch_timeout,
            max_retries=max_retries
        )
        
        # Convert symbol format (BTCUSDC -> BTCUSDT for Kronos)
        self.kronos_symbol = self._convert_symbol(symbol)
        
        # Initialize JSON history manager for fallback
        use_json_fallback = self.config.get('use_json_fallback', True)
        if use_json_fallback:
            json_history_dir = self.config.get('json_history_dir', 'charts')
            json_max_age = self.config.get('json_max_age_minutes', fetch_timeout)
            
            self.json_history = JSONHistoryManager(
                strategy_name=name,
                history_dir=json_history_dir,
                max_age_minutes=json_max_age
            )
            logger.info(f"JSON fallback enabled: {json_history_dir} (max age: {json_max_age} min)")
        else:
            self.json_history = None
            logger.info("JSON fallback disabled")
        
        logger.info(f"Kronos Trader initialized: {symbol} -> {self.kronos_symbol}")
        logger.info(f"Max data staleness: {max_staleness} hours")
        logger.info(f"Fetch timeout: {fetch_timeout} minutes")
        logger.info(f"Max retries per fetch: {max_retries}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'max_staleness_hours': 72,  # Maximum allowed data age in hours
            'fetch_timeout_minutes': 60,  # Maximum minutes to keep trying before giving up (Kronos updates hourly)
            'max_retries': 3,  # Maximum number of retry attempts per fetch
            'use_json_fallback': True,  # Use JSON history for fallback
            'json_history_dir': 'charts',  # Directory for JSON history
            'json_max_age_minutes': 700,  # Maximum age of JSON to use for fallback (with buffer for edge cases)
            'kronos_url': 'https://shiyu-coder.github.io/Kronos-demo/',
            'confidence_threshold': 0.5,  # Minimum confidence for trading
            'position_multiplier': 1.0,   # Position size multiplier (0-1)
            'enable_safety_check': True,  # Enable data freshness check
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def _convert_symbol(self, symbol: str) -> str:
        """
        Convert symbol format for Kronos
        
        Binance uses BTCUSDC/BTCUSDT, Kronos might use BTCUSDT
        
        Args:
            symbol: Original symbol (e.g., BTCUSDC)
            
        Returns:
            Converted symbol (e.g., BTCUSDT)
        """
        # Simple conversion: USDC -> USDT
        if symbol.endswith('USDC'):
            return symbol.replace('USDC', 'USDT')
        return symbol
    
    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Kronos trader doesn't need chart generation
        Data preparation is done in analyze() by scraping Kronos website
        
        Returns:
            (None, None) as no chart is needed
        """
        logger.info("Kronos trader: skipping chart generation")
        return None, None
    
    def analyze(self, stats: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze market using Kronos predictions
        
        Args:
            stats: Market statistics dictionary
            **kwargs: Additional parameters
            
        Returns:
            Analysis result with position recommendation
        """
        logger.info("Fetching Kronos prediction...")
        
        # Fetch prediction from Kronos
        prediction = self.scraper.fetch_prediction(self.kronos_symbol)
        
        if not prediction:
            # Fetch failed - try JSON history fallback if enabled
            if self.json_history:
                logger.warning("⚠️ Fetch failed - attempting JSON history fallback...")
                prediction = self.json_history.get_fallback_prediction()
            
            if not prediction:
                # Total failure - no data available
                logger.error("❌ Failed to fetch Kronos prediction - no data available (fetch failed + no JSON history)")
                return {
                    'position': 0.0,
                    'confidence': 'low',
                    'reasoning': 'Failed to fetch Kronos prediction - timeout exceeded and no JSON history available. Clearing position for safety.',
                    'is_safe': False,
                    'prediction': None,
                    'fetch_status': 'timeout_exceeded'
                }
        
        # Check fetch status
        fetch_status = prediction.get('fetch_status', 'unknown')
        
        if fetch_status == 'temporary_failure':
            # Temporary failure - using cached data to maintain position
            consecutive_failures = prediction.get('consecutive_failures', 0)
            time_since_failure = prediction.get('time_since_first_failure', 0)
            
            logger.warning(
                f"⚠️ Using scraper cache due to temporary fetch failure. "
                f"Failures: {consecutive_failures}, Time: {time_since_failure:.1f} min"
            )
        elif fetch_status == 'json_fallback':
            # Using JSON history fallback
            json_source = prediction.get('json_source', 'unknown')
            json_age = prediction.get('json_age_minutes', 0)
            
            logger.warning(
                f"⚠️ Using JSON history fallback to maintain position. "
                f"Source: {json_source}, Age: {json_age:.1f} min"
            )
        
        # Check if data is safe to trade
        is_safe = True
        if self.config.get('enable_safety_check', True):
            is_safe = self.scraper.is_safe_to_trade(prediction)
        
        # Calculate position signal
        position = self._calculate_position(prediction, stats)
        
        # Determine confidence level
        confidence = self._calculate_confidence(prediction, is_safe)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(prediction, stats, is_safe)
        
        return {
            'position': position,
            'confidence': confidence,
            'reasoning': reasoning,
            'is_safe': is_safe,
            'prediction': prediction,
            'fetch_status': fetch_status
        }
    
    def _calculate_position(self, prediction: Dict[str, Any], stats: Dict[str, Any]) -> float:
        """
        Calculate position based on Kronos prediction
        
        Args:
            prediction: Kronos prediction data
            stats: Market statistics
            
        Returns:
            Position from -1.0 (full short) to 1.0 (full long)
        """
        # Get base signal from prediction
        signal = self.scraper.get_position_signal(prediction)
        
        # Apply confidence threshold
        confidence = prediction.get('confidence', 0.5)
        min_confidence = self.config.get('confidence_threshold', 0.5)
        
        if confidence < min_confidence:
            logger.info(f"Confidence {confidence:.2%} below threshold {min_confidence:.2%}, reducing position")
            signal *= (confidence / min_confidence)
        
        # Apply position multiplier (risk control)
        multiplier = self.config.get('position_multiplier', 1.0)
        signal *= multiplier
        
        # Block trading if data is stale
        if prediction.get('is_stale', True) and self.config.get('enable_safety_check', True):
            logger.warning("⚠️ Data is stale - blocking trading signal")
            return 0.0
        
        return round(signal, 2)
    
    def _calculate_confidence(self, prediction: Dict[str, Any], is_safe: bool) -> str:
        """
        Calculate confidence level
        
        Args:
            prediction: Kronos prediction data
            is_safe: Whether data is safe to trade
            
        Returns:
            Confidence level: 'low', 'medium', or 'high'
        """
        if not is_safe:
            return 'low'
        
        confidence = prediction.get('confidence', 0.5)
        
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _generate_reasoning(self, prediction: Dict[str, Any], stats: Dict[str, Any], is_safe: bool) -> str:
        """
        Generate human-readable reasoning
        
        Args:
            prediction: Kronos prediction data
            stats: Market statistics
            is_safe: Whether data is safe to trade
            
        Returns:
            Reasoning string
        """
        parts = []
        
        # Check fetch status first
        fetch_status = prediction.get('fetch_status', 'unknown')
        if fetch_status == 'temporary_failure':
            consecutive_failures = prediction.get('consecutive_failures', 0)
            time_since_failure = prediction.get('time_since_first_failure', 0)
            parts.append(
                f"⚠️ WARNING: Using scraper cache due to fetch failures. "
                f"Consecutive failures: {consecutive_failures}, "
                f"Time since first failure: {time_since_failure:.1f} minutes. "
                f"Maintaining current position."
            )
        elif fetch_status == 'json_fallback':
            json_source = prediction.get('json_source', 'unknown')
            json_age = prediction.get('json_age_minutes', 0)
            parts.append(
                f"⚠️ WARNING: Using JSON history fallback. "
                f"Source: {json_source}, "
                f"Age: {json_age:.1f} minutes. "
                f"Maintaining position from historical data."
            )
        
        # Data freshness warning
        if not is_safe:
            staleness = prediction.get('staleness_hours', 'unknown')
            parts.append(f"⚠️ WARNING: Kronos data is {staleness} hours old (stale)")
            parts.append("Official website may have stopped updating - trading NOT recommended")
            return ". ".join(parts) + "."
        
        # Prediction summary
        trend = prediction.get('trend', 'unknown')
        confidence = prediction.get('confidence', 0)
        predicted_price = prediction.get('predicted_price')
        current_price = prediction.get('current_price') or stats.get('current_price')
        
        if trend == 'up':
            parts.append(f"Kronos predicts BULLISH trend (confidence: {confidence:.1%})")
        elif trend == 'down':
            parts.append(f"Kronos predicts BEARISH trend (confidence: {confidence:.1%})")
        else:
            parts.append(f"Kronos predicts NEUTRAL trend (confidence: {confidence:.1%})")
        
        # Price targets
        if predicted_price and current_price:
            change_pct = (predicted_price - current_price) / current_price * 100
            parts.append(f"Target: ${predicted_price:.2f} (current: ${current_price:.2f}, {change_pct:+.2f}%)")
        
        # Data freshness
        update_time = prediction.get('update_time')
        if update_time:
            staleness_hours = prediction.get('staleness_hours', 0)
            parts.append(f"Data updated {staleness_hours:.1f} hours ago")
        
        # Confidence assessment
        if confidence >= 0.8:
            parts.append("High confidence prediction")
        elif confidence >= 0.6:
            parts.append("Moderate confidence prediction")
        else:
            parts.append("Low confidence prediction - consider reducing position size")
        
        return ". ".join(parts) + "."
    
    def generate_output(self, result: Dict[str, Any], stats: Dict[str, Any], args) -> Dict[str, Any]:
        """
        Generate standardized output
        
        Args:
            result: Analysis result from analyze()
            stats: Market statistics
            args: Command line arguments
            
        Returns:
            Standardized output dictionary
        """
        prediction = result.get('prediction') or {}
        
        return {
            'symbol': self.symbol,
            'position': result['position'],
            'confidence': result['confidence'],
            'current_price': stats.get('current_price'),
            'reasoning': result['reasoning'],
            'analysis_type': 'kronos',
            'is_safe': result.get('is_safe', False),
            'fetch_status': result.get('fetch_status', 'unknown'),
            'kronos_data': {
                'trend': prediction.get('trend') if prediction else None,
                'predicted_price': prediction.get('predicted_price') if prediction else None,
                'kronos_confidence': prediction.get('confidence') if prediction else None,
                'update_time': prediction['update_time'].isoformat() if prediction and prediction.get('update_time') else None,
                'staleness_hours': prediction.get('staleness_hours') if prediction else None,
                'is_stale': prediction.get('is_stale', True) if prediction else True,
                'consecutive_failures': prediction.get('consecutive_failures', 0) if prediction else 0,
                'time_since_first_failure': prediction.get('time_since_first_failure', 0) if prediction else 0,
            }
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
    )
    
    print("\n=== Testing Kronos Trader ===\n")
    
    # Create trader
    trader = KronosTrader(symbol="BTCUSDC", name="test")
    
    # Mock stats
    mock_stats = {
        'current_price': 95000.0,
        'current_ma7': 94500.0,
        'current_rsi': 55.0,
    }
    
    # Run analysis
    result = trader.analyze(mock_stats)
    
    print(f"Position: {result['position']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Is Safe: {result.get('is_safe', False)}")
    print(f"\nReasoning:\n{result['reasoning']}")
    
    # Generate final output
    output = trader.generate_output(result, mock_stats, None)
    print(f"\n=== Final Output ===")
    print(json.dumps(output, indent=2, default=str))
