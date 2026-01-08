#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kronos Quant Trader - Trading strategy with quantized position sizing
Based on Kronos predictions but with non-linear position quantization
to reduce frequent small position changes
"""

import logging
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .kronos import KronosTrader

logger = logging.getLogger(__name__)


class KronosQuantTrader(KronosTrader):
    """
    Kronos-based Trading Strategy with Quantized Position Sizing
    
    Features:
    1. Inherits all Kronos trader functionality
    2. Non-linear position quantization to reduce noise
    3. Confidence-weighted position changes
    4. Adaptive thresholds based on confidence level
    
    Position Quantization Strategy:
    - Low confidence (<0.6): Dampened positions (reduce risk)
    - Medium confidence (0.6-0.8): Granular control (fine-tuned positions)
    - High confidence (>0.8): Amplified positions (maximize opportunity)
    
    Example mappings:
    - Confidence 0.50 -> Position 50% (dampened)
    - Confidence 0.70 -> Position 75% (granular)
    - Confidence 0.85 -> Position 100% (amplified to full position)
    """
    
    def __init__(self, symbol: str = "BTCUSDC", name: str = "kronos_quant", 
                 config_path: Optional[str] = None):
        """
        Initialize Kronos Quant trader
        
        Args:
            symbol: Trading symbol (default: BTCUSDC)
            name: Strategy name
            config_path: Path to configuration file (optional)
        """
        # Initialize parent
        super().__init__(symbol, name, config_path)
        
        # Load quantization parameters
        self.quant_config = self._load_quant_config()
        
        # Track last executed position for change detection
        self.last_position = 0.0
        self.last_confidence = 0.5
        
        logger.info(f"Kronos Quant Trader initialized with quantization:")
        logger.info(f"  Position levels: {self.quant_config['position_levels']}")
        logger.info(f"  Min change threshold: {self.quant_config['min_change_threshold']:.1%}")
        logger.info(f"  Sensitivity curve: {self.quant_config['sensitivity_curve']}")
    
    def _load_quant_config(self) -> Dict[str, Any]:
        """
        Load quantization configuration from parent config or defaults
        
        Returns:
            Quantization configuration dictionary
        """
        # Check if quant config exists in parent config
        if 'quantization' in self.config:
            quant_config = self.config['quantization']
            logger.info("Loaded quantization config from strategy config")
        else:
            # Use defaults
            quant_config = {
                'position_levels': [0.0, 0.25, 0.5, 0.75, 1.0],  # Discrete position levels
                'min_change_threshold': 0.1,  # Minimum 10% change to trigger adjustment
                'sensitivity_curve': 'sigmoid',  # 'sigmoid', 'quadratic', or 'cubic'
                'sensitivity_power': 2.0,  # Controls non-linearity strength
                'high_confidence_zone': [0.6, 0.8],  # Region with higher sensitivity
                'high_confidence_multiplier': 1.5,  # Sensitivity boost in high confidence zone
            }
            logger.info("Using default quantization config")
        
        return quant_config
    
    def _calculate_position(self, prediction: Dict[str, Any], stats: Dict[str, Any]) -> float:
        """
        Calculate quantized position based on Kronos prediction
        
        Args:
            prediction: Kronos prediction data
            stats: Market statistics
            
        Returns:
            Quantized position from -1.0 (full short) to 1.0 (full long)
        """
        # Get raw position from parent class
        raw_position = super()._calculate_position(prediction, stats)
        
        # If position is zero (blocked), return immediately
        if abs(raw_position) < 0.01:
            return 0.0
        
        # Get confidence
        confidence = prediction.get('confidence', 0.5)
        
        # Apply non-linear quantization
        quantized_position = self._quantize_position(raw_position, confidence)
        
        # Check if change is significant enough
        if not self._should_update_position(quantized_position, confidence):
            logger.info(f"Position change too small, keeping last position: {self.last_position:.2f}")
            return self.last_position
        
        # Update tracking variables
        self.last_position = quantized_position
        self.last_confidence = confidence
        
        return quantized_position
    
    def _quantize_position(self, raw_position: float, confidence: float) -> float:
        """
        Apply non-linear quantization to position
        
        Args:
            raw_position: Raw position signal
            confidence: Confidence level (0-1)
            
        Returns:
            Quantized position
        """
        # Get sign and magnitude
        sign = 1.0 if raw_position >= 0 else -1.0
        magnitude = abs(raw_position)
        
        # Apply non-linear sensitivity based on confidence
        adjusted_magnitude = self._apply_sensitivity_curve(magnitude, confidence)
        
        # Map to discrete position levels
        position_levels = self.quant_config['position_levels']
        quantized_magnitude = self._map_to_levels(adjusted_magnitude, position_levels)
        
        return sign * quantized_magnitude
    
    def _apply_sensitivity_curve(self, magnitude: float, confidence: float) -> float:
        """
        Apply non-linear sensitivity curve based on confidence level
        
        Strategy:
        - Low confidence (<0.6): Dampened (reduce position)
        - Medium confidence (0.6-0.8): Granular control (fine-tuned)
        - High confidence (>0.8): Amplified (increase position)
        
        Args:
            magnitude: Position magnitude (0-1)
            confidence: Confidence level (0-1)
            
        Returns:
            Adjusted magnitude
        """
        curve_type = self.quant_config['sensitivity_curve']
        power = self.quant_config['sensitivity_power']
        
        # Determine confidence zone
        high_conf_zone = self.quant_config['high_confidence_zone']
        
        if confidence < high_conf_zone[0]:
            # Low confidence zone - dampen position
            # Use power > 1 to reduce magnitude
            if curve_type == 'sigmoid':
                adjusted = self._sigmoid_map(magnitude, steepness=2.0)
            elif curve_type == 'quadratic':
                adjusted = math.pow(magnitude, power * 1.2)  # Stronger dampening
            else:  # cubic
                adjusted = math.pow(magnitude, power * 1.5)
                
        elif high_conf_zone[0] <= confidence <= high_conf_zone[1]:
            # Medium confidence zone - granular control
            # Use moderate mapping for fine-tuned positions
            multiplier = self.quant_config['high_confidence_multiplier']
            
            if curve_type == 'sigmoid':
                adjusted = self._sigmoid_map(magnitude, steepness=4.0 * multiplier)
            elif curve_type == 'quadratic':
                adjusted = math.pow(magnitude, 1.0 / (power / multiplier))
            else:  # cubic
                adjusted = math.pow(magnitude, 1.0 / (power * 1.5 / multiplier))
                
        else:
            # High confidence zone (>0.8) - amplify position
            # Use power < 1 to boost magnitude toward higher levels
            if curve_type == 'sigmoid':
                adjusted = self._sigmoid_map(magnitude, steepness=6.0)  # Steeper = more aggressive
            elif curve_type == 'quadratic':
                adjusted = math.pow(magnitude, 1.0 / (power * 1.5))  # Strong amplification
            else:  # cubic
                adjusted = math.pow(magnitude, 1.0 / (power * 2.0))  # Very strong amplification
            
            # Additional boost for very high confidence
            if confidence >= 0.85:
                adjusted = min(1.0, adjusted * 1.1)  # 10% boost, capped at 1.0
        
        return adjusted
    
    def _sigmoid_map(self, x: float, steepness: float = 4.0) -> float:
        """
        Sigmoid mapping function for smooth non-linear transitions
        
        Args:
            x: Input value (0-1)
            steepness: Controls curve steepness (higher = steeper)
            
        Returns:
            Mapped value (0-1)
        """
        # Sigmoid: 1 / (1 + exp(-k*(x-0.5)))
        # Normalized to [0, 1] range
        k = steepness
        sigmoid = 1.0 / (1.0 + math.exp(-k * (x - 0.5)))
        
        # Normalize to [0, 1]
        sigmoid_min = 1.0 / (1.0 + math.exp(k * 0.5))
        sigmoid_max = 1.0 / (1.0 + math.exp(-k * 0.5))
        normalized = (sigmoid - sigmoid_min) / (sigmoid_max - sigmoid_min)
        
        return normalized
    
    def _map_to_levels(self, magnitude: float, levels: list) -> float:
        """
        Map continuous magnitude to discrete position levels
        
        Args:
            magnitude: Continuous magnitude (0-1)
            levels: List of discrete position levels
            
        Returns:
            Quantized position level
        """
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Find nearest level
        min_distance = float('inf')
        nearest_level = sorted_levels[0]
        
        for level in sorted_levels:
            distance = abs(magnitude - level)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
        
        return nearest_level
    
    def _should_update_position(self, new_position: float, confidence: float) -> bool:
        """
        Determine if position change is significant enough to execute
        
        Args:
            new_position: Proposed new position
            confidence: Current confidence level
            
        Returns:
            True if position should be updated
        """
        # Always update if current position is zero (entering market)
        if abs(self.last_position) < 0.01:
            return abs(new_position) > 0.01
        
        # Calculate relative change
        position_change = abs(new_position - self.last_position)
        relative_change = position_change / max(abs(self.last_position), 0.01)
        
        # Get threshold
        min_threshold = self.quant_config['min_change_threshold']
        
        # Adjust threshold based on confidence change
        confidence_change = abs(confidence - self.last_confidence)
        
        # If confidence changed significantly, lower threshold
        if confidence_change > 0.15:  # >15% confidence change
            adjusted_threshold = min_threshold * 0.5
            logger.info(f"Large confidence change ({confidence_change:.1%}), lowering threshold to {adjusted_threshold:.1%}")
        else:
            adjusted_threshold = min_threshold
        
        should_update = relative_change >= adjusted_threshold
        
        if not should_update:
            logger.debug(
                f"Position change {relative_change:.1%} below threshold {adjusted_threshold:.1%} "
                f"({self.last_position:.2f} -> {new_position:.2f})"
            )
        
        return should_update
    
    def _generate_reasoning(self, prediction: Dict[str, Any], stats: Dict[str, Any], is_safe: bool) -> str:
        """
        Generate human-readable reasoning with quantization info
        
        Args:
            prediction: Kronos prediction data
            stats: Market statistics
            is_safe: Whether data is safe to trade
            
        Returns:
            Reasoning string
        """
        # Get base reasoning from parent
        base_reasoning = super()._generate_reasoning(prediction, stats, is_safe)
        
        # Add quantization info
        confidence = prediction.get('confidence', 0.5)
        high_conf_zone = self.quant_config['high_confidence_zone']
        
        quant_parts = []
        
        # Confidence zone info
        if high_conf_zone[0] <= confidence <= high_conf_zone[1]:
            quant_parts.append(
                f"In high-confidence zone ({high_conf_zone[0]:.0%}-{high_conf_zone[1]:.0%}) "
                f"- using granular position control"
            )
        else:
            quant_parts.append(
                f"Outside high-confidence zone - using coarser position control"
            )
        
        # Position change info
        if abs(self.last_position) > 0.01:
            quant_parts.append(f"Last position: {self.last_position:.2f}")
        
        # Combine
        if quant_parts:
            return base_reasoning + " [Quant: " + "; ".join(quant_parts) + "]"
        else:
            return base_reasoning


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
    )
    
    print("\n=== Testing Kronos Quant Trader ===\n")
    
    # Create trader
    trader = KronosQuantTrader(symbol="BTCUSDC", name="test_quant")
    
    # Mock stats
    mock_stats = {
        'current_price': 95000.0,
        'current_ma7': 94500.0,
        'current_rsi': 55.0,
    }
    
    # Test multiple confidence levels
    test_confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    print("Testing position quantization at different confidence levels:\n")
    
    for conf in test_confidences:
        # Simulate prediction
        mock_prediction = {
            'confidence': conf,
            'trend': 'up',
            'predicted_price': 96000.0,
            'current_price': 95000.0,
            'is_stale': False,
            'staleness_hours': 0.5,
        }
        
        # Calculate raw and quantized positions
        raw_pos = conf  # Simple mapping for test
        quant_pos = trader._quantize_position(raw_pos, conf)
        
        print(f"Confidence: {conf:.2f} | Raw: {raw_pos:.3f} | Quantized: {quant_pos:.2f}")
    
    print("\n=== Full Analysis Test ===\n")
    
    # Run full analysis
    result = trader.analyze(mock_stats)
    
    print(f"Position: {result['position']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Is Safe: {result.get('is_safe', False)}")
    print(f"\nReasoning:\n{result['reasoning']}")
    
    # Generate final output
    output = trader.generate_output(result, mock_stats, None)
    print(f"\n=== Final Output ===")
    print(json.dumps(output, indent=2, default=str))
