#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Quantitative Predictor - Prediction model based on funding rate and technical indicators
Predicts market direction and provides position recommendations
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


class QuantPredictor:
    """
    Quantitative Predictor - Predicts market direction based on multiple factor scores

    Scoring Factors:
    1. Funding Rate - Reflects long/short balance
    2. Momentum - Price momentum trend
    3. Impulse - Momentum acceleration
    4. MA Trend - Price position relative to MA7
    5. RSI - Overbought/oversold indicator
    6. MACD - MACD crossover signals
    7. Volume - Volume confirmation
    """

    def __init__(self, symbol: str = "BTCUSDC", config_dir: str = "config"):
        """
        Initialize predictor
        
        Args:
            symbol: Trading pair symbol (used to load corresponding threshold config)
            config_dir: Configuration file directory
        """
        self.symbol = symbol
        self.weights = {
            "funding_rate": 0.20,  # Funding rate weight
            "momentum": 0.20,      # Momentum weight
            "impulse": 0.15,       # Impulse weight (momentum acceleration)
            "ma_trend": 0.15,      # MA trend weight
            "rsi": 0.12,           # RSI weight
            "macd": 0.10,          # MACD weight
            "volume": 0.08,        # Volume weight
        }
        
        # Load threshold configuration (must exist, otherwise throw exception)
        self.thresholds = self._load_thresholds(symbol, config_dir)
    
    def _load_thresholds(self, symbol: str, config_dir: str) -> Dict:
        """
        Load threshold configuration from JSON file
        
        Args:
            symbol: Trading pair symbol
            config_dir: Configuration file directory
            
        Returns:
            Threshold configuration dictionary
            
        Raises:
            FileNotFoundError: Config file does not exist
            ValueError: Config file format error
        """
        config_path = Path(config_dir) / f"thresholds_{symbol.lower()}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Threshold config file not found: {config_path}\n"
                f"Please run calibrator first: python -m vquant.model.calibrator --symbol {symbol}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'thresholds' not in data:
                raise ValueError(f"Config file format error: missing 'thresholds' field")
            
            logger.info(f"Loaded threshold config: {config_path} (updated: {data.get('updated_at', 'unknown')})")
            return data['thresholds']
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file JSON format error: {e}")
    def predict(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict market direction based on market data and provide position recommendations

        Args:
            stats: Dictionary containing technical indicators and market data
                - funding_rate: Current funding rate
                - momentum: Price momentum (rate of change between recent and earlier averages)
                - current_ma7: Current MA7 value
                - current_price: Current price
                - current_rsi: Current RSI value
                - current_macd: Current MACD value
                - current_signal: Current MACD signal line value
                - volume_strength: Volume strength

        Returns:
            {
                "position": float,  # Position recommendation from -1.0 to 1.0
                "confidence": str,  # "low", "medium", "high"
                "score": float,     # Composite score from -100 to 100
                "factors": dict,    # Individual factor scores
                "reasoning": str,   # Decision reasoning
            }
        """
        logger.info("Starting market direction prediction...")

        # Calculate individual factor scores
        factors = {}

        # 1. Funding Rate score (-100 to 100)
        factors["funding_rate"] = self._score_funding_rate(stats.get("funding_rate", 0))

        # 2. Momentum score (-100 to 100)
        factors["momentum"] = self._score_momentum(
            stats.get("momentum", 0)
        )

        # 3. Impulse score (-100 to 100)
        factors["impulse"] = self._score_impulse(
            stats.get("impulse", 0)
        )

        # 4. MA trend score (-100 to 100)
        factors["ma_trend"] = self._score_ma_trend(
            stats.get("current_ma7"), stats.get("current_price")
        )

        # 5. RSI score (-100 to 100)
        factors["rsi"] = self._score_rsi(stats.get("current_rsi"))

        # 6. MACD score (-100 to 100)
        factors["macd"] = self._score_macd(
            stats.get("current_macd"), stats.get("current_signal")
        )

        # 7. Volume score (-100 to 100)
        factors["volume"] = self._score_volume(stats.get("volume_strength", 0))

        # Calculate weighted composite score
        total_score = sum(factors[key] * self.weights[key] for key in factors.keys())

        # Convert score to position recommendation (-1.0 to 1.0)
        position = self._score_to_position(total_score)

        # Assess confidence
        confidence = self._calculate_confidence(factors, total_score)

        # Generate decision reasoning
        reasoning = self._generate_reasoning(factors, total_score, stats)

        result = {
            "position": round(position, 2),
            "confidence": confidence,
            "score": round(total_score, 2),
            "factors": {k: round(v, 2) for k, v in factors.items()},
            "reasoning": reasoning,
        }

        logger.info(
            f"Prediction complete: position={result['position']}, confidence={result['confidence']}"
        )
        logger.debug(f"Factor scores: {result['factors']}")

        return result

    def _score_funding_rate(self, funding_rate: Optional[float]) -> float:
        """
        Funding rate scoring (adjusted based on DOGE/BTC actual historical data)
        
        Actual data analysis (100 periods history):
        - Mean: 0.003%
        - 90th percentile: 0.009%
        - 10th percentile: -0.002%
        - Max: 0.010%
        - Min: -0.011%
        - 100% data within -0.02% to 0.02%
        
        Positive funding rate: Longs pay shorts, market is bullish -> May be overheated
        Negative funding rate: Shorts pay longs, market is bearish -> May be oversold
        """
        if funding_rate is None:
            return 0.0
        
        # Set thresholds based on actual data distribution
        # 90th percentile ~0.009%, 10th percentile ~-0.002%
        if funding_rate > 0.010:
            return -100.0  # Extremely rare, strongly bearish
        elif funding_rate > 0.008:
            return -80.0   # >90th percentile, longs overheated
        elif funding_rate > 0.006:
            return -50.0   # >75th percentile, longs strong
        elif funding_rate > 0.003:
            return -20.0   # >Median, longs slightly strong
        elif funding_rate > -0.002:
            return 0.0     # Neutral range (10th percentile to median)
        elif funding_rate > -0.005:
            return 30.0    # <10th percentile, shorts strong
        elif funding_rate > -0.008:
            return 60.0    # Shorts quite strong
        elif funding_rate > -0.010:
            return 80.0    # Near historical minimum
        else:
            return 100.0   # Extremely rare, strongly bullish
    
    def _score_momentum(self, momentum: float) -> float:
        """
        Momentum scoring (based on thresholds in config file)
        
        momentum: Rate of change between recent and earlier averages (%)
        """
        t = self.thresholds.get('momentum', {})
        
        if momentum > t.get('extreme_bullish', 4):
            return 100.0   # Extremely strong uptrend
        elif momentum > t.get('strong_bullish', 2.5):
            return 80.0    # Strong uptrend
        elif momentum > t.get('bullish', 1.0):
            return 50.0    # Uptrend
        elif momentum > t.get('neutral_high', 0):
            return 20.0    # Slightly up
        elif momentum > t.get('neutral_low', -1.5):
            return -20.0   # Slightly down
        elif momentum > t.get('bearish', -3.0):
            return -50.0   # Downtrend
        elif momentum > t.get('strong_bearish', -4.0):
            return -80.0   # Strong downtrend
        else:
            return -100.0  # Extremely weak downtrend

    def _score_impulse(self, impulse: float) -> float:
        """
        Impulse scoring (based on thresholds in config file)
        
        Impulse measures momentum acceleration - indicates if momentum is speeding up or slowing down
        Positive impulse: momentum accelerating (bullish)
        Negative impulse: momentum decelerating (bearish)
        """
        t = self.thresholds.get('impulse', {})
        
        if impulse > t.get('extreme_bullish', 5):
            return 100.0   # Extremely strong acceleration
        elif impulse > t.get('strong_bullish', 3):
            return 80.0    # Strong acceleration
        elif impulse > t.get('bullish', 1.5):
            return 50.0    # Acceleration
        elif impulse > t.get('neutral_high', 0):
            return 20.0    # Slight acceleration
        elif impulse > t.get('neutral_low', -1.5):
            return -20.0   # Slight deceleration
        elif impulse > t.get('bearish', -3):
            return -50.0   # Deceleration
        elif impulse > t.get('strong_bearish', -5):
            return -80.0   # Strong deceleration
        else:
            return -100.0  # Extremely weak deceleration

    def _score_ma_trend(
        self, ma7: Optional[float], current_price: Optional[float]
    ) -> float:
        """
        MA7 trend scoring (based on thresholds in config file)

        Price above MA7: Uptrend
        Price below MA7: Downtrend
        """
        if ma7 is None or current_price is None:
            return 0.0

        # Calculate price deviation from MA7
        deviation = (current_price - ma7) / ma7 * 100 if ma7 > 0 else 0

        # Score using configured thresholds
        t = self.thresholds.get('ma_deviation', {})
        
        if deviation > t.get('extreme_bullish', 1.5):
            return 100.0   # Extremely strong uptrend
        elif deviation > t.get('strong_bullish', 1.0):
            return 80.0    # Strong uptrend
        elif deviation > t.get('bullish', 0.4):
            return 50.0    # Uptrend
        elif deviation > t.get('neutral_high', 0):
            return 20.0    # Slightly up
        elif deviation > t.get('neutral_low', -0.4):
            return -20.0   # Slightly down
        elif deviation > t.get('bearish', -1.0):
            return -50.0   # Downtrend
        elif deviation > t.get('strong_bearish', -2.0):
            return -80.0   # Strong downtrend
        else:
            return -100.0  # Extremely weak downtrend

    def _score_rsi(self, rsi: Optional[float]) -> float:
        """
        RSI scoring (based on thresholds in config file)

        High RSI: Overbought, possible pullback
        Low RSI: Oversold, possible bounce
        """
        if rsi is None:
            return 0.0

        t = self.thresholds.get('rsi', {})
        
        if rsi > t.get('extreme_overbought', 79):
            return -100.0  # Extremely overbought
        elif rsi > t.get('overbought', 73):
            return -80.0   # Overbought
        elif rsi > t.get('slightly_overbought', 61):
            return -50.0   # Slightly strong
        elif rsi > t.get('neutral_high', 48):
            return -20.0   # Moderately strong
        elif rsi > t.get('neutral_low', 35):
            return 20.0    # Moderately weak
        elif rsi > t.get('slightly_oversold', 20):
            return 50.0    # Slightly weak
        elif rsi > t.get('oversold', 16):
            return 80.0    # Oversold
        else:
            return 100.0   # Extremely oversold

    def _score_macd(self, macd: Optional[float], signal: Optional[float]) -> float:
        """
        MACD scoring (based on thresholds in config file)

        MACD > Signal: Golden cross, bullish
        MACD < Signal: Death cross, bearish
        """
        if macd is None or signal is None:
            return 0.0

        diff = macd - signal
        t = self.thresholds.get('macd_diff', {})

        if diff > t.get('extreme_bullish', 0.004):
            return 100.0        # Extremely strong golden cross
        elif diff > t.get('strong_bullish', 0.003):
            return 80.0         # Strong golden cross
        elif diff > t.get('bullish', 0.0015):
            return 50.0         # Golden cross
        elif diff > t.get('neutral_high', 0):
            return 20.0         # Weak golden cross
        elif diff > t.get('neutral_low', -0.0014):
            return -20.0        # Weak death cross
        elif diff > t.get('bearish', -0.003):
            return -50.0        # Death cross
        elif diff > t.get('strong_bearish', -0.004):
            return -80.0        # Strong death cross
        else:
            return -100.0       # Extremely strong death cross

    def _score_volume(self, volume_strength: float) -> float:
        """
        Volume scoring (based on thresholds in config file)

        volume_strength: Percentage change relative to 20-period average
        High volume confirms trend, low volume weakens trend
        """
        t = self.thresholds.get('volume_change', {})
        
        if volume_strength > t.get('extreme_volume', 230):
            return 100.0   # Extremely high volume
        elif volume_strength > t.get('high_volume', 120):
            return 80.0    # Significantly high volume
        elif volume_strength > t.get('above_average', 20):
            return 50.0    # High volume
        elif volume_strength > t.get('neutral', -30):
            return 0.0     # Neutral range
        elif volume_strength > t.get('below_average', -55):
            return -30.0   # Low volume
        elif volume_strength > t.get('low_volume', -70):
            return -60.0   # Significantly low volume
        else:
            return -80.0   # Extremely low volume

    def _score_to_position(self, score: float) -> float:
        """
        Convert score to position recommendation

        score: -100 to 100
        position: -1.0 to 1.0

        Use non-linear mapping to amplify extreme signals
        """
        # Linear mapping
        position = score / 100.0

        # Non-linear adjustment: amplify strong signals, suppress weak signals
        if abs(position) > 0.6:
            # Strong signal: maintain or amplify
            position = position * 1.1
        elif abs(position) < 0.2:
            # Weak signal: further suppress
            position = position * 0.5

        # Limit to -1.0 to 1.0
        return max(-1.0, min(1.0, position))

    def _calculate_confidence(
        self, factors: Dict[str, float], total_score: float
    ) -> str:
        """
        Calculate confidence level

        high: Factors highly aligned, and large absolute total score
        medium: Factors partially aligned, or moderate total score
        low: Factors divergent, or total score close to 0
        """
        # Count positive and negative factors
        positive = sum(1 for v in factors.values() if v > 20)
        negative = sum(1 for v in factors.values() if v < -20)
        neutral = len(factors) - positive - negative

        # Direction consistency
        consistency = max(positive, negative) / len(factors)

        # Overall assessment
        abs_score = abs(total_score)

        if consistency > 0.75 and abs_score > 50:
            return "high"
        elif consistency > 0.5 and abs_score > 30:
            return "medium"
        else:
            return "low"

    def _generate_reasoning(
        self, factors: Dict[str, float], total_score: float, stats: Dict[str, Any]
    ) -> str:
        """Generate decision reasoning - Structured output: conclusion + supporting factors + hedging factors"""

        # Classify factors: bullish, bearish, neutral
        bullish_factors = []
        bearish_factors = []
        neutral_factors = []

        # Funding Rate (adjusted based on actual data distribution)
        fr = stats.get("funding_rate", 0)
        if fr is not None:
            if fr > 0.008:
                bearish_factors.append(f"Funding rate {fr:.4f}% too high (>90th percentile), longs overheated")
            elif fr > 0.006:
                bearish_factors.append(f"Funding rate {fr:.4f}% high (>75th percentile)")
            elif fr > 0.003:
                bearish_factors.append(f"Funding rate {fr:.4f}% slightly high")
            elif fr < -0.005:
                bullish_factors.append(f"Funding rate {fr:.4f}% low, shorts dominating")
            elif fr < -0.002:
                bullish_factors.append(f"Funding rate {fr:.4f}% slightly low (<10th percentile)")
            else:
                neutral_factors.append(f"Funding rate {fr:.4f}% neutral")

        # Momentum
        momentum = stats.get("momentum", 0)
        if momentum > 2:
            bullish_factors.append(f"Strong momentum +{momentum:.1f}%")
        elif momentum > 0.5:
            bullish_factors.append(f"Upward momentum +{momentum:.1f}%")
        elif momentum < -2:
            bearish_factors.append(f"Declining momentum {momentum:.1f}%")
        elif momentum < -0.5:
            bearish_factors.append(f"Downward momentum {momentum:.1f}%")

        # MA7 trend
        ma7 = stats.get("current_ma7")
        current_price = stats.get("current_price")
        if ma7 and current_price:
            deviation = (current_price - ma7) / ma7 * 100 if ma7 > 0 else 0
            if deviation > 2:
                bullish_factors.append(f"Price above MA7 +{deviation:.1f}%")
            elif deviation > 0.5:
                bullish_factors.append(f"Price slightly above MA7 +{deviation:.1f}%")
            elif deviation < -2:
                bearish_factors.append(f"Price below MA7 {deviation:.1f}%")
            elif deviation < -0.5:
                bearish_factors.append(f"Price slightly below MA7 {deviation:.1f}%")
            else:
                neutral_factors.append("Price near MA7")

        # RSI
        rsi = stats.get("current_rsi")
        if rsi:
            if rsi > 70:
                bearish_factors.append(f"RSI {rsi:.1f} overbought")
            elif rsi > 60:
                bearish_factors.append(f"RSI {rsi:.1f} high")
            elif rsi < 30:
                bullish_factors.append(f"RSI {rsi:.1f} oversold")
            elif rsi < 40:
                bullish_factors.append(f"RSI {rsi:.1f} low")

        # MACD
        macd = stats.get("current_macd")
        signal = stats.get("current_signal")
        if macd and signal:
            if macd > signal:
                if macd > 0:
                    bullish_factors.append("MACD strong golden cross")
                else:
                    bullish_factors.append("MACD weak golden cross")
            else:
                if macd > 0:
                    bearish_factors.append("MACD weak death cross")
                else:
                    bearish_factors.append("MACD strong death cross")

        # Volume
        vol_strength = stats.get("volume_strength", 0)
        if vol_strength > 50:
            bullish_factors.append(f"Volume increased {vol_strength:.0f}%")
        elif vol_strength < -50:
            bearish_factors.append(f"Volume decreased {vol_strength:.0f}%")

        # Build reasoning
        if total_score > 50:
            conclusion = "[LONG] Strong bullish signal"
        elif total_score > 20:
            conclusion = "[SMALL LONG] Moderately bullish"
        elif total_score < -50:
            conclusion = "[SHORT] Strong bearish signal"
        elif total_score < -20:
            conclusion = "[SMALL SHORT] Moderately bearish"
        else:
            conclusion = "[HOLD] Neutral signal"

        parts = [conclusion]

        if bullish_factors:
            parts.append("Bullish signals: " + ", ".join(bullish_factors))

        if bearish_factors:
            parts.append("Bearish signals: " + ", ".join(bearish_factors))

        if neutral_factors:
            parts.append("Neutral: " + ", ".join(neutral_factors))

        # Special note if neutral with conflicting signals
        if abs(total_score) <= 20 and (bullish_factors or bearish_factors):
            parts.append("Bullish and bearish signals offset, no clear direction")

        return ". ".join(parts) + "."


if __name__ == "__main__":
    # Test case
    logging.basicConfig(level=logging.INFO)

    predictor = QuantPredictor()

    # Mock data
    test_stats = {
        "funding_rate": 0.08,  # Funding rate high
        "momentum": -3.5,           # Momentum declining
        "current_ma7": 100.5,
        "current_price": 98.0,  # Price below MA7
        "current_rsi": 65.0,
        "current_macd": 0.5,
        "current_signal": 0.3,
        "volume_strength": 80.0,
    }

    result = predictor.predict(test_stats)

    print("\n=== Prediction Result ===")
    print(f"Position: {result['position']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Score: {result['score']}")
    print(f"Factors: {result['factors']}")
    print(f"Reasoning: {result['reasoning']}")
