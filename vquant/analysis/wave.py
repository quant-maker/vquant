#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Wave Trader - A simple strategy that buys on dips and sells on rises
Tracks last trade price to avoid frequent trading
"""


import json
import logging

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from .base import BasePredictor


logger = logging.getLogger(__name__)


class WaveTrader(BasePredictor):
    """
    Wave-based trading strategy
    
    Strategy Logic:
    - Buy (increase position) when price drops by threshold percentage
    - Sell (decrease position) when price rises by threshold percentage
    - Track last trade to avoid frequent trading
    - Store state in JSON file for persistence
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDC",
        name: str = "default",
        config_dir: str = "config"
    ):
        """
        Initialize wave trader
        
        Args:
            symbol: Trading pair symbol
            name: Strategy name (used for state file naming)
            config_dir: Directory containing config files
        """
        super().__init__(symbol, name)
        
        # Load configuration from file
        config = self._load_config(self.name, self.symbol, config_dir)

        self.buy_threshold = config.get('buy_threshold', -0.5)
        self.sell_threshold = config.get('sell_threshold', 0.5)
        self.min_trade_interval = config.get('min_trade_interval', 300)
        self.state_file = Path(f"data/wave_state_{name}.json")
        
        logger.info(
            f"Wave trader initialized: buy={self.buy_threshold}%, "
            f"sell={self.sell_threshold}%, interval={self.min_trade_interval}s"
        )
        # Load or initialize state
        self.state = self._load_state()
    
    def _load_config(self, name: str, symbol: str, config_dir: str) -> Dict[str, Any]:
        """
        Load wave trader configuration from file
        
        Args:
            symbol: Trading pair symbol
            config_dir: Directory containing config files
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: Config file does not exist
        """
        config_path = Path(config_dir) / f"wave_{name}.json"
        
        # Try specific config file first, then fallback to wave.json
        if not config_path.exists():
            fallback_path = Path(config_dir) / "wave.json"
            if fallback_path.exists():
                logger.info(f"Config {config_path.name} not found, using fallback: {fallback_path.name}")
                config_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Wave trader config file not found: {config_path}\n"
                    f"Also tried fallback: {fallback_path}\n"
                    f"Please create config file for {symbol}"
                )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate symbol matches (if specified in config)
            if "symbol" in config and config["symbol"] != symbol:
                logger.warning(
                    f"Config file symbol '{config['symbol']}' doesn't match requested '{symbol}', "
                    f"using config symbol"
                )
            
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file JSON format error: {e}")
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load trading state from file
        
        Returns:
            State dictionary with last trade info
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"Loaded trading state from {self.state_file}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}, using default state")
        
        # Default state
        return {
            "symbol": self.symbol,
            "last_trade_price": None,
            "last_trade_time": None,
            "last_trade_action": None,
            "last_trade_volume": None
        }
    
    def _save_state(self):
        """Save trading state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved trading state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
    
    def _can_trade(self) -> bool:
        """
        Check if enough time has passed since last trade
        
        Returns:
            True if can trade, False otherwise
        """
        if self.state["last_trade_time"] is None:
            return True
        
        last_time = datetime.fromisoformat(self.state["last_trade_time"])
        current_time = datetime.now()
        elapsed_seconds = (current_time - last_time).total_seconds()
        
        if elapsed_seconds < self.min_trade_interval:
            logger.info(
                f"Too soon to trade. Last trade: {elapsed_seconds:.0f}s ago, "
                f"minimum interval: {self.min_trade_interval}s"
            )
            return False
        
        return True
    
    def _calculate_price_change(self, current_price: float) -> Optional[float]:
        """
        Calculate price change percentage since last trade
        
        Args:
            current_price: Current market price
            
        Returns:
            Price change percentage, or None if no last trade price
        """
        if self.state["last_trade_price"] is None:
            return None
        
        last_price = self.state["last_trade_price"]
        change_pct = ((current_price - last_price) / last_price) * 100
        return change_pct
    
    def _record_trade(self, action: str, price: float, volume: float, reasoning: str):
        """
        Record trade in state history
        
        Args:
            action: Trade action (buy/sell/hold)
            price: Trade price
            volume: Trade volume
            reasoning: Trade reasoning
        """
        self.state["last_trade_price"] = price
        self.state["last_trade_time"] = datetime.now().isoformat()
        self.state["last_trade_action"] = action
        self.state["last_trade_volume"] = volume
        
        self._save_state()
        logger.info(f"Recorded trade: {action} {volume} @ {price}")
    
    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Wave trader doesn't need chart generation
        
        Returns:
            (None, None) as no chart is needed
        """
        logger.info("Wave trader mode: skipping chart generation")
        return None, None
    
    def analyze(self, stats: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze using wave trading strategy
        
        Args:
            stats: Statistics dictionary
            **kwargs: Additional parameters (volume from args)
            
        Returns:
            Trading signal
        """
        current_price = stats.get('current_price')
        args = kwargs.get('args')
        volume = args.volume if args and args.volume > 0 else 0.1
        
        return self.generate_signal(
            current_price=current_price,
            volume=volume,
            stats=stats
        )
    
    def generate_output(self, result: Dict[str, Any], stats: Dict[str, Any], args) -> Dict[str, Any]:
        """
        Generate standardized output for wave trader
        
        Args:
            result: Signal result from analyze()
            stats: Statistics dictionary
            args: Command line arguments
            
        Returns:
            Standardized output dictionary
        """
        # Convert action to position
        action_to_position = {
            'buy': 1.0,
            'sell': -1.0,
            'hold': 0.0
        }
        
        return {
            'symbol': result['symbol'],
            'position': action_to_position.get(result['action'], 0.0),
            'confidence': 'high' if result['action'] in ['buy', 'sell'] else 'low',
            'current_price': result['current_price'],
            'action': result['action'],
            'volume': result['volume'],
            'reasoning': result['reasoning'],
            'analysis_type': 'wave',
            'price_change': result.get('price_change'),
        }
    
    def generate_signal(
        self,
        current_price: float,
        volume: float = 0.1,
        stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal based on price wave
        
        Args:
            current_price: Current market price
            volume: Default trade volume
            stats: Optional market statistics for additional context
            
        Returns:
            Trade signal dictionary with format:
            {
                "action": "buy"/"sell"/"hold",
                "volume": float,
                "symbol": str,
                "current_price": float,
                "reasoning": str,
                "price_change": float (optional),
                "current_position": float
            }
        """
        logger.info(f"Generating signal for {self.symbol} @ ${current_price}")
        
        # Check if we can trade
        if not self._can_trade():
            return {
                "action": "hold",
                "volume": 0.0,
                "symbol": self.symbol,
                "current_price": current_price,
                "reasoning": "Minimum trade interval not reached"
            }
        
        # Calculate price change since last trade
        price_change = self._calculate_price_change(current_price)
        
        # Initialize with hold signal
        signal = {
            "action": "hold",
            "volume": 0.0,
            "symbol": self.symbol,
            "current_price": current_price,
            "reasoning": "",
            "price_change": price_change
        }
        # First trade - establish initial position
        if price_change is None:
            signal["reasoning"] = f"Initial entry at ${current_price:.2f}"
            self._record_trade("hold", current_price, 0.0, signal["reasoning"])
            return signal
        
        # Check for buy signal (price dropped)
        if price_change <= self.buy_threshold:
            signal["action"] = "buy"
            signal["volume"] = volume
            signal["reasoning"] = (
                f"Price dropped {price_change:.2f}% from ${self.state['last_trade_price']:.2f} "
                f"to ${current_price:.2f}, adding position"
            )
            self._record_trade("buy", current_price, volume, signal["reasoning"])
        
        # Check for sell signal (price rose)
        elif price_change >= self.sell_threshold:
            signal["action"] = "sell"
            signal["volume"] = volume
            signal["reasoning"] = (
                f"Price rose {price_change:.2f}% from ${self.state['last_trade_price']:.2f} "
                f"to ${current_price:.2f}, taking profit"
            )
            self._record_trade("sell", current_price, volume, signal["reasoning"])
        
        else:
            signal["reasoning"] = (
                f"Price change {price_change:.2f}% within threshold "
                f"(buy: {self.buy_threshold}%, sell: {self.sell_threshold}%), holding position"
            )
        
        logger.info(
            f"Signal: {signal['action'].upper()} - {signal['reasoning']}"
        )
        
        return signal
