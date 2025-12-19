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
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


class WaveTrader:
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
        buy_threshold: float = -0.5,  # Buy when price drops 0.5%
        sell_threshold: float = 0.5,   # Sell when price rises 0.5%
        min_trade_interval: int = 300,  # Minimum 5 minutes between trades (seconds)
        max_position: float = 1.0,      # Maximum position size
        state_file: str = "data/wave_state.json"
    ):
        """
        Initialize wave trader
        
        Args:
            symbol: Trading pair symbol
            buy_threshold: Price drop percentage to trigger buy (negative value)
            sell_threshold: Price rise percentage to trigger sell (positive value)
            min_trade_interval: Minimum seconds between trades
            max_position: Maximum position size
            state_file: File path to store trading state
        """
        self.symbol = symbol
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_trade_interval = min_trade_interval
        self.max_position = max_position
        self.state_file = Path(state_file)
        
        # Load or initialize state
        self.state = self._load_state()
    
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
            "last_trade_action": None
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
            return 0.0
        
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
        
        self._save_state()
        logger.info(f"Recorded trade: {action} {volume} @ {price}")
    
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
            signal["action"] = "buy"
            signal["volume"] = volume
            signal["reasoning"] = f"Initial entry at ${current_price:.2f}"
            self._record_trade("buy", current_price, volume, signal["reasoning"])
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
    
    def reset_state(self):
        """Reset trading state (use with caution)"""
        logger.warning("Resetting trading state")
        self.state = {
            "symbol": self.symbol,
            "last_trade_price": None,
            "last_trade_time": None,
            "last_trade_action": None
        }
        self._save_state()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current trading state
        
        Returns:
            Dictionary with state summary
        """
        return {
            "symbol": self.state["symbol"],
            "last_trade": {
                "price": self.state["last_trade_price"],
                "time": self.state["last_trade_time"],
                "action": self.state["last_trade_action"]
            }
        }


def main():
    """Command line usage example"""
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python wave.py <current_price> [symbol] [volume]")
        print("Example: python wave.py 88888 BTCUSDC 0.1")
        sys.exit(1)
    
    current_price = float(sys.argv[1])
    symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDC"
    volume = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    # Create trader
    trader = WaveTrader(symbol=symbol)
    
    # Generate signal
    signal = trader.generate_signal(current_price, volume)
    
    # Print result
    print(json.dumps(signal, indent=2))
    
    # Print state summary
    print("\nCurrent State:")
    print(json.dumps(trader.get_state_summary(), indent=2))


if __name__ == "__main__":
    main()
