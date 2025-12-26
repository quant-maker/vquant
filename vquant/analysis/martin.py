#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Martin Trader - Martingale strategy that doubles position on losses
Tracks position history and adjusts volume based on win/loss
"""


import json
import logging

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from .base import BasePredictor


logger = logging.getLogger(__name__)


class MartinTrader(BasePredictor):
    """
    Martingale-based trading strategy
    
    Strategy Logic:
    - Start with base volume
    - On loss: double the volume for next trade (attempt to recover)
    - On win: reset to base volume
    - Track position and P&L to determine win/loss
    - Store state in JSON file for persistence
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDC",
        name: str = "default",
        config_dir: str = "config"
    ):
        """
        Initialize martin trader
        
        Args:
            symbol: Trading pair symbol
            name: Strategy name (used for state file naming)
            config_dir: Directory containing config files
        """
        super().__init__(symbol, name)
        
        # Load configuration from file
        config = self._load_config(self.name, self.symbol, config_dir)

        self.base_volume = config.get('base_volume', 0.01)
        self.max_volume = config.get('max_volume', 1.0)
        self.max_multiplier = config.get('max_multiplier', 8)
        self.profit_target = config.get('profit_target', 0.5)  # % profit to take
        self.loss_threshold = config.get('loss_threshold', -0.5)  # % loss to trigger martingale
        self.min_trade_interval = config.get('min_trade_interval', 300)
        
        # Safety parameters to prevent liquidation
        self.max_total_position = config.get('max_total_position', 0.5)  # Max total position size
        self.hard_stop_loss = config.get('hard_stop_loss', -5.0)  # Hard stop loss %
        self.max_drawdown = config.get('max_drawdown', -10.0)  # Max drawdown from peak %
        self.account_balance = config.get('account_balance', 10000.0)  # Account balance for risk calculation
        self.max_position_pct = config.get('max_position_pct', 0.3)  # Max position as % of balance
        self.cooldown_after_losses = config.get('cooldown_after_losses', 3)  # Enter cooldown after N losses
        self.cooldown_duration = config.get('cooldown_duration', 3600)  # Cooldown duration in seconds
        self.emergency_stop = config.get('emergency_stop', False)  # Manual emergency stop
        
        self.state_file = Path(f"data/martin_state_{name}.json")
        
        logger.info(
            f"Martin trader initialized: base={self.base_volume}, "
            f"max={self.max_volume}, multiplier={self.max_multiplier}, "
            f"profit={self.profit_target}%, loss={self.loss_threshold}%"
        )
        logger.info(
            f"Safety limits: max_position={self.max_total_position}, "
            f"hard_stop={self.hard_stop_loss}%, max_drawdown={self.max_drawdown}%, "
            f"cooldown_after={self.cooldown_after_losses} losses"
        )
        # Load or initialize state
        self.state = self._load_state()
    
    def _load_config(self, name: str, symbol: str, config_dir: str) -> Dict[str, Any]:
        """
        Load martin trader configuration from file
        
        Args:
            name: Strategy name
            symbol: Trading pair symbol
            config_dir: Directory containing config files
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: Config file does not exist
        """
        config_path = Path(config_dir) / f"martin_{name}.json"
        
        # Try specific config file first, then fallback to martin.json
        if not config_path.exists():
            fallback_path = Path(config_dir) / "martin.json"
            if fallback_path.exists():
                logger.info(f"Config {config_path.name} not found, using fallback: {fallback_path.name}")
                config_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Martin trader config file not found: {config_path}\n"
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
            State dictionary with position and trade history
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
            "current_position": 0.0,  # Current position size (positive = long, negative = short)
            "entry_price": None,  # Average entry price
            "current_multiplier": 1,  # Current martingale multiplier
            "consecutive_losses": 0,  # Number of consecutive losses
            "last_trade_time": None,
            "last_trade_action": None,
            "last_trade_volume": None,
            "total_pnl": 0.0,  # Total P&L
            "trade_count": 0,  # Total number of trades
            "win_count": 0,  # Number of winning trades
            "peak_pnl": 0.0,  # Peak P&L for drawdown calculation
            "cooldown_until": None,  # Cooldown end time
            "emergency_stop_triggered": False,  # Emergency stop flag
            "max_position_reached": 0.0,  # Max position ever reached
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
    
    def _is_in_cooldown(self) -> bool:
        """
        Check if strategy is in cooldown period after consecutive losses
        
        Returns:
            True if in cooldown, False otherwise
        """
        if self.state["cooldown_until"] is None:
            return False
        
        cooldown_end = datetime.fromisoformat(self.state["cooldown_until"])
        if datetime.now() < cooldown_end:
            remaining = (cooldown_end - datetime.now()).total_seconds()
            logger.warning(
                f"In cooldown period. Remaining: {remaining:.0f}s "
                f"({self.state['consecutive_losses']} consecutive losses)"
            )
            return True
        else:
            # Cooldown expired, reset
            self.state["cooldown_until"] = None
            logger.info("Cooldown period ended, resuming trading")
            return False
    
    def _check_safety_limits(self, current_price: float, new_volume: float = 0) -> Tuple[bool, str]:
        """
        Check if trading would violate safety limits
        
        Args:
            current_price: Current market price
            new_volume: Additional volume to be added
            
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check emergency stop
        if self.emergency_stop or self.state.get("emergency_stop_triggered", False):
            return False, "Emergency stop activated"
        
        # Check total position limit
        potential_position = abs(self.state["current_position"]) + new_volume
        if potential_position > self.max_total_position:
            return False, f"Position limit: {potential_position:.4f} > {self.max_total_position}"
        
        # Check position as percentage of account balance
        position_value = potential_position * current_price
        position_pct = (position_value / self.account_balance) * 100
        max_pct = self.max_position_pct * 100
        if position_pct > max_pct:
            return False, f"Position value {position_pct:.1f}% > {max_pct:.1f}% of balance"
        
        # Check hard stop loss
        pnl_pct = self._calculate_pnl(current_price)
        if pnl_pct is not None and pnl_pct <= self.hard_stop_loss:
            return False, f"Hard stop loss: {pnl_pct:.2f}% <= {self.hard_stop_loss}%"
        
        # Check max drawdown from peak
        peak = self.state.get("peak_pnl", 0.0)
        current_total_pnl = self.state["total_pnl"]
        if pnl_pct is not None:
            current_total_pnl += pnl_pct
        
        drawdown = current_total_pnl - peak
        if drawdown <= self.max_drawdown:
            return False, f"Max drawdown: {drawdown:.2f}% from peak {peak:.2f}%"
        
        # Check consecutive losses cooldown
        if self.state["consecutive_losses"] >= self.cooldown_after_losses:
            if self.state["cooldown_until"] is None:
                # Enter cooldown
                cooldown_end = datetime.now() + timedelta(seconds=self.cooldown_duration)
                self.state["cooldown_until"] = cooldown_end.isoformat()
                self._save_state()
                logger.warning(
                    f"Entering cooldown for {self.cooldown_duration}s after "
                    f"{self.state['consecutive_losses']} consecutive losses"
                )
            return False, f"Cooldown after {self.state['consecutive_losses']} losses"
        
        return True, "All safety checks passed"
    
    def _calculate_pnl(self, current_price: float) -> Optional[float]:
        """
        Calculate unrealized P&L percentage
        
        Args:
            current_price: Current market price
            
        Returns:
            P&L percentage, or None if no position
        """
        if self.state["entry_price"] is None or self.state["current_position"] == 0:
            return None
        
        entry_price = self.state["entry_price"]
        position = self.state["current_position"]
        
        # P&L calculation based on position direction
        if position > 0:  # Long position
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # Short position
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        return pnl_pct
    
    def _calculate_next_volume(self) -> float:
        """
        Calculate next trade volume based on martingale multiplier
        
        Returns:
            Next trade volume
        """
        volume = self.base_volume * self.state["current_multiplier"]
        return min(volume, self.max_volume)
    
    def _record_trade(self, action: str, price: float, volume: float, reasoning: str):
        """
        Record trade and update position
        
        Args:
            action: Trade action (buy/sell/close)
            price: Trade price
            volume: Trade volume
            reasoning: Trade reasoning
        """
        old_position = self.state["current_position"]
        
        if action == "buy":
            # Add to long position
            new_position = old_position + volume
            # Update entry price (weighted average)
            if self.state["entry_price"] is None:
                self.state["entry_price"] = price
            else:
                total_cost = old_position * self.state["entry_price"] + volume * price
                self.state["entry_price"] = total_cost / new_position if new_position != 0 else price
            self.state["current_position"] = new_position
            
        elif action == "sell":
            # Add to short position or reduce long position
            new_position = old_position - volume
            if old_position > 0 and new_position <= 0:
                # Closing long or flipping to short
                if self.state["entry_price"]:
                    # Calculate realized P&L for closed portion
                    closed_volume = min(volume, old_position)
                    pnl = ((price - self.state["entry_price"]) / self.state["entry_price"]) * 100
                    self._update_pnl(pnl)
                
                if new_position < 0:
                    # Flipped to short
                    self.state["entry_price"] = price
                else:
                    # Fully closed
                    self.state["entry_price"] = None
            elif old_position < 0:
                # Adding to short position
                if self.state["entry_price"] is None:
                    self.state["entry_price"] = price
                else:
                    total_cost = abs(old_position) * self.state["entry_price"] + volume * price
                    self.state["entry_price"] = total_cost / abs(new_position) if new_position != 0 else price
            
            self.state["current_position"] = new_position
            
        elif action == "close":
            # Close entire position
            if old_position != 0 and self.state["entry_price"]:
                if old_position > 0:
                    pnl = ((price - self.state["entry_price"]) / self.state["entry_price"]) * 100
                else:
                    pnl = ((self.state["entry_price"] - price) / self.state["entry_price"]) * 100
                self._update_pnl(pnl)
            
            self.state["current_position"] = 0.0
            self.state["entry_price"] = None
        
        self.state["last_trade_time"] = datetime.now().isoformat()
        self.state["last_trade_action"] = action
        self.state["last_trade_volume"] = volume
        self.state["trade_count"] += 1
        
        self._save_state()
        logger.info(f"Recorded trade: {action} {volume} @ {price} | Position: {self.state['current_position']}")
    
    def _update_pnl(self, pnl_pct: float):
        """
        Update P&L and adjust martingale multiplier
        
        Args:
            pnl_pct: Realized P&L percentage
        """
        self.state["total_pnl"] += pnl_pct
        
        # Update peak P&L
        if self.state["total_pnl"] > self.state.get("peak_pnl", 0.0):
            self.state["peak_pnl"] = self.state["total_pnl"]
            logger.info(f"New peak P&L: {self.state['peak_pnl']:.2f}%")
        
        if pnl_pct > 0:
            # Win: reset multiplier and clear cooldown
            self.state["win_count"] += 1
            self.state["consecutive_losses"] = 0
            self.state["current_multiplier"] = 1
            self.state["cooldown_until"] = None  # Clear cooldown on win
            logger.info(f"Win! P&L: +{pnl_pct:.2f}%, resetting multiplier to 1")
        else:
            # Loss: increase multiplier
            self.state["consecutive_losses"] += 1
            new_multiplier = min(2 ** self.state["consecutive_losses"], self.max_multiplier)
            self.state["current_multiplier"] = new_multiplier
            logger.warning(
                f"Loss! P&L: {pnl_pct:.2f}%, increasing multiplier to {new_multiplier} "
                f"(losses: {self.state['consecutive_losses']})"
            )
    
    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Martin trader doesn't need chart generation
        
        Returns:
            (None, None) as no chart is needed
        """
        logger.info("Martin trader mode: skipping chart generation")
        return None, None
    
    def analyze(self, stats: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze using martingale trading strategy
        
        Args:
            stats: Statistics dictionary
            **kwargs: Additional parameters (volume from args)
            
        Returns:
            Trading signal
        """
        current_price = stats.get('current_price')
        args = kwargs.get('args')
        
        return self.generate_signal(
            current_price=current_price,
            stats=stats
        )
    
    def generate_output(self, result: Dict[str, Any], stats: Dict[str, Any], args) -> Dict[str, Any]:
        """
        Generate standardized output for martin trader
        
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
            'close': 0.0,
            'hold': 0.0
        }
        
        return {
            'symbol': result['symbol'],
            'position': action_to_position.get(result['action'], 0.0),
            'confidence': 'high' if result['action'] in ['buy', 'sell', 'close'] else 'low',
            'current_price': result['current_price'],
            'action': result['action'],
            'volume': result['volume'],
            'reasoning': result['reasoning'],
            'analysis_type': 'martin',
            'current_position': result.get('current_position'),
            'pnl_pct': result.get('pnl_pct'),
            'multiplier': result.get('multiplier'),
        }
    
    def generate_signal(
        self,
        current_price: float,
        stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal based on martingale strategy
        
        Args:
            current_price: Current market price
            stats: Optional market statistics for additional context
            
        Returns:
            Trade signal dictionary
        """
        logger.info(
            f"Generating signal for {self.symbol} @ ${current_price} | "
            f"Position: {self.state['current_position']}, Multiplier: {self.state['current_multiplier']}"
        )
        
        # Check if we can trade
        if not self._can_trade():
            return {
                "action": "hold",
                "volume": 0.0,
                "symbol": self.symbol,
                "current_price": current_price,
                "current_position": self.state["current_position"],
                "reasoning": "Minimum trade interval not reached"
            }
        
        # Check cooldown
        if self._is_in_cooldown():
            return {
                "action": "hold",
                "volume": 0.0,
                "symbol": self.symbol,
                "current_price": current_price,
                "current_position": self.state["current_position"],
                "reasoning": "In cooldown period after consecutive losses"
            }
        
        # Calculate current P&L if we have a position
        pnl_pct = self._calculate_pnl(current_price)
        
        # Calculate next volume
        next_volume = self._calculate_next_volume()
        
        # Initialize signal
        signal = {
            "action": "hold",
            "volume": 0.0,
            "symbol": self.symbol,
            "current_price": current_price,
            "current_position": self.state["current_position"],
            "pnl_pct": pnl_pct,
            "multiplier": self.state["current_multiplier"],
            "reasoning": ""
        }
        
        # First trade - establish initial position
        if self.state["current_position"] == 0:
            # Safety check before initial entry
            is_safe, reason = self._check_safety_limits(current_price, self.base_volume)
            if not is_safe:
                signal["reasoning"] = f"Cannot open position: {reason}"
                logger.warning(signal["reasoning"])
                return signal
            
            signal["action"] = "buy"
            signal["volume"] = self.base_volume
            signal["reasoning"] = f"Initial entry at ${current_price:.2f} with base volume {self.base_volume}"
            self._record_trade("buy", current_price, self.base_volume, signal["reasoning"])
            return signal
        
        # Check if we should close position (profit target or stop loss)
        if pnl_pct is not None:
            # Check for forced closure due to safety limits
            is_safe, reason = self._check_safety_limits(current_price, 0)
            if not is_safe and "stop loss" in reason.lower() or "drawdown" in reason.lower():
                # Force close position due to safety limits
                signal["action"] = "close"
                signal["volume"] = abs(self.state["current_position"])
                signal["reasoning"] = f"FORCED CLOSE: {reason}"
                logger.error(signal["reasoning"])
                self.state["emergency_stop_triggered"] = True
                self._record_trade("close", current_price, signal["volume"], signal["reasoning"])
                return signal
            
            # Take profit
            if pnl_pct >= self.profit_target:
                signal["action"] = "close"
                signal["volume"] = abs(self.state["current_position"])
                signal["reasoning"] = (
                    f"Taking profit: P&L {pnl_pct:.2f}% >= target {self.profit_target}% "
                    f"(entry: ${self.state['entry_price']:.2f}, current: ${current_price:.2f})"
                )
                self._record_trade("close", current_price, signal["volume"], signal["reasoning"])
                return signal
            
            # Martingale on loss
            elif pnl_pct <= self.loss_threshold:
                # Safety check before martingale
                is_safe, reason = self._check_safety_limits(current_price, next_volume)
                if not is_safe:
                    signal["reasoning"] = f"Martingale BLOCKED: {reason} (P&L: {pnl_pct:.2f}%)"
                    logger.warning(signal["reasoning"])
                    # Consider emergency close if at hard limits
                    if "stop loss" in reason.lower():
                        signal["action"] = "close"
                        signal["volume"] = abs(self.state["current_position"])
                        signal["reasoning"] = f"Emergency close: {reason}"
                        logger.error(signal["reasoning"])
                        self._record_trade("close", current_price, signal["volume"], signal["reasoning"])
                    return signal
                
                # Double down on losing position
                if self.state["current_position"] > 0:
                    # Long position losing, add more
                    signal["action"] = "buy"
                    signal["volume"] = next_volume
                    signal["reasoning"] = (
                        f"Martingale: P&L {pnl_pct:.2f}% <= {self.loss_threshold}%, "
                        f"adding {next_volume} (multiplier: {self.state['current_multiplier']}x)"
                    )
                    self._record_trade("buy", current_price, next_volume, signal["reasoning"])
                else:
                    # Short position losing, add more
                    signal["action"] = "sell"
                    signal["volume"] = next_volume
                    signal["reasoning"] = (
                        f"Martingale: P&L {pnl_pct:.2f}% <= {self.loss_threshold}%, "
                        f"adding {next_volume} short (multiplier: {self.state['current_multiplier']}x)"
                    )
                    self._record_trade("sell", current_price, next_volume, signal["reasoning"])
                return signal
        
        # Hold position
        signal["reasoning"] = (
            f"Holding position {self.state['current_position']:.4f}, "
            f"P&L: {pnl_pct:.2f}% (entry: ${self.state['entry_price']:.2f})" 
            if pnl_pct is not None else "No position"
        )
        
        logger.info(f"Signal: {signal['action'].upper()} - {signal['reasoning']}")
        
        return signal
